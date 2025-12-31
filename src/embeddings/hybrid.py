"""
Hybrid retrieval combining dense (vector) and sparse (BM25) search.

Hybrid search improves retrieval by combining:
- Dense retrieval: Semantic similarity (vector search)
- Sparse retrieval: Keyword matching (BM25)

This often outperforms either approach alone.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

from src.utils.logging_config import setup_logging

logger = setup_logging("rag.hybrid_retrieval")


@dataclass
class SearchResult:
    """
    Represents a search result with score.
    
    Attributes:
        doc_id: Document/chunk identifier
        content: Text content
        score: Relevance score
        metadata: Additional metadata
        source: Which retrieval method found this ("dense", "sparse", "hybrid")
    """
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str = "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata,
            'source': self.source
        }


class BM25Retriever:
    """
    BM25 (Best Matching 25) sparse retrieval.
    
    BM25 is a ranking function for keyword-based search.
    It's particularly good at:
        - Exact keyword matching
        - Rare term emphasis
        - Length normalization
    
    Algorithm:
        1. Tokenize documents and query
        2. Calculate IDF (Inverse Document Frequency) for terms
        3. Score documents based on term frequencies and IDF
    
    Parameters:
        - k1: Term frequency saturation (default: 1.5)
        - b: Length normalization (default: 0.75)
    """
    
    def __init__(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadata_list: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            metadata_list: List of metadata dicts
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.documents = documents
        self.doc_ids = doc_ids
        self.metadata_list = metadata_list
        
        # Tokenize documents (simple whitespace tokenization)
        # In production, use better tokenization (nltk, spacy)
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=k1,
            b=b
        )
        
        logger.info(f"Initialized BM25 with {len(documents)} documents (k1={k1}, b={b})")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append(SearchResult(
                    doc_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(scores[idx]),
                    metadata=self.metadata_list[idx],
                    source="sparse"
                ))
        
        logger.debug(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
        return results


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse search.
    
    Approach:
        1. Perform dense search (vector similarity)
        2. Perform sparse search (BM25)
        3. Combine results with weighted scores
        4. Rerank and deduplicate
    
    Combining strategies:
        - Weighted sum: score = α * dense_score + (1-α) * sparse_score
        - Reciprocal Rank Fusion (RRF): Combines rankings instead of scores
        - Learned weights: Train weights for your specific use case
    """
    
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion_method: str = "weighted_sum"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
            fusion_method: "weighted_sum" or "rrf" (Reciprocal Rank Fusion)
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_method = fusion_method
        
        # Validate weights
        if abs(dense_weight + sparse_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights don't sum to 1.0: {dense_weight} + {sparse_weight} = "
                f"{dense_weight + sparse_weight}. Normalizing..."
            )
            total = dense_weight + sparse_weight
            self.dense_weight = dense_weight / total
            self.sparse_weight = sparse_weight / total
        
        logger.info(
            f"Initialized HybridRetriever: dense={self.dense_weight:.2f}, "
            f"sparse={self.sparse_weight:.2f}, method={fusion_method}"
        )
    
    def combine_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Combine dense and sparse search results.
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from BM25
            top_k: Number of final results
            
        Returns:
            Combined and ranked results
        """
        if self.fusion_method == "weighted_sum":
            return self._weighted_sum_fusion(dense_results, sparse_results, top_k)
        elif self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_sum_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine using weighted sum of normalized scores.
        
        Steps:
            1. Normalize scores to 0-1 range for each method
            2. Combine scores: α * dense + (1-α) * sparse
            3. Sort by combined score
        """
        # Create score dictionaries
        dense_scores = {}
        sparse_scores = {}
        all_docs = {}
        
        # Normalize dense scores
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            for result in dense_results:
                norm_score = result.score / max_dense if max_dense > 0 else 0
                dense_scores[result.doc_id] = norm_score
                all_docs[result.doc_id] = result
        
        # Normalize sparse scores
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results)
            for result in sparse_results:
                norm_score = result.score / max_sparse if max_sparse > 0 else 0
                sparse_scores[result.doc_id] = norm_score
                if result.doc_id not in all_docs:
                    all_docs[result.doc_id] = result
        
        # Combine scores
        combined_results = []
        for doc_id, doc in all_docs.items():
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)
            
            combined_score = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
            
            combined_results.append(SearchResult(
                doc_id=doc.doc_id,
                content=doc.content,
                score=combined_score,
                metadata={
                    **doc.metadata,
                    'dense_score': dense_score,
                    'sparse_score': sparse_score
                },
                source="hybrid"
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(
            f"Weighted sum fusion: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(combined_results[:top_k])} combined"
        )
        
        return combined_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
        k: int = 60
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF).
        
        RRF formula: RRF(d) = Σ 1 / (k + rank(d))
        
        Advantages:
            - No need for score normalization
            - Handles different score ranges well
            - Simple and effective
        
        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            top_k: Number of results
            k: RRF parameter (default: 60)
        """
        rrf_scores = {}
        all_docs = {}
        
        # Add dense rankings
        for rank, result in enumerate(dense_results, 1):
            rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0) + 1 / (k + rank)
            all_docs[result.doc_id] = result
        
        # Add sparse rankings
        for rank, result in enumerate(sparse_results, 1):
            rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0) + 1 / (k + rank)
            if result.doc_id not in all_docs:
                all_docs[result.doc_id] = result
        
        # Create combined results
        combined_results = []
        for doc_id, rrf_score in rrf_scores.items():
            doc = all_docs[doc_id]
            combined_results.append(SearchResult(
                doc_id=doc.doc_id,
                content=doc.content,
                score=rrf_score,
                metadata={**doc.metadata, 'rrf_score': rrf_score},
                source="hybrid"
            ))
        
        # Sort by RRF score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"RRF fusion: {len(combined_results[:top_k])} results")
        
        return combined_results[:top_k]


def create_hybrid_retriever(
    documents: List[str],
    doc_ids: List[str],
    metadata_list: List[Dict[str, Any]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3
) -> Tuple[BM25Retriever, HybridRetriever]:
    """
    Convenience function to create both BM25 and hybrid retrievers.
    
    Args:
        documents: List of document texts
        doc_ids: List of document IDs
        metadata_list: List of metadata
        dense_weight: Weight for dense retrieval
        sparse_weight: Weight for sparse retrieval
        
    Returns:
        Tuple of (BM25Retriever, HybridRetriever)
        
    Example:
        >>> bm25, hybrid = create_hybrid_retriever(docs, ids, metadata)
        >>> # Use with vector search
        >>> dense_results = vector_search(query)
        >>> sparse_results = bm25.search(query)
        >>> final_results = hybrid.combine_results(dense_results, sparse_results)
    """
    bm25 = BM25Retriever(documents, doc_ids, metadata_list)
    hybrid = HybridRetriever(dense_weight, sparse_weight)
    
    return bm25, hybrid
