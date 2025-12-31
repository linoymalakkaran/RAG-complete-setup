"""
Reranking - Improve retrieval relevance with cross-encoder models

Reranks retrieved documents using more sophisticated models:
- Cross-encoder models (bi-encoder retrieval -> cross-encoder rerank)
- LLM-based reranking
- Hybrid scoring
"""

from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from src.llm.openai_client import OpenAIClient
from src.utils.logging_config import RAGLogger


class Reranker(ABC):
    """Base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents."""
        pass


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder based reranking.
    
    Uses a cross-encoder model (e.g., from sentence-transformers)
    to score query-document pairs. More accurate than bi-encoders
    but slower (can't pre-compute).
    
    Note: Requires sentence-transformers with cross-encoder support.
    Falls back to simple scoring if not available.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model from sentence-transformers
        """
        self.model_name = model_name
        self.logger = RAGLogger.get_logger("cross_encoder_reranker")
        
        # Try to load cross-encoder
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
            self.logger.info(f"Loaded cross-encoder: {model_name}")
        except ImportError:
            self.logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
            self.available = False
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: User query
            documents: Retrieved documents
            top_k: Return top K documents (None = all)
            
        Returns:
            Reranked documents with scores
        """
        if not self.available:
            self.logger.warning("Cross-encoder not available, skipping rerank")
            return [
                {"document": doc, "score": 0.5, "method": "no_rerank"}
                for doc in documents
            ]
        
        # Extract document content
        doc_texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                doc_texts.append(doc.page_content)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get('content', doc.get('page_content', '')))
            else:
                doc_texts.append(str(doc))
        
        # Create query-document pairs
        pairs = [[query, doc_text] for doc_text in doc_texts]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Combine documents with scores
        scored_docs = [
            {
                "document": doc,
                "score": float(score),
                "method": "cross_encoder"
            }
            for doc, score in zip(documents, scores)
        ]
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k if specified
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        self.logger.info(f"Reranked {len(documents)} -> {len(scored_docs)} documents")
        return scored_docs


class LLMReranker(Reranker):
    """
    LLM-based reranking.
    
    Uses an LLM to assess relevance of each document to the query.
    More flexible than cross-encoders but slower and more expensive.
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize LLM reranker.
        
        Args:
            llm_client: OpenAI client
            model: Model for reranking
        """
        self.llm_client = llm_client or OpenAIClient()
        self.model = model
        self.logger = RAGLogger.get_logger("llm_reranker")
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank using LLM relevance assessment.
        
        Args:
            query: User query
            documents: Retrieved documents
            top_k: Return top K documents
            
        Returns:
            Reranked documents with scores
        """
        scored_docs = []
        
        for i, doc in enumerate(documents):
            # Extract content
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('page_content', ''))
            else:
                content = str(doc)
            
            # Truncate if too long
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            # Score relevance
            score = self._score_relevance(query, content)
            
            scored_docs.append({
                "document": doc,
                "score": score,
                "method": "llm_rerank"
            })
            
            self.logger.debug(f"Document {i+1}: relevance score = {score:.3f}")
        
        # Sort by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        self.logger.info(f"LLM reranked {len(documents)} -> {len(scored_docs)} documents")
        return scored_docs
    
    def _score_relevance(self, query: str, document: str) -> float:
        """
        Score document relevance to query.
        
        Args:
            query: User query
            document: Document content
            
        Returns:
            Relevance score (0-1)
        """
        system_prompt = """You are a relevance assessment assistant.
Rate how relevant a document is to answering a query.

Respond with ONLY a number from 0 to 10:
- 0: Completely irrelevant
- 5: Somewhat relevant
- 10: Perfectly relevant

Consider:
1. Does the document contain information to answer the query?
2. How directly does it address the query?
3. Is the information accurate and useful?

Respond with ONLY the number, nothing else."""
        
        user_prompt = f"""Query: {query}

Document: {document}

Relevance score (0-10):"""
        
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=10
            )
            
            # Extract number
            score_str = response.strip()
            score = float(score_str)
            
            # Normalize to 0-1
            normalized = score / 10.0
            
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            self.logger.error(f"LLM scoring failed: {e}")
            return 0.5  # Default neutral score


class HybridReranker(Reranker):
    """
    Hybrid reranking combining multiple signals.
    
    Combines:
    - Original retrieval scores
    - Cross-encoder scores
    - LLM scores (optional)
    - Metadata signals (recency, source quality, etc.)
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = True,
        use_llm: bool = False,
        cross_encoder_weight: float = 0.6,
        retrieval_weight: float = 0.3,
        llm_weight: float = 0.1
    ):
        """
        Initialize hybrid reranker.
        
        Args:
            use_cross_encoder: Use cross-encoder scoring
            use_llm: Use LLM scoring (slower but more accurate)
            cross_encoder_weight: Weight for cross-encoder scores
            retrieval_weight: Weight for original retrieval scores
            llm_weight: Weight for LLM scores
        """
        self.use_cross_encoder = use_cross_encoder
        self.use_llm = use_llm
        self.cross_encoder_weight = cross_encoder_weight
        self.retrieval_weight = retrieval_weight
        self.llm_weight = llm_weight
        
        self.logger = RAGLogger.get_logger("hybrid_reranker")
        
        # Initialize rerankers
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
        
        if use_llm:
            self.llm_reranker = LLMReranker()
    
    def rerank(
        self,
        query: str,
        documents: List[tuple],  # (doc, original_score)
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid reranking with multiple signals.
        
        Args:
            query: User query
            documents: List of (document, original_score) tuples
            top_k: Return top K documents
            
        Returns:
            Reranked documents with combined scores
        """
        # Extract docs and scores
        docs = [doc for doc, _ in documents]
        original_scores = [score for _, score in documents]
        
        # Normalize original scores to 0-1
        if original_scores:
            max_score = max(original_scores)
            min_score = min(original_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            normalized_original = [
                (score - min_score) / score_range
                for score in original_scores
            ]
        else:
            normalized_original = [0.5] * len(docs)
        
        # Get cross-encoder scores
        if self.use_cross_encoder and self.cross_encoder.available:
            ce_results = self.cross_encoder.rerank(query, docs)
            ce_scores = [r["score"] for r in ce_results]
        else:
            ce_scores = [0.5] * len(docs)
        
        # Get LLM scores
        if self.use_llm:
            llm_results = self.llm_reranker.rerank(query, docs)
            llm_scores = [r["score"] for r in llm_results]
        else:
            llm_scores = [0.5] * len(docs)
        
        # Combine scores
        combined_scores = []
        for i in range(len(docs)):
            combined = (
                self.retrieval_weight * normalized_original[i] +
                self.cross_encoder_weight * ce_scores[i] +
                self.llm_weight * llm_scores[i]
            )
            combined_scores.append(combined)
        
        # Create scored documents
        scored_docs = [
            {
                "document": doc,
                "score": combined_scores[i],
                "original_score": original_scores[i],
                "cross_encoder_score": ce_scores[i],
                "llm_score": llm_scores[i] if self.use_llm else None,
                "method": "hybrid"
            }
            for i, doc in enumerate(docs)
        ]
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        self.logger.info(f"Hybrid reranked {len(documents)} -> {len(scored_docs)} documents")
        return scored_docs


if __name__ == "__main__":
    # Example usage
    
    # Cross-encoder reranking
    ce_reranker = CrossEncoderReranker()
    
    # Mock documents
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
    
    docs = [
        MockDoc("RAG stands for Retrieval Augmented Generation."),
        MockDoc("Paris is the capital of France."),
        MockDoc("RAG combines retrieval and generation for better AI responses.")
    ]
    
    query = "What is RAG?"
    
    if ce_reranker.available:
        results = ce_reranker.rerank(query, docs, top_k=2)
        
        print("Reranked results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['document'].page_content[:80]}...")
