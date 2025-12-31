"""
HyDE (Hypothetical Document Embeddings) - Generate hypothetical answers

Instead of searching with the query, generate a hypothetical answer
and use it for retrieval. Often more effective for semantic search.

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496
"""

from typing import List, Dict, Any, Optional
import logging

from src.llm.openai_client import OpenAIClient
from src.utils.logging_config import RAGLogger


class HyDE:
    """
    Hypothetical Document Embeddings.
    
    Generates a hypothetical answer to the query, then uses
    that answer's embedding for retrieval. This works because:
    1. Answers are semantically closer to relevant documents
    2. Query-document embedding gap is reduced
    3. Better captures answer patterns
    
    Example:
        >>> hyde = HyDE()
        >>> 
        >>> # Generate hypothetical document
        >>> hyp_doc = hyde.generate_document("What is RAG?")
        >>> 
        >>> # Use for retrieval
        >>> results = hyde.retrieve(vector_store, "What is RAG?", top_k=5)
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        model: str = "gpt-4o-mini",
        num_documents: int = 1
    ):
        """
        Initialize HyDE.
        
        Args:
            llm_client: OpenAI client (creates new if None)
            model: Model for document generation
            num_documents: Number of hypothetical documents to generate
        """
        self.llm_client = llm_client or OpenAIClient()
        self.model = model
        self.num_documents = num_documents
        
        self.logger = RAGLogger.get_logger("hyde")
    
    def generate_document(
        self,
        query: str,
        domain: Optional[str] = None,
        style: str = "informative"
    ) -> str:
        """
        Generate hypothetical document for query.
        
        Args:
            query: User query
            domain: Optional domain context (e.g., "medical", "legal")
            style: Generation style ('informative', 'concise', 'detailed')
            
        Returns:
            Hypothetical document text
        """
        # Build system prompt based on style
        if style == "informative":
            style_instruction = "Write a clear, informative paragraph that directly answers the question."
        elif style == "concise":
            style_instruction = "Write a brief, concise answer in 2-3 sentences."
        else:  # detailed
            style_instruction = "Write a comprehensive, detailed explanation with examples."
        
        system_prompt = f"""You are a helpful assistant that generates hypothetical documents.
Given a question, write a document that would answer it.

{style_instruction}

Do not include preambles like "Here's an answer" or "This document explains".
Write ONLY the document content."""
        
        if domain:
            system_prompt += f"\n\nDomain context: {domain}"
        
        try:
            response = self.llm_client.generate(
                prompt=query,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.7,
                max_tokens=300
            )
            
            self.logger.info(f"Generated hypothetical document ({len(response)} chars)")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"HyDE generation failed: {e}")
            # Fallback to original query
            return query
    
    def generate_multiple_documents(
        self,
        query: str,
        num_docs: Optional[int] = None
    ) -> List[str]:
        """
        Generate multiple hypothetical documents.
        
        Args:
            query: User query
            num_docs: Number of documents (uses instance default if None)
            
        Returns:
            List of hypothetical documents
        """
        n = num_docs or self.num_documents
        
        documents = []
        for i in range(n):
            # Vary temperature for diversity
            temp = 0.7 + (i * 0.1)
            
            system_prompt = f"""Generate a hypothetical document that answers the question.
Variation {i+1} of {n} - provide a different perspective or emphasis."""
            
            try:
                response = self.llm_client.generate(
                    prompt=query,
                    system_prompt=system_prompt,
                    model=self.model,
                    temperature=min(temp, 1.0),
                    max_tokens=300
                )
                documents.append(response.strip())
            except Exception as e:
                self.logger.error(f"HyDE generation {i+1} failed: {e}")
        
        return documents
    
    def retrieve(
        self,
        vector_store: Any,
        query: str,
        top_k: int = 5,
        use_multiple: bool = False,
        fusion_method: str = "rrf"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using HyDE.
        
        Args:
            vector_store: Vector store for retrieval
            query: User query
            top_k: Number of documents to retrieve
            use_multiple: Generate and use multiple hypothetical docs
            fusion_method: How to fuse results if using multiple
            
        Returns:
            List of retrieved documents with scores
        """
        if use_multiple:
            # Generate multiple hypothetical documents
            hyp_docs = self.generate_multiple_documents(query)
            
            # Retrieve with each
            all_results = []
            for i, doc in enumerate(hyp_docs):
                try:
                    results = vector_store.similarity_search_with_score(doc, k=top_k)
                    all_results.append(results)
                    self.logger.debug(f"HyDE doc {i+1} retrieved {len(results)} docs")
                except Exception as e:
                    self.logger.error(f"Retrieval with HyDE doc {i+1} failed: {e}")
            
            # Fuse results
            if fusion_method == "rrf":
                return self._reciprocal_rank_fusion(all_results)
            else:
                # Simple concatenation and deduplication
                return self._unique_fusion(all_results, top_k)
        
        else:
            # Single hypothetical document
            hyp_doc = self.generate_document(query)
            
            # Retrieve using hypothetical document
            try:
                results = vector_store.similarity_search_with_score(hyp_doc, k=top_k)
                
                self.logger.info(f"HyDE retrieved {len(results)} documents")
                
                return [
                    {
                        "document": doc,
                        "score": score,
                        "method": "hyde"
                    }
                    for doc, score in results
                ]
            except Exception as e:
                self.logger.error(f"HyDE retrieval failed: {e}")
                return []
    
    def hybrid_retrieve(
        self,
        vector_store: Any,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: Combine query and HyDE.
        
        Args:
            vector_store: Vector store
            query: User query
            top_k: Number of documents
            alpha: Weight for HyDE (1-alpha for query)
                   0.0 = query only, 1.0 = HyDE only, 0.5 = balanced
            
        Returns:
            List of retrieved documents
        """
        # Retrieve with original query
        query_results = vector_store.similarity_search_with_score(query, k=top_k)
        
        # Retrieve with HyDE
        hyde_results = self.retrieve(vector_store, query, top_k)
        
        # Combine scores
        doc_scores = {}
        
        # Add query results
        for doc, score in query_results:
            doc_id = doc.page_content[:100]
            doc_scores[doc_id] = {
                "doc": doc,
                "score": score * (1 - alpha)
            }
        
        # Add HyDE results
        for result in hyde_results:
            doc = result["document"]
            score = result["score"]
            doc_id = doc.page_content[:100]
            
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += score * alpha
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "score": score * alpha
                }
        
        # Sort and return
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return [
            {
                "document": item["doc"],
                "score": item["score"],
                "method": "hybrid"
            }
            for item in sorted_docs
        ]
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[tuple]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """RRF fusion of multiple result lists."""
        doc_scores = {}
        
        for results in result_lists:
            for rank, (doc, score) in enumerate(results, start=1):
                doc_id = doc.page_content[:100]
                
                rrf_score = 1.0 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                
                doc_scores[doc_id]["score"] += rrf_score
        
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [
            {"document": item["doc"], "score": item["score"], "method": "hyde_multi"}
            for item in sorted_docs
        ]
    
    def _unique_fusion(
        self,
        result_lists: List[List[tuple]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Keep unique documents with highest scores."""
        doc_scores = {}
        
        for results in result_lists:
            for doc, score in results:
                doc_id = doc.page_content[:100]
                
                if doc_id not in doc_scores or score > doc_scores[doc_id]["score"]:
                    doc_scores[doc_id] = {"doc": doc, "score": score}
        
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return [
            {"document": item["doc"], "score": item["score"], "method": "hyde_multi"}
            for item in sorted_docs
        ]


if __name__ == "__main__":
    # Example usage
    hyde = HyDE()
    
    # Generate hypothetical document
    query = "What is the capital of France?"
    hyp_doc = hyde.generate_document(query)
    
    print(f"Query: {query}")
    print(f"\nHypothetical Document:\n{hyp_doc}")
    
    # Generate multiple
    hyp_docs = hyde.generate_multiple_documents(query, num_docs=3)
    print(f"\n\nGenerated {len(hyp_docs)} variations")
    for i, doc in enumerate(hyp_docs, 1):
        print(f"\n{i}. {doc[:100]}...")
