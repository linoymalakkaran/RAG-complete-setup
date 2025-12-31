"""
RAG Orchestrator - Integrated pipeline with all enhancements

Orchestrates the complete RAG pipeline:
- Query enhancement (multi-query, HyDE, expansion)
- Context management (conversation memory, window management)
- Retrieval (hybrid search, reranking)
- Generation with context
- Evaluation
"""

from typing import List, Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum

from src.context import ConversationMemory, ConversationBuffer, WindowManager
from src.query_enhancement import (
    LLMQueryGenerator, HyDE, HybridReranker, QueryExpander
)
from src.llm.openai_client import OpenAIClient
from src.utils.logging_config import RAGLogger


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SIMPLE = "simple"  # Basic similarity search
    MULTI_QUERY = "multi_query"  # Multi-query generation
    HYDE = "hyde"  # Hypothetical Document Embeddings
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class RAGConfig:
    """Configuration for RAG orchestrator."""
    
    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMPLE
    top_k: int = 5
    use_reranking: bool = True
    use_query_expansion: bool = False
    
    # Context settings
    use_conversation_memory: bool = True
    max_context_tokens: int = 4000
    reserve_tokens: int = 1000
    
    # Generation settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Multi-query settings
    num_queries: int = 3
    fusion_method: str = "rrf"
    
    # HyDE settings
    use_multiple_hyde: bool = False
    hyde_num_docs: int = 1
    
    # Reranking settings
    reranker_type: str = "hybrid"  # 'cross_encoder', 'llm', 'hybrid'


class RAGOrchestrator:
    """
    Complete RAG pipeline orchestrator.
    
    Integrates all RAG components into a unified pipeline:
    1. Query enhancement
    2. Context retrieval
    3. Reranking
    4. Context management
    5. Response generation
    6. Conversation tracking
    
    Example:
        >>> config = RAGConfig(
        >>>     retrieval_strategy=RetrievalStrategy.MULTI_QUERY,
        >>>     use_reranking=True,
        >>>     use_conversation_memory=True
        >>> )
        >>> orchestrator = RAGOrchestrator(vector_store, config)
        >>> 
        >>> # Single query
        >>> response = orchestrator.query(
        >>>     "What is RAG?",
        >>>     conversation_id="conv_123"
        >>> )
        >>> 
        >>> # Multi-turn conversation
        >>> response1 = orchestrator.query("What is RAG?", conversation_id="conv_1")
        >>> response2 = orchestrator.query("How does it work?", conversation_id="conv_1")
    """
    
    def __init__(
        self,
        vector_store: Any,
        config: Optional[RAGConfig] = None,
        llm_client: Optional[OpenAIClient] = None
    ):
        """
        Initialize RAG orchestrator.
        
        Args:
            vector_store: Vector store for retrieval
            config: RAG configuration
            llm_client: OpenAI client (creates new if None)
        """
        self.vector_store = vector_store
        self.config = config or RAGConfig()
        self.llm_client = llm_client or OpenAIClient()
        
        self.logger = RAGLogger.get_logger("rag_orchestrator")
        
        # Initialize components
        self._init_components()
        
        self.logger.info(
            f"Initialized RAG orchestrator "
            f"(strategy={self.config.retrieval_strategy.value}, "
            f"reranking={self.config.use_reranking})"
        )
    
    def _init_components(self):
        """Initialize all RAG components."""
        
        # Context management
        if self.config.use_conversation_memory:
            self.memory = ConversationMemory()
            self.buffer = ConversationBuffer(self.memory)
            self.window_manager = WindowManager(
                max_tokens=self.config.max_context_tokens,
                reserve_tokens=self.config.reserve_tokens
            )
        else:
            self.memory = None
            self.buffer = None
            self.window_manager = None
        
        # Query enhancement
        self.query_generator = LLMQueryGenerator(self.llm_client)
        self.hyde = HyDE(self.llm_client)
        
        if self.config.use_query_expansion:
            self.query_expander = QueryExpander(self.llm_client)
        else:
            self.query_expander = None
        
        # Reranking
        if self.config.use_reranking:
            if self.config.reranker_type == "cross_encoder":
                from src.query_enhancement import CrossEncoderReranker
                self.reranker = CrossEncoderReranker()
            elif self.config.reranker_type == "llm":
                from src.query_enhancement import LLMReranker
                self.reranker = LLMReranker(self.llm_client)
            else:  # hybrid
                from src.query_enhancement import HybridReranker
                self.reranker = HybridReranker()
        else:
            self.reranker = None
    
    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete RAG query.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for multi-turn
            metadata: Optional metadata for tracking
            stream: Whether to stream response (not yet implemented)
            
        Returns:
            Dict with:
                - answer: Generated response
                - sources: Retrieved documents
                - conversation_id: Conversation ID
                - metadata: Processing metadata
        """
        self.logger.info(f"Processing query: {query[:50]}...")
        
        # Create conversation if needed
        if conversation_id is None and self.config.use_conversation_memory:
            conversation_id = self.memory.create_conversation()
        
        # Step 1: Query enhancement
        enhanced_query = self._enhance_query(query)
        
        # Step 2: Retrieve documents
        retrieved_docs = self._retrieve_documents(enhanced_query)
        
        # Step 3: Rerank if enabled
        if self.config.use_reranking and self.reranker:
            reranked = self._rerank_documents(enhanced_query, retrieved_docs)
        else:
            reranked = retrieved_docs
        
        # Step 4: Build context with window management
        context = self._build_context(
            enhanced_query,
            reranked,
            conversation_id
        )
        
        # Step 5: Generate response
        answer = self._generate_response(
            enhanced_query,
            context,
            stream=stream
        )
        
        # Step 6: Update conversation memory
        if conversation_id and self.memory:
            self.memory.add_message(conversation_id, "user", query)
            self.memory.add_message(conversation_id, "assistant", answer)
        
        # Prepare response
        response = {
            "answer": answer,
            "sources": self._format_sources(reranked[:self.config.top_k]),
            "conversation_id": conversation_id,
            "metadata": {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "num_documents_retrieved": len(retrieved_docs),
                "num_documents_used": len(reranked[:self.config.top_k]),
                "reranking_used": self.config.use_reranking,
                **(metadata or {})
            }
        }
        
        self.logger.info("Query completed successfully")
        return response
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with expansion if enabled."""
        if self.config.use_query_expansion and self.query_expander:
            enhanced = self.query_expander.expand(query)
            self.logger.debug(f"Enhanced query: {enhanced[:80]}...")
            return enhanced
        return query
    
    def _retrieve_documents(self, query: str) -> List[tuple]:
        """
        Retrieve documents using configured strategy.
        
        Returns:
            List of (document, score) tuples
        """
        strategy = self.config.retrieval_strategy
        
        if strategy == RetrievalStrategy.SIMPLE:
            # Simple similarity search
            results = self.vector_store.similarity_search_with_score(
                query,
                k=self.config.top_k * 2  # Get more for reranking
            )
            
        elif strategy == RetrievalStrategy.MULTI_QUERY:
            # Multi-query generation
            queries = self.query_generator.generate(
                query,
                num_queries=self.config.num_queries
            )
            
            # Retrieve with each query
            all_results = []
            for q in queries:
                results = self.vector_store.similarity_search_with_score(
                    q,
                    k=self.config.top_k
                )
                all_results.append(results)
            
            # Fuse results
            results = self._fuse_results(all_results)
            
        elif strategy == RetrievalStrategy.HYDE:
            # HyDE retrieval
            if self.config.use_multiple_hyde:
                # Multiple hypothetical docs
                hyp_docs = self.hyde.generate_multiple_documents(
                    query,
                    num_docs=self.config.hyde_num_docs
                )
                
                all_results = []
                for hyp_doc in hyp_docs:
                    results = self.vector_store.similarity_search_with_score(
                        hyp_doc,
                        k=self.config.top_k
                    )
                    all_results.append(results)
                
                results = self._fuse_results(all_results)
            else:
                # Single hypothetical doc
                hyp_doc = self.hyde.generate_document(query)
                results = self.vector_store.similarity_search_with_score(
                    hyp_doc,
                    k=self.config.top_k * 2
                )
            
        else:  # HYBRID
            # Combine simple + multi-query
            simple_results = self.vector_store.similarity_search_with_score(
                query,
                k=self.config.top_k
            )
            
            queries = self.query_generator.generate(query, num_queries=2)
            multi_results = []
            for q in queries:
                r = self.vector_store.similarity_search_with_score(q, k=self.config.top_k)
                multi_results.append(r)
            
            results = self._fuse_results([simple_results] + multi_results)
        
        self.logger.debug(f"Retrieved {len(results)} documents")
        return results
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[tuple]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using configured reranker."""
        reranked = self.reranker.rerank(
            query,
            documents,
            top_k=self.config.top_k * 2
        )
        
        self.logger.debug(f"Reranked to {len(reranked)} documents")
        return reranked
    
    def _build_context(
        self,
        query: str,
        documents: List[Any],
        conversation_id: Optional[str]
    ) -> str:
        """Build context with window management."""
        
        if not self.window_manager:
            # Simple context building
            doc_texts = []
            for i, doc_item in enumerate(documents[:self.config.top_k]):
                if isinstance(doc_item, dict):
                    doc = doc_item["document"]
                else:
                    doc, _ = doc_item
                
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                doc_texts.append(f"Document {i+1}:\n{content}")
            
            return "\n\n".join(doc_texts)
        
        # Advanced context management with window manager
        self.window_manager.clear()
        
        # Add system prompt
        system_prompt = "You are a helpful AI assistant. Use the provided documents to answer the user's question."
        self.window_manager.add_system_prompt(system_prompt)
        
        # Add documents
        doc_list = []
        for doc_item in documents[:self.config.top_k]:
            if isinstance(doc_item, dict):
                doc = doc_item["document"]
                score = doc_item.get("score", 0.0)
            else:
                doc, score = doc_item
            
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            doc_list.append({
                "content": content,
                "metadata": metadata
            })
        
        # Add to window manager with relevance scores
        relevance_scores = [
            doc_item.get("score", 0.0) if isinstance(doc_item, dict) else doc_item[1]
            for doc_item in documents[:self.config.top_k]
        ]
        
        self.window_manager.add_document_context(doc_list, relevance_scores)
        
        # Add conversation history if available
        if conversation_id and self.memory:
            messages = self.memory.get_messages(conversation_id)
            if messages:
                self.window_manager.add_conversation_context(messages)
        
        # Build final context
        result = self.window_manager.build_context(strategy="balanced")
        
        self.logger.debug(
            f"Built context: {result['tokens_used']} tokens, "
            f"{result['items_selected']}/{result['items_total']} items"
        )
        
        return result["context"]
    
    def _generate_response(
        self,
        query: str,
        context: str,
        stream: bool = False
    ) -> str:
        """Generate response using LLM."""
        
        prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
        
        system_prompt = """You are a helpful AI assistant. Answer based on the provided context.
If the context doesn't contain enough information, say so clearly.
Include relevant citations when possible."""
        
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.strip()
    
    def _fuse_results(self, result_lists: List[List[tuple]]) -> List[tuple]:
        """Fuse multiple result lists using RRF."""
        doc_scores = {}
        k = 60
        
        for results in result_lists:
            for rank, (doc, score) in enumerate(results, start=1):
                doc_id = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                
                rrf_score = 1.0 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                
                doc_scores[doc_id]["score"] += rrf_score
        
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [(item["doc"], item["score"]) for item in sorted_docs]
    
    def _format_sources(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Format documents as sources."""
        sources = []
        
        for i, doc_item in enumerate(documents):
            if isinstance(doc_item, dict):
                doc = doc_item["document"]
                score = doc_item.get("score", 0.0)
            else:
                doc, score = doc_item
            
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            sources.append({
                "index": i + 1,
                "content": content[:500] + "..." if len(content) > 500 else content,
                "score": float(score),
                "metadata": metadata
            })
        
        return sources
    
    def get_conversation_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if not self.memory:
            return []
        
        messages = self.memory.get_messages(conversation_id)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]


if __name__ == "__main__":
    # Example usage
    print("RAG Orchestrator - Example Usage\n")
    
    # Create config
    config = RAGConfig(
        retrieval_strategy=RetrievalStrategy.MULTI_QUERY,
        use_reranking=True,
        use_conversation_memory=True,
        num_queries=3
    )
    
    print(f"Config: {config}")
    print("\nNote: Requires vector_store to be initialized")
    print("Example:")
    print("  orchestrator = RAGOrchestrator(vector_store, config)")
    print("  response = orchestrator.query('What is RAG?')")
