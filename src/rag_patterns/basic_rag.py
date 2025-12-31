"""
Basic RAG implementation - the foundation pattern.

Retrieval Augmented Generation:
1. Retrieve relevant documents based on query
2. Augment the prompt with retrieved context
3. Generate response using LLM
"""

from typing import List, Dict, Any, Optional
import openai

from src.vectordb.chromadb_client import ChromaDBClient
from src.embeddings.hybrid import HybridRetriever, SearchResult, BM25Retriever
from src.utils.logging_config import setup_logging

logger = setup_logging("rag.basic")


class BasicRAG:
    """
    Basic RAG pattern implementation.
    
    Flow:
        User Query → Embed Query → Vector Search → Retrieve Top-K →
        Build Prompt → LLM Generate → Return Answer
    
    This is the simplest RAG pattern and serves as the baseline.
    """
    
    def __init__(
        self,
        vectordb: ChromaDBClient,
        llm_client: openai.OpenAI,
        model: str = "gpt-4-turbo-preview",
        top_k: int = 5,
        use_hybrid: bool = False,
        bm25_retriever: Optional[BM25Retriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None
    ):
        """
        Initialize Basic RAG.
        
        Args:
            vectordb: Vector database client
            llm_client: OpenAI client
            model: LLM model name
            top_k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid retrieval
            bm25_retriever: BM25 retriever (if use_hybrid)
            hybrid_retriever: Hybrid retriever (if use_hybrid)
        """
        self.vectordb = vectordb
        self.llm_client = llm_client
        self.model = model
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        
        logger.info(f"Initialized BasicRAG: model={model}, top_k={top_k}, hybrid={use_hybrid}")
    
    def query(
        self,
        question: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            metadata_filter: Optional metadata filter
            
        Returns:
            Dictionary containing:
                - answer: Generated answer
                - sources: Retrieved source documents
                - metadata: Additional metadata (tokens used, etc.)
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self._retrieve(question, metadata_filter)
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt(question, retrieved_docs)
        
        # Step 3: Generate answer
        answer, metadata = self._generate(prompt)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': doc.score
                }
                for doc in retrieved_docs
            ],
            'metadata': metadata
        }
    
    def _retrieve(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents.
        
        Uses either pure vector search or hybrid search.
        """
        if self.use_hybrid and self.bm25_retriever and self.hybrid_retriever:
            # Hybrid retrieval
            logger.debug("Using hybrid retrieval")
            
            # Dense search
            dense_results = self.vectordb.search(query, self.top_k, metadata_filter)
            dense_search_results = [
                SearchResult(
                    doc_id=dense_results['ids'][0][i],
                    content=dense_results['documents'][0][i],
                    score=1 - dense_results['distances'][0][i],  # Convert distance to similarity
                    metadata=dense_results['metadatas'][0][i],
                    source="dense"
                )
                for i in range(len(dense_results['ids'][0]))
            ]
            
            # Sparse search
            sparse_search_results = self.bm25_retriever.search(query, self.top_k)
            
            # Combine
            retrieved = self.hybrid_retriever.combine_results(
                dense_search_results,
                sparse_search_results,
                self.top_k
            )
        else:
            # Pure vector search
            logger.debug("Using vector search")
            results = self.vectordb.search(query, self.top_k, metadata_filter)
            
            retrieved = [
                SearchResult(
                    doc_id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    score=1 - results['distances'][0][i],
                    metadata=results['metadatas'][0][i],
                    source="dense"
                )
                for i in range(len(results['ids'][0]))
            ]
        
        logger.info(f"Retrieved {len(retrieved)} documents")
        return retrieved
    
    def _build_prompt(
        self,
        question: str,
        documents: List[SearchResult]
    ) -> str:
        """
        Build prompt with retrieved context.
        
        Format:
            Context: [Retrieved documents]
            Question: [User question]
            Answer: [To be generated]
        """
        # Format context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}]\n{doc.content}\n")
        
        context = "\n".join(context_parts)
        
        # Build full prompt
        prompt = f"""You are a helpful assistant answering questions about company policies and documentation.

Use the following context to answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _generate(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        """
        Generate answer using LLM.
        
        Returns:
            Tuple of (answer, metadata)
        """
        logger.debug("Generating answer with LLM")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            metadata = {
                'model': self.model,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            
            logger.info(f"Generated answer ({metadata['tokens_used']} tokens)")
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise


def create_basic_rag(
    vectordb: ChromaDBClient,
    model: str = "gpt-4-turbo-preview",
    top_k: int = 5,
    use_hybrid: bool = False
) -> BasicRAG:
    """
    Convenience function to create BasicRAG instance.
    
    Example:
        >>> from src.vectordb.chromadb_client import ChromaDBClient
        >>> vectordb = ChromaDBClient()
        >>> rag = create_basic_rag(vectordb, use_hybrid=True)
        >>> result = rag.query("What is the vacation policy?")
        >>> print(result['answer'])
    """
    import os
    llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    return BasicRAG(
        vectordb=vectordb,
        llm_client=llm_client,
        model=model,
        top_k=top_k,
        use_hybrid=use_hybrid
    )
