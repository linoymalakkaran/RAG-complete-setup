"""
Corrective RAG (CRAG) Pattern

This module implements Corrective RAG, which evaluates retrieval quality
and falls back to web search when internal documents are insufficient.

The CRAG pattern follows these steps:
1. Retrieve documents from vector store (as in Basic RAG)
2. Evaluate retrieval quality using an LLM
3. If quality is LOW → trigger web search for external knowledge
4. If quality is AMBIGUOUS → combine internal + external sources
5. If quality is HIGH → use only internal documents
6. Generate answer with corrected/augmented context

Reference: https://arxiv.org/abs/2401.15884
"""

from typing import List, Dict, Any, Optional, Literal
import logging
from enum import Enum

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage

from src.rag_patterns.basic_rag import BasicRAG
from src.utils.logging_config import RAGLogger


class RetrievalQuality(Enum):
    """Enum for retrieval quality levels."""
    HIGH = "high"
    AMBIGUOUS = "ambiguous"
    LOW = "low"


class CorrectiveRAG(BasicRAG):
    """
    Corrective RAG implementation with web search fallback.
    
    This pattern improves upon Basic RAG by:
    - Evaluating retrieval quality before generation
    - Triggering web search for out-of-domain queries
    - Combining internal and external knowledge when needed
    - Providing more robust answers for difficult questions
    
    Example:
        >>> crag = CorrectiveRAG(
        ...     vector_store=chroma_client,
        ...     llm_client=openai_client,
        ...     enable_web_search=True
        ... )
        >>> result = crag.query("What is the latest Python release?")
        >>> print(result['answer'])
        >>> print(f"Used web search: {result['metadata']['used_web_search']}")
    """
    
    def __init__(
        self,
        vector_store,
        llm_client,
        embedding_model=None,
        enable_hybrid: bool = False,
        enable_web_search: bool = True,
        quality_threshold_high: float = 0.7,
        quality_threshold_low: float = 0.3,
        max_web_results: int = 3,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Corrective RAG.
        
        Args:
            vector_store: Vector database client (ChromaDB, FAISS, etc.)
            llm_client: Language model client (OpenAI, etc.)
            embedding_model: Embedding model for retrieval
            enable_hybrid: Whether to use hybrid search
            enable_web_search: Whether to enable web search fallback
            quality_threshold_high: Score above which quality is considered HIGH
            quality_threshold_low: Score below which quality is considered LOW
            max_web_results: Maximum number of web search results
            logger: Optional logger instance
        """
        super().__init__(
            vector_store=vector_store,
            llm_client=llm_client,
            embedding_model=embedding_model,
            enable_hybrid=enable_hybrid,
            logger=logger
        )
        
        self.enable_web_search = enable_web_search
        self.quality_threshold_high = quality_threshold_high
        self.quality_threshold_low = quality_threshold_low
        self.max_web_results = max_web_results
        
        # Initialize web search
        if self.enable_web_search:
            try:
                self.web_search = DuckDuckGoSearchAPIWrapper()
                self.logger.info("Web search enabled with DuckDuckGo")
            except Exception as e:
                self.logger.warning(f"Failed to initialize web search: {e}")
                self.web_search = None
                self.enable_web_search = False
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Corrective RAG query with quality evaluation and web search fallback.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            filters: Optional metadata filters for retrieval
            temperature: LLM temperature for generation
            **kwargs: Additional arguments
            
        Returns:
            Dict containing:
                - answer: Generated answer
                - sources: Retrieved documents with scores
                - metadata: Query metadata including:
                    - retrieval_quality: Quality assessment
                    - used_web_search: Whether web search was used
                    - web_results: Web search results if used
                    - correction_applied: Whether correction was needed
        """
        self.logger.info(f"Corrective RAG query: {question[:100]}...")
        
        # Step 1: Retrieve from vector store
        retrieved_docs = self._retrieve(question, top_k, filters)
        
        # Step 2: Evaluate retrieval quality
        quality_assessment = self._evaluate_retrieval_quality(
            question=question,
            documents=retrieved_docs
        )
        
        quality_level = quality_assessment['quality']
        self.logger.info(f"Retrieval quality: {quality_level.value} (score: {quality_assessment['score']:.3f})")
        
        # Step 3: Apply correction based on quality
        corrected_docs = retrieved_docs
        used_web_search = False
        web_results = []
        
        if quality_level == RetrievalQuality.LOW:
            # Low quality → Use web search only
            if self.enable_web_search and self.web_search:
                self.logger.info("Low quality retrieval - using web search")
                web_results = self._perform_web_search(question)
                corrected_docs = web_results
                used_web_search = True
            else:
                self.logger.warning("Low quality but web search disabled - using original docs")
        
        elif quality_level == RetrievalQuality.AMBIGUOUS:
            # Ambiguous → Combine internal + external
            if self.enable_web_search and self.web_search:
                self.logger.info("Ambiguous quality - combining internal + web sources")
                web_results = self._perform_web_search(question)
                # Combine: prioritize high-scoring internal docs + web results
                corrected_docs = self._combine_sources(retrieved_docs, web_results)
                used_web_search = True
        
        else:  # HIGH quality
            # High quality → Use internal docs only
            self.logger.info("High quality retrieval - using internal documents")
            corrected_docs = retrieved_docs
        
        # Step 4: Build prompt with corrected context
        prompt = self._build_prompt(question, corrected_docs)
        
        # Step 5: Generate answer
        answer = self._generate(prompt, temperature)
        
        # Prepare result
        result = {
            'answer': answer,
            'sources': [
                {
                    'content': doc['content'],
                    'score': doc.get('score', 0.0),
                    'metadata': doc.get('metadata', {}),
                    'source_type': doc.get('source_type', 'internal')
                }
                for doc in corrected_docs
            ],
            'metadata': {
                'retrieval_quality': quality_level.value,
                'quality_score': quality_assessment['score'],
                'quality_reasoning': quality_assessment['reasoning'],
                'used_web_search': used_web_search,
                'web_results_count': len(web_results),
                'correction_applied': quality_level != RetrievalQuality.HIGH,
                'total_sources': len(corrected_docs),
                'model': 'corrective_rag',
                'temperature': temperature
            }
        }
        
        self.logger.info(
            f"CRAG completed - Quality: {quality_level.value}, "
            f"Web search: {used_web_search}, Sources: {len(corrected_docs)}"
        )
        
        return result
    
    def _evaluate_retrieval_quality(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieved documents for the given question.
        
        Uses an LLM to assess whether the retrieved documents contain
        sufficient information to answer the question.
        
        Args:
            question: User's question
            documents: Retrieved documents
            
        Returns:
            Dict with:
                - quality: RetrievalQuality enum value
                - score: Numerical quality score (0-1)
                - reasoning: Explanation of the assessment
        """
        if not documents:
            return {
                'quality': RetrievalQuality.LOW,
                'score': 0.0,
                'reasoning': 'No documents retrieved'
            }
        
        # Build evaluation prompt
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc['content'][:500]}"
            for i, doc in enumerate(documents[:3])  # Evaluate top 3
        ])
        
        eval_prompt = f"""You are a retrieval quality evaluator. Assess whether the retrieved documents contain sufficient information to answer the user's question.

Question: {question}

Retrieved Documents:
{docs_text}

Evaluation Criteria:
- HIGH: Documents directly contain the answer or highly relevant information
- AMBIGUOUS: Documents contain some relevant info but may be incomplete
- LOW: Documents don't contain relevant information for this question

Provide your assessment in this exact format:
QUALITY: [HIGH/AMBIGUOUS/LOW]
SCORE: [0.0-1.0]
REASONING: [Brief explanation]"""
        
        try:
            # Call LLM for evaluation
            messages = [
                SystemMessage(content="You are a precise retrieval quality evaluator."),
                HumanMessage(content=eval_prompt)
            ]
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for evaluation
                messages=[{"role": m.type, "content": m.content} for m in messages],
                temperature=0.0,  # Low temperature for consistent evaluation
                max_tokens=200
            )
            
            eval_text = response.choices[0].message.content.strip()
            
            # Parse response
            quality_line = [l for l in eval_text.split('\n') if l.startswith('QUALITY:')]
            score_line = [l for l in eval_text.split('\n') if l.startswith('SCORE:')]
            reasoning_line = [l for l in eval_text.split('\n') if l.startswith('REASONING:')]
            
            # Extract quality level
            quality_str = quality_line[0].split(':')[1].strip().upper() if quality_line else 'AMBIGUOUS'
            quality = RetrievalQuality[quality_str] if quality_str in ['HIGH', 'AMBIGUOUS', 'LOW'] else RetrievalQuality.AMBIGUOUS
            
            # Extract score
            try:
                score = float(score_line[0].split(':')[1].strip()) if score_line else 0.5
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except:
                score = 0.5
            
            # Map score to quality if needed
            if score >= self.quality_threshold_high:
                quality = RetrievalQuality.HIGH
            elif score <= self.quality_threshold_low:
                quality = RetrievalQuality.LOW
            
            reasoning = reasoning_line[0].split(':')[1].strip() if reasoning_line else 'Quality assessed'
            
            return {
                'quality': quality,
                'score': score,
                'reasoning': reasoning
            }
        
        except Exception as e:
            self.logger.error(f"Quality evaluation failed: {e}")
            # Fallback: Use document scores as proxy
            avg_score = sum(doc.get('score', 0) for doc in documents) / len(documents)
            if avg_score >= self.quality_threshold_high:
                quality = RetrievalQuality.HIGH
            elif avg_score <= self.quality_threshold_low:
                quality = RetrievalQuality.LOW
            else:
                quality = RetrievalQuality.AMBIGUOUS
            
            return {
                'quality': quality,
                'score': avg_score,
                'reasoning': f'Fallback evaluation based on retrieval scores (avg: {avg_score:.3f})'
            }
    
    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search for external knowledge.
        
        Args:
            query: Search query
            
        Returns:
            List of web search results formatted as documents
        """
        if not self.web_search:
            self.logger.warning("Web search not available")
            return []
        
        try:
            # Perform search
            results = self.web_search.results(query, max_results=self.max_web_results)
            
            # Format as documents
            web_docs = []
            for i, result in enumerate(results):
                web_docs.append({
                    'content': f"{result.get('title', '')}\n\n{result.get('snippet', '')}",
                    'score': 1.0 - (i * 0.1),  # Decreasing scores
                    'metadata': {
                        'source': 'web_search',
                        'url': result.get('link', ''),
                        'title': result.get('title', '')
                    },
                    'source_type': 'web'
                })
            
            self.logger.info(f"Retrieved {len(web_docs)} web search results")
            return web_docs
        
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []
    
    def _combine_sources(
        self,
        internal_docs: List[Dict[str, Any]],
        web_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Intelligently combine internal and web sources.
        
        Strategy:
        - Keep high-scoring internal documents (score > 0.7)
        - Add top web search results
        - Limit total to reasonable number
        
        Args:
            internal_docs: Documents from vector store
            web_docs: Documents from web search
            
        Returns:
            Combined list of documents
        """
        combined = []
        
        # Add high-quality internal docs
        for doc in internal_docs:
            if doc.get('score', 0) > 0.7:
                combined.append(doc)
        
        # Add web docs
        combined.extend(web_docs[:2])  # Top 2 web results
        
        # If we have very few, add more internal docs
        if len(combined) < 3:
            for doc in internal_docs:
                if doc not in combined and len(combined) < 5:
                    combined.append(doc)
        
        self.logger.info(
            f"Combined sources: {sum(1 for d in combined if d.get('source_type') == 'internal')} internal, "
            f"{sum(1 for d in combined if d.get('source_type') == 'web')} web"
        )
        
        return combined
