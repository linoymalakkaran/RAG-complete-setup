"""
Self-RAG Pattern Implementation

Self-RAG adds self-reflection to the basic RAG pattern:
1. Decides if retrieval is needed
2. Retrieves documents if needed
3. Generates answer
4. Evaluates answer quality
5. Re-generates if quality is low
"""

from typing import Dict, Any, Optional, List
import openai

from src.rag_patterns.basic_rag import BasicRAG
from src.utils.logging_config import setup_logging

logger = setup_logging("rag.self_rag")


class SelfRAG(BasicRAG):
    """
    Self-RAG with reflection capabilities.
    
    Enhancements over Basic RAG:
    1. Retrieval necessity check
    2. Answer quality evaluation
    3. Self-correction loop
    
    This reduces unnecessary retrievals and improves answer quality.
    """
    
    def __init__(
        self,
        *args,
        retrieval_threshold: float = 0.6,
        quality_threshold: float = 0.7,
        max_reflections: int = 2,
        **kwargs
    ):
        """
        Initialize Self-RAG.
        
        Args:
            retrieval_threshold: Confidence threshold for retrieval necessity
            quality_threshold: Minimum acceptable answer quality
            max_reflections: Maximum number of self-correction attempts
        """
        super().__init__(*args, **kwargs)
        self.retrieval_threshold = retrieval_threshold
        self.quality_threshold = quality_threshold
        self.max_reflections = max_reflections
        
        logger.info(
            f"Initialized Self-RAG: retrieval_threshold={retrieval_threshold}, "
            f"quality_threshold={quality_threshold}, max_reflections={max_reflections}"
        )
    
    def query(
        self,
        question: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer question with self-reflection.
        
        Flow:
        1. Check if retrieval is needed
        2. Retrieve if needed, else answer directly
        3. Generate answer
        4. Evaluate answer quality
        5. Re-generate if quality is low
        """
        logger.info(f"Self-RAG processing: {question[:100]}...")
        
        # Step 1: Check if retrieval is needed
        needs_retrieval, retrieval_confidence = self._check_retrieval_necessity(question)
        
        retrieved_docs = []
        if needs_retrieval:
            logger.info(f"Retrieval needed (confidence: {retrieval_confidence:.2f})")
            retrieved_docs = self._retrieve(question, metadata_filter)
        else:
            logger.info(f"Retrieval not needed (confidence: {retrieval_confidence:.2f})")
        
        # Step 2: Generate answer with reflection loop
        for attempt in range(self.max_reflections + 1):
            # Build prompt
            if retrieved_docs:
                prompt = self._build_prompt(question, retrieved_docs)
            else:
                prompt = self._build_direct_prompt(question)
            
            # Generate answer
            answer, gen_metadata = self._generate(prompt)
            
            # Evaluate quality
            quality_score, quality_reasoning = self._evaluate_answer_quality(
                question, answer, retrieved_docs
            )
            
            logger.info(
                f"Answer quality: {quality_score:.2f} "
                f"(attempt {attempt + 1}/{self.max_reflections + 1})"
            )
            
            # Check if quality is acceptable
            if quality_score >= self.quality_threshold:
                logger.info("Answer quality acceptable!")
                break
            elif attempt < self.max_reflections:
                logger.info(f"Quality too low. Reason: {quality_reasoning}. Retrying...")
                # Add quality feedback to next attempt
                prompt = self._add_quality_feedback(prompt, quality_reasoning)
            else:
                logger.warning("Max reflections reached. Using last answer.")
        
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
            'metadata': {
                **gen_metadata,
                'retrieval_needed': needs_retrieval,
                'retrieval_confidence': retrieval_confidence,
                'quality_score': quality_score,
                'quality_reasoning': quality_reasoning,
                'reflection_attempts': attempt + 1
            }
        }
    
    def _check_retrieval_necessity(self, question: str) -> tuple[bool, float]:
        """
        Determine if retrieval is needed for this question.
        
        Uses LLM to decide if external knowledge is required.
        
        Returns:
            (needs_retrieval, confidence)
        """
        prompt = f"""Analyze this question and determine if external document retrieval is needed to answer it.

Question: {question}

Respond in this format:
NEEDS_RETRIEVAL: [yes/no]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]

Consider:
- Does the question ask for specific factual information?
- Can it be answered with general knowledge alone?
- Does it reference company-specific policies or data?
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            needs_retrieval = "yes" in content.split("NEEDS_RETRIEVAL:")[1].split("\n")[0].lower()
            confidence_str = content.split("CONFIDENCE:")[1].split("\n")[0].strip()
            confidence = float(confidence_str)
            
            return needs_retrieval, confidence
            
        except Exception as e:
            logger.error(f"Error checking retrieval necessity: {str(e)}")
            # Default to retrieval on error
            return True, 0.5
    
    def _evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        sources: List[Any]
    ) -> tuple[float, str]:
        """
        Evaluate the quality of generated answer.
        
        Criteria:
        - Relevance to question
        - Grounding in sources
        - Completeness
        - Accuracy
        
        Returns:
            (quality_score, reasoning)
        """
        sources_text = "\n\n".join([s.content for s in sources]) if sources else "No sources used"
        
        prompt = f"""Evaluate this answer's quality on a scale of 0.0 to 1.0.

Question: {question}

Answer: {answer}

Sources used:
{sources_text}

Evaluate based on:
1. Relevance: Does the answer address the question?
2. Grounding: Is the answer supported by the sources?
3. Completeness: Does it fully answer the question?
4. Accuracy: Is the information correct?

Respond in this format:
SCORE: [0.0-1.0]
REASONING: [brief explanation of score]
ISSUES: [any problems found]
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            # Parse score
            score_str = content.split("SCORE:")[1].split("\n")[0].strip()
            score = float(score_str)
            
            # Extract reasoning
            reasoning = content.split("REASONING:")[1].split("ISSUES:")[0].strip()
            
            return score, reasoning
            
        except Exception as e:
            logger.error(f"Error evaluating answer quality: {str(e)}")
            return 0.5, "Evaluation failed"
    
    def _build_direct_prompt(self, question: str) -> str:
        """Build prompt for answering without retrieval"""
        return f"""Answer the following question based on your general knowledge.
If you're not certain, say so.

Question: {question}

Answer:"""
    
    def _add_quality_feedback(self, prompt: str, feedback: str) -> str:
        """Add quality feedback to prompt for re-generation"""
        return f"""{prompt}

Previous attempt had issues: {feedback}
Please address these issues in your new answer."""
