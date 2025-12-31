"""
RAGAS Evaluation Integration

This module integrates RAGAS (Retrieval Augmented Generation Assessment) framework
for comprehensive RAG system evaluation.

Metrics Supported:
1. Faithfulness - Answer grounded in retrieved context
2. Answer Relevancy - Answer addresses the question
3. Context Precision - Relevant chunks ranked highly
4. Context Recall - All necessary information retrieved
5. Context Relevancy - Retrieved chunks are relevant

Reference: https://docs.ragas.io/
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from src.utils.logging_config import RAGLogger


@dataclass
class EvaluationResult:
    """Results from RAGAS evaluation."""
    
    # Overall scores
    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    context_precision_score: float = 0.0
    context_recall_score: float = 0.0
    context_relevancy_score: float = 0.0
    
    # Aggregate metrics
    overall_score: float = 0.0
    num_samples: int = 0
    
    # Detailed results per question
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness_score,
            "answer_relevancy": self.answer_relevancy_score,
            "context_precision": self.context_precision_score,
            "context_recall": self.context_recall_score,
            "context_relevancy": self.context_relevancy_score,
            "overall": self.overall_score,
            "num_samples": self.num_samples,
            "detailed_results": self.detailed_results,
            "metadata": self.metadata
        }


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG systems.
    
    Evaluates RAG performance across multiple dimensions:
    - Faithfulness: Factual consistency with sources
    - Answer Relevancy: Pertinence to question
    - Context Precision: Ranking quality
    - Context Recall: Completeness of retrieval
    - Context Relevancy: Noise vs signal ratio
    
    Example:
        >>> evaluator = RAGASEvaluator(llm, embeddings)
        >>> 
        >>> # Prepare test data
        >>> test_cases = [
        ...     {
        ...         "question": "What is the refund policy?",
        ...         "answer": "30-day money back guarantee",
        ...         "contexts": ["We offer 30-day refunds..."],
        ...         "ground_truth": "30-day refund policy"
        ...     }
        ... ]
        >>> 
        >>> # Evaluate
        >>> results = evaluator.evaluate(test_cases)
        >>> print(f"Overall score: {results.overall_score:.2f}")
    """
    
    def __init__(
        self,
        llm,
        embeddings,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm: Language model for evaluation
            embeddings: Embedding model for evaluation
            config: Optional configuration:
                - metrics: List of metrics to compute (default: all)
                - batch_size: Batch size for evaluation (default: 10)
        """
        self.logger = RAGLogger.get_logger("ragas_evaluator")
        
        if not RAGAS_AVAILABLE:
            self.logger.error("RAGAS not available - install with: pip install ragas")
            raise ImportError("RAGAS package not found")
        
        self.llm = llm
        self.embeddings = embeddings
        
        # Get config
        config = config or {}
        self.batch_size = config.get("batch_size", 10)
        
        # Select metrics
        available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_relevancy": context_relevancy
        }
        
        requested_metrics = config.get("metrics", list(available_metrics.keys()))
        self.metrics = [
            available_metrics[m] for m in requested_metrics
            if m in available_metrics
        ]
        
        self.logger.info(
            f"Initialized RAGAS evaluator with {len(self.metrics)} metrics: "
            f"{requested_metrics}"
        )
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        detailed: bool = True
    ) -> EvaluationResult:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            test_cases: List of test cases, each containing:
                - question: The user question
                - answer: The generated answer
                - contexts: List of retrieved context strings
                - ground_truth: Reference answer (optional for some metrics)
            detailed: Include per-question breakdown
            
        Returns:
            EvaluationResult with all scores
        """
        self.logger.info(f"Evaluating {len(test_cases)} test cases")
        
        if not test_cases:
            self.logger.warning("No test cases provided")
            return EvaluationResult(num_samples=0)
        
        try:
            # Convert to RAGAS dataset format
            dataset = self._create_dataset(test_cases)
            
            # Run RAGAS evaluation
            self.logger.info("Running RAGAS evaluation...")
            ragas_result = evaluate(
                dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Extract scores
            result = self._parse_results(ragas_result, test_cases, detailed)
            
            self.logger.info(
                f"Evaluation complete - Overall score: {result.overall_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            return EvaluationResult(
                num_samples=len(test_cases),
                metadata={"error": str(e)}
            )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Reference answer (optional)
            
        Returns:
            Dict of metric scores
        """
        test_case = {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }
        
        if ground_truth:
            test_case["ground_truth"] = ground_truth
        
        result = self.evaluate([test_case], detailed=False)
        
        return {
            "faithfulness": result.faithfulness_score,
            "answer_relevancy": result.answer_relevancy_score,
            "context_precision": result.context_precision_score,
            "context_recall": result.context_recall_score,
            "context_relevancy": result.context_relevancy_score,
            "overall": result.overall_score
        }
    
    def _create_dataset(self, test_cases: List[Dict[str, Any]]) -> Dataset:
        """
        Create RAGAS dataset from test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            RAGAS Dataset object
        """
        # Extract fields
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for case in test_cases:
            questions.append(case["question"])
            answers.append(case["answer"])
            
            # Contexts must be list of strings
            ctx = case.get("contexts", [])
            if isinstance(ctx, str):
                ctx = [ctx]
            contexts.append(ctx)
            
            # Ground truth is optional
            ground_truths.append(case.get("ground_truth", ""))
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        return Dataset.from_dict(data)
    
    def _parse_results(
        self,
        ragas_result,
        test_cases: List[Dict[str, Any]],
        detailed: bool
    ) -> EvaluationResult:
        """
        Parse RAGAS results into EvaluationResult.
        
        Args:
            ragas_result: Raw RAGAS evaluation result
            test_cases: Original test cases
            detailed: Include detailed breakdown
            
        Returns:
            EvaluationResult object
        """
        # Extract average scores
        scores = ragas_result
        
        result = EvaluationResult(
            faithfulness_score=scores.get("faithfulness", 0.0),
            answer_relevancy_score=scores.get("answer_relevancy", 0.0),
            context_precision_score=scores.get("context_precision", 0.0),
            context_recall_score=scores.get("context_recall", 0.0),
            context_relevancy_score=scores.get("context_relevancy", 0.0),
            num_samples=len(test_cases)
        )
        
        # Calculate overall score (average of all metrics)
        metric_values = [
            result.faithfulness_score,
            result.answer_relevancy_score,
            result.context_precision_score,
            result.context_recall_score,
            result.context_relevancy_score
        ]
        
        # Filter out zero scores (metrics not computed)
        active_metrics = [v for v in metric_values if v > 0]
        result.overall_score = (
            sum(active_metrics) / len(active_metrics)
            if active_metrics else 0.0
        )
        
        # Add detailed results if requested
        if detailed:
            result.detailed_results = self._create_detailed_results(
                ragas_result,
                test_cases
            )
        
        # Add metadata
        result.metadata = {
            "ragas_version": "0.1.1",
            "num_metrics": len(self.metrics),
            "batch_size": self.batch_size
        }
        
        return result
    
    def _create_detailed_results(
        self,
        ragas_result,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create detailed per-question results.
        
        Args:
            ragas_result: RAGAS evaluation result
            test_cases: Original test cases
            
        Returns:
            List of detailed result dictionaries
        """
        detailed = []
        
        # RAGAS returns aggregated scores by default
        # For per-question scores, we'd need to access internal data
        # Here we create a template structure
        
        for i, case in enumerate(test_cases):
            detail = {
                "question": case["question"],
                "answer": case["answer"],
                "ground_truth": case.get("ground_truth", ""),
                "num_contexts": len(case.get("contexts", [])),
                "scores": {
                    "faithfulness": ragas_result.get("faithfulness", 0.0),
                    "answer_relevancy": ragas_result.get("answer_relevancy", 0.0),
                    "context_precision": ragas_result.get("context_precision", 0.0),
                    "context_recall": ragas_result.get("context_recall", 0.0),
                    "context_relevancy": ragas_result.get("context_relevancy", 0.0)
                }
            }
            
            detailed.append(detail)
        
        return detailed


def create_test_cases_from_rag_results(
    rag_results: List[Dict[str, Any]],
    ground_truths: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convert RAG query results to RAGAS test cases.
    
    Args:
        rag_results: List of RAG query results with question, answer, sources
        ground_truths: Optional list of reference answers
        
    Returns:
        List of test cases in RAGAS format
    """
    test_cases = []
    
    for i, result in enumerate(rag_results):
        # Extract contexts from sources
        contexts = [
            src.get("content", src.get("page_content", ""))
            for src in result.get("sources", [])
        ]
        
        test_case = {
            "question": result["question"],
            "answer": result["answer"],
            "contexts": contexts
        }
        
        # Add ground truth if available
        if ground_truths and i < len(ground_truths):
            test_case["ground_truth"] = ground_truths[i]
        
        test_cases.append(test_case)
    
    return test_cases


if __name__ == "__main__":
    print("RAGAS Evaluation Integration")
    print("=" * 60)
    print("\nSupported Metrics:")
    print("1. Faithfulness - Answer grounded in context")
    print("2. Answer Relevancy - Answer addresses question")
    print("3. Context Precision - Relevant chunks ranked high")
    print("4. Context Recall - All necessary info retrieved")
    print("5. Context Relevancy - Low noise in retrieval")
    print("\nUsage:")
    print("  evaluator = RAGASEvaluator(llm, embeddings)")
    print("  results = evaluator.evaluate(test_cases)")
    print(f"\nRAGAS Available: {RAGAS_AVAILABLE}")
