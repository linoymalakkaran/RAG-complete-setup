"""
Evaluation Module

Comprehensive evaluation tools for RAG systems.

Modules:
- ragas_integration: RAGAS framework integration
- retrieval_metrics: Retrieval quality metrics (Precision, Recall, MRR, NDCG)
- response_metrics: Response quality metrics (BLEU, ROUGE, BERT Score)
"""

from src.evaluation.ragas_integration import (
    RAGASEvaluator,
    EvaluationResult,
    create_test_cases_from_rag_results
)

from src.evaluation.retrieval_metrics import (
    RetrievalEvaluator,
    RetrievalMetrics,
    calculate_retrieval_latency,
    calculate_relevance_distribution
)

from src.evaluation.response_metrics import (
    ResponseEvaluator,
    ResponseMetrics,
    calculate_answer_quality
)

__all__ = [
    # RAGAS Integration
    "RAGASEvaluator",
    "EvaluationResult",
    "create_test_cases_from_rag_results",
    
    # Retrieval Metrics
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "calculate_retrieval_latency",
    "calculate_relevance_distribution",
    
    # Response Metrics
    "ResponseEvaluator",
    "ResponseMetrics",
    "calculate_answer_quality",
]
