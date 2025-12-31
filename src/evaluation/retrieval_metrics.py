"""
Retrieval Metrics

Metrics for evaluating retrieval quality in RAG systems.
Includes precision, recall, MRR, NDCG, and custom metrics.
"""

from typing import List, Dict, Any, Optional, Set
import logging
from dataclasses import dataclass
import numpy as np

from src.utils.logging_config import RAGLogger


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    
    # Basic metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Ranking metrics
    mean_reciprocal_rank: float = 0.0  # MRR
    mean_average_precision: float = 0.0  # MAP
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    # Coverage metrics
    hit_rate: float = 0.0
    coverage: float = 0.0
    
    # Diversity metrics
    diversity: float = 0.0
    
    # Metadata
    num_queries: int = 0
    avg_retrieved: float = 0.0
    avg_relevant: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mrr": self.mean_reciprocal_rank,
            "map": self.mean_average_precision,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "coverage": self.coverage,
            "diversity": self.diversity,
            "num_queries": self.num_queries,
            "avg_retrieved": self.avg_retrieved,
            "avg_relevant": self.avg_relevant
        }


class RetrievalEvaluator:
    """
    Evaluator for retrieval quality metrics.
    
    Computes various metrics to assess how well the retrieval
    component finds relevant documents.
    
    Example:
        >>> evaluator = RetrievalEvaluator()
        >>> 
        >>> # Define relevant doc IDs for each query
        >>> test_cases = [
        ...     {
        ...         "query": "What is Python?",
        ...         "retrieved": ["doc1", "doc2", "doc5"],
        ...         "relevant": ["doc1", "doc2", "doc3"]
        ...     }
        ... ]
        >>> 
        >>> metrics = evaluator.evaluate(test_cases)
        >>> print(f"Precision: {metrics.precision:.3f}")
        >>> print(f"Recall: {metrics.recall:.3f}")
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize retrieval evaluator.
        
        Args:
            k: Cutoff for precision@k and recall@k
        """
        self.k = k
        self.logger = RAGLogger.get_logger("retrieval_evaluator")
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            test_cases: List of test cases, each with:
                - query: The search query
                - retrieved: List of retrieved document IDs
                - relevant: List of relevant document IDs
                
        Returns:
            RetrievalMetrics object with all metrics
        """
        self.logger.info(f"Evaluating {len(test_cases)} retrieval test cases")
        
        if not test_cases:
            return RetrievalMetrics()
        
        # Calculate all metrics
        precisions = []
        recalls = []
        f1_scores = []
        reciprocal_ranks = []
        avg_precisions = []
        ndcgs = []
        hits = []
        
        total_retrieved = 0
        total_relevant = 0
        all_relevant_docs = set()
        retrieved_docs = set()
        
        for case in test_cases:
            retrieved = case.get("retrieved", [])[:self.k]
            relevant = set(case.get("relevant", []))
            
            # Track for coverage
            all_relevant_docs.update(relevant)
            retrieved_docs.update(retrieved)
            
            total_retrieved += len(retrieved)
            total_relevant += len(relevant)
            
            # Calculate metrics for this case
            p, r, f1 = self._precision_recall_f1(retrieved, relevant)
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            
            reciprocal_ranks.append(self._reciprocal_rank(retrieved, relevant))
            avg_precisions.append(self._average_precision(retrieved, relevant))
            ndcgs.append(self._ndcg(retrieved, relevant))
            hits.append(1.0 if len(set(retrieved) & relevant) > 0 else 0.0)
        
        # Aggregate metrics
        metrics = RetrievalMetrics(
            precision=np.mean(precisions),
            recall=np.mean(recalls),
            f1_score=np.mean(f1_scores),
            mean_reciprocal_rank=np.mean(reciprocal_ranks),
            mean_average_precision=np.mean(avg_precisions),
            ndcg=np.mean(ndcgs),
            hit_rate=np.mean(hits),
            coverage=len(retrieved_docs & all_relevant_docs) / len(all_relevant_docs)
                     if all_relevant_docs else 0.0,
            diversity=len(retrieved_docs) / (total_retrieved or 1),
            num_queries=len(test_cases),
            avg_retrieved=total_retrieved / len(test_cases),
            avg_relevant=total_relevant / len(test_cases)
        )
        
        self.logger.info(
            f"Evaluation complete - Precision: {metrics.precision:.3f}, "
            f"Recall: {metrics.recall:.3f}, MRR: {metrics.mean_reciprocal_rank:.3f}"
        )
        
        return metrics
    
    def _precision_recall_f1(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, and F1."""
        if not retrieved:
            return 0.0, 0.0, 0.0
        
        retrieved_set = set(retrieved)
        true_positives = len(retrieved_set & relevant)
        
        precision = true_positives / len(retrieved_set)
        recall = true_positives / len(relevant) if relevant else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    def _reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """Calculate reciprocal rank - 1/rank of first relevant doc."""
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0
    
    def _average_precision(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """Calculate average precision."""
        if not relevant:
            return 0.0
        
        precisions_at_k = []
        num_relevant_found = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                precisions_at_k.append(precision_at_i)
        
        if not precisions_at_k:
            return 0.0
        
        return sum(precisions_at_k) / len(relevant)
    
    def _ndcg(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: Optional[int] = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if k is None:
            k = self.k
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate ideal DCG (IDCG)
        idcg = sum(
            1.0 / np.log2(i + 1)
            for i in range(1, min(len(relevant), k) + 1)
        )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


def calculate_retrieval_latency(
    retrieval_times: List[float]
) -> Dict[str, float]:
    """
    Calculate latency statistics for retrieval.
    
    Args:
        retrieval_times: List of retrieval times in milliseconds
        
    Returns:
        Dict with latency statistics
    """
    if not retrieval_times:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    
    times = np.array(retrieval_times)
    
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "min": float(np.min(times)),
        "max": float(np.max(times))
    }


def calculate_relevance_distribution(
    relevance_scores: List[List[float]]
) -> Dict[str, Any]:
    """
    Analyze distribution of relevance scores.
    
    Args:
        relevance_scores: List of score lists (one per query)
        
    Returns:
        Dict with distribution statistics
    """
    if not relevance_scores:
        return {}
    
    all_scores = [score for scores in relevance_scores for score in scores]
    
    if not all_scores:
        return {}
    
    scores = np.array(all_scores)
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
        "num_scores": len(all_scores)
    }


if __name__ == "__main__":
    print("Retrieval Metrics")
    print("=" * 60)
    print("\nSupported Metrics:")
    print("1. Precision@K - Fraction of retrieved docs that are relevant")
    print("2. Recall@K - Fraction of relevant docs that are retrieved")
    print("3. F1 Score - Harmonic mean of precision and recall")
    print("4. MRR - Mean Reciprocal Rank")
    print("5. MAP - Mean Average Precision")
    print("6. NDCG - Normalized Discounted Cumulative Gain")
    print("7. Hit Rate - Fraction of queries with at least one relevant doc")
    print("8. Coverage - Fraction of relevant docs retrieved overall")
