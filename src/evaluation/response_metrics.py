"""
Response Metrics

Metrics for evaluating generated response quality in RAG systems.
Includes BLEU, ROUGE, BERT Score, semantic similarity, and custom metrics.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

from src.utils.logging_config import RAGLogger


@dataclass
class ResponseMetrics:
    """Container for response quality metrics."""
    
    # Lexical overlap metrics
    bleu_score: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    
    # Semantic similarity
    bert_score_f1: float = 0.0
    semantic_similarity: float = 0.0
    
    # Response characteristics
    avg_length: float = 0.0
    length_ratio: float = 0.0  # answer/reference length
    
    # Quality indicators
    has_answer: float = 0.0  # Fraction with non-empty answers
    exact_match: float = 0.0  # Exact string match rate
    
    # Metadata
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bleu": self.bleu_score,
            "rouge1_f1": self.rouge1_f1,
            "rouge2_f1": self.rouge2_f1,
            "rougeL_f1": self.rougeL_f1,
            "bert_score_f1": self.bert_score_f1,
            "semantic_similarity": self.semantic_similarity,
            "avg_length": self.avg_length,
            "length_ratio": self.length_ratio,
            "has_answer": self.has_answer,
            "exact_match": self.exact_match,
            "num_samples": self.num_samples
        }


class ResponseEvaluator:
    """
    Evaluator for response quality metrics.
    
    Computes various metrics to assess the quality of generated answers
    compared to reference answers.
    
    Example:
        >>> evaluator = ResponseEvaluator()
        >>> 
        >>> test_cases = [
        ...     {
        ...         "answer": "Paris is the capital of France.",
        ...         "reference": "The capital of France is Paris."
        ...     }
        ... ]
        >>> 
        >>> metrics = evaluator.evaluate(test_cases)
        >>> print(f"BLEU: {metrics.bleu_score:.3f}")
        >>> print(f"ROUGE-L: {metrics.rougeL_f1:.3f}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize response evaluator.
        
        Args:
            config: Optional configuration:
                - use_bert_score: Enable BERT Score (default: True if available)
                - bert_model: Model for BERT Score (default: roberta-large)
        """
        self.logger = RAGLogger.get_logger("response_evaluator")
        
        config = config or {}
        self.use_bert_score = config.get("use_bert_score", BERT_SCORE_AVAILABLE)
        self.bert_model = config.get("bert_model", "roberta-large")
        
        # Initialize ROUGE scorer
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        else:
            self.logger.warning("NLTK not available - some metrics will be disabled")
            self.rouge_scorer = None
        
        self.logger.info(
            f"Initialized response evaluator (BERT Score: {self.use_bert_score})"
        )
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> ResponseMetrics:
        """
        Evaluate response quality.
        
        Args:
            test_cases: List of test cases, each with:
                - answer: Generated answer
                - reference: Ground truth reference answer
                
        Returns:
            ResponseMetrics object with all metrics
        """
        self.logger.info(f"Evaluating {len(test_cases)} response test cases")
        
        if not test_cases:
            return ResponseMetrics()
        
        # Extract answers and references
        answers = [case.get("answer", "") for case in test_cases]
        references = [case.get("reference", "") for case in test_cases]
        
        # Calculate lexical metrics
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for answer, reference in zip(answers, references):
            if NLTK_AVAILABLE and self.rouge_scorer:
                # BLEU
                bleu = self._calculate_bleu(answer, reference)
                bleu_scores.append(bleu)
                
                # ROUGE
                rouge = self.rouge_scorer.score(reference, answer)
                rouge1_scores.append(rouge['rouge1'].fmeasure)
                rouge2_scores.append(rouge['rouge2'].fmeasure)
                rougeL_scores.append(rouge['rougeL'].fmeasure)
        
        # Calculate BERT Score if enabled
        bert_f1 = 0.0
        if self.use_bert_score and BERT_SCORE_AVAILABLE and references:
            bert_f1 = self._calculate_bert_score_batch(answers, references)
        
        # Calculate response characteristics
        answer_lengths = [len(ans.split()) for ans in answers]
        ref_lengths = [len(ref.split()) for ref in references]
        
        length_ratios = [
            al / rl if rl > 0 else 0.0
            for al, rl in zip(answer_lengths, ref_lengths)
        ]
        
        has_answer_rate = sum(1 for ans in answers if ans.strip()) / len(answers)
        exact_match_rate = sum(
            1 for ans, ref in zip(answers, references)
            if ans.strip().lower() == ref.strip().lower()
        ) / len(answers)
        
        # Aggregate metrics
        metrics = ResponseMetrics(
            bleu_score=np.mean(bleu_scores) if bleu_scores else 0.0,
            rouge1_f1=np.mean(rouge1_scores) if rouge1_scores else 0.0,
            rouge2_f1=np.mean(rouge2_scores) if rouge2_scores else 0.0,
            rougeL_f1=np.mean(rougeL_scores) if rougeL_scores else 0.0,
            bert_score_f1=bert_f1,
            semantic_similarity=bert_f1,  # Use BERT Score as semantic sim
            avg_length=np.mean(answer_lengths),
            length_ratio=np.mean(length_ratios),
            has_answer=has_answer_rate,
            exact_match=exact_match_rate,
            num_samples=len(test_cases)
        )
        
        self.logger.info(
            f"Evaluation complete - BLEU: {metrics.bleu_score:.3f}, "
            f"ROUGE-L: {metrics.rougeL_f1:.3f}, BERT F1: {metrics.bert_score_f1:.3f}"
        )
        
        return metrics
    
    def _calculate_bleu(self, candidate: str, reference: str) -> float:
        """Calculate BLEU score for single pair."""
        if not candidate or not reference:
            return 0.0
        
        try:
            # Tokenize
            candidate_tokens = candidate.split()
            reference_tokens = [reference.split()]  # List of references
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                reference_tokens,
                candidate_tokens,
                smoothing_function=smoothing
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating BLEU: {str(e)}")
            return 0.0
    
    def _calculate_bert_score_batch(
        self,
        candidates: List[str],
        references: List[str]
    ) -> float:
        """Calculate average BERT Score for batch."""
        try:
            # Filter out empty strings
            valid_pairs = [
                (c, r) for c, r in zip(candidates, references)
                if c.strip() and r.strip()
            ]
            
            if not valid_pairs:
                return 0.0
            
            cands, refs = zip(*valid_pairs)
            
            # Calculate BERT Score
            P, R, F1 = bert_score_fn(
                list(cands),
                list(refs),
                model_type=self.bert_model,
                verbose=False
            )
            
            # Return mean F1
            return float(F1.mean())
            
        except Exception as e:
            self.logger.error(f"Error calculating BERT Score: {str(e)}")
            return 0.0


def calculate_answer_quality(
    answers: List[str],
    contexts: List[List[str]]
) -> Dict[str, Any]:
    """
    Analyze answer quality characteristics.
    
    Args:
        answers: List of generated answers
        contexts: List of context lists used for each answer
        
    Returns:
        Dict with quality indicators
    """
    if not answers:
        return {}
    
    # Length statistics
    lengths = [len(ans.split()) for ans in answers]
    
    # Check for common issues
    too_short = sum(1 for l in lengths if l < 5) / len(lengths)
    too_long = sum(1 for l in lengths if l > 200) / len(lengths)
    empty = sum(1 for ans in answers if not ans.strip()) / len(lengths)
    
    # Repetition check (simple heuristic)
    repetitive = []
    for ans in answers:
        words = ans.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            repetitive.append(unique_ratio < 0.5)
        else:
            repetitive.append(False)
    
    repetition_rate = sum(repetitive) / len(repetitive) if repetitive else 0.0
    
    # Context usage (answers should reference context)
    uses_context = []
    for ans, ctx_list in zip(answers, contexts):
        if not ctx_list:
            uses_context.append(False)
            continue
        
        # Check if answer contains words from context
        ans_words = set(ans.lower().split())
        ctx_words = set()
        for ctx in ctx_list:
            ctx_words.update(ctx.lower().split())
        
        overlap = len(ans_words & ctx_words)
        uses_context.append(overlap > 3)  # At least 3 words from context
    
    context_usage_rate = sum(uses_context) / len(uses_context)
    
    return {
        "avg_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "too_short_rate": too_short,
        "too_long_rate": too_long,
        "empty_rate": empty,
        "repetition_rate": repetition_rate,
        "context_usage_rate": context_usage_rate,
        "num_samples": len(answers)
    }


if __name__ == "__main__":
    print("Response Metrics")
    print("=" * 60)
    print("\nSupported Metrics:")
    print("1. BLEU - Lexical overlap with reference")
    print("2. ROUGE-1/2/L - N-gram overlap metrics")
    print("3. BERT Score - Semantic similarity using embeddings")
    print("4. Exact Match - Exact string match rate")
    print("5. Length Ratio - Answer/reference length ratio")
    print("\nQuality Indicators:")
    print("- Answer completeness")
    print("- Repetition detection")
    print("- Context usage")
    print("- Length appropriateness")
    print(f"\nNLTK Available: {NLTK_AVAILABLE}")
    print(f"BERT Score Available: {BERT_SCORE_AVAILABLE}")
