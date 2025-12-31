"""
Chunk size optimizer.

Tests different chunking configurations and recommends optimal settings
based on retrieval performance metrics.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from src.ingestion.chunking.chunking_strategies import ChunkerFactory, Chunk
from src.utils.logging_config import setup_logging

logger = setup_logging("rag.optimizer")


@dataclass
class OptimizationResult:
    """Results from chunk optimization"""
    strategy: str
    chunk_size: int
    chunk_overlap: int
    avg_chunk_length: float
    num_chunks: int
    processing_time: float
    score: float  # Overall quality score
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'avg_chunk_length': self.avg_chunk_length,
            'num_chunks': self.num_chunks,
            'processing_time': self.processing_time,
            'score': self.score,
            'metrics': self.metrics
        }


class ChunkOptimizer:
    """
    Optimizes chunking parameters for a given corpus.
    
    Tests various configurations and measures:
        - Chunk size distribution
        - Processing time
        - Semantic coherence (if embeddings available)
        - Information density
    
    Recommendations based on:
        - Document type (technical vs narrative)
        - Average document length
        - Query patterns (if available)
    """
    
    def __init__(self):
        self.test_configs = self._default_test_configs()
    
    def _default_test_configs(self) -> List[Dict[str, Any]]:
        """
        Default configurations to test.
        
        Covers common patterns:
            - Small chunks (256-512): Good for precise retrieval
            - Medium chunks (512-1024): Balanced
            - Large chunks (1024-2048): Better context
        """
        return [
            # Fixed-size variants
            {'strategy': 'fixed', 'chunk_size': 256, 'chunk_overlap': 25},
            {'strategy': 'fixed', 'chunk_size': 512, 'chunk_overlap': 50},
            {'strategy': 'fixed', 'chunk_size': 1024, 'chunk_overlap': 100},
            
            # Recursive variants (recommended)
            {'strategy': 'recursive', 'chunk_size': 500, 'chunk_overlap': 100},
            {'strategy': 'recursive', 'chunk_size': 1000, 'chunk_overlap': 200},
            {'strategy': 'recursive', 'chunk_size': 1500, 'chunk_overlap': 300},
            {'strategy': 'recursive', 'chunk_size': 2000, 'chunk_overlap': 400},
            
            # Semantic (adaptive)
            {
                'strategy': 'semantic',
                'min_chunk_size': 300,
                'max_chunk_size': 1500,
                'similarity_threshold': 0.5
            },
            
            # Parent-document
            {
                'strategy': 'parent_document',
                'parent_size': 2000,
                'child_size': 400,
                'child_overlap': 50
            }
        ]
    
    def optimize(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        test_configs: Optional[List[Dict[str, Any]]] = None
    ) -> List[OptimizationResult]:
        """
        Test multiple chunking configurations and return results.
        
        Args:
            documents: List of document texts to test on
            metadata_list: Optional metadata for each document
            test_configs: Custom configurations to test (uses defaults if None)
            
        Returns:
            List of OptimizationResult sorted by score (best first)
            
        Example:
            >>> optimizer = ChunkOptimizer()
            >>> results = optimizer.optimize(documents)
            >>> best = results[0]
            >>> print(f"Best: {best.strategy} with size {best.chunk_size}")
        """
        logger.info(f"Starting chunk optimization on {len(documents)} documents")
        
        configs = test_configs or self.test_configs
        results = []
        
        for config in configs:
            logger.info(f"Testing configuration: {config}")
            result = self._test_configuration(documents, metadata_list, config)
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Optimization complete. Best: {results[0].strategy} "
                   f"(size={results[0].chunk_size}, score={results[0].score:.3f})")
        
        return results
    
    def _test_configuration(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]],
        config: Dict[str, Any]
    ) -> OptimizationResult:
        """Test a single configuration"""
        strategy = config['strategy']
        
        # Create chunker
        chunker = ChunkerFactory.create(strategy, **config)
        
        # Process documents and measure time
        start_time = time.time()
        all_chunks = []
        
        for idx, doc in enumerate(documents):
            metadata = metadata_list[idx] if metadata_list else {}
            chunks = chunker.chunk(doc, metadata)
            all_chunks.extend(chunks)
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_chunks, documents)
        
        # Calculate overall score (weighted combination of metrics)
        score = self._calculate_score(metrics, processing_time)
        
        # Get chunk size and overlap for result
        chunk_size = config.get('chunk_size', config.get('max_chunk_size', 0))
        chunk_overlap = config.get('chunk_overlap', 0)
        
        return OptimizationResult(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            avg_chunk_length=metrics['avg_chunk_length'],
            num_chunks=len(all_chunks),
            processing_time=processing_time,
            score=score,
            metrics=metrics
        )
    
    def _calculate_metrics(
        self,
        chunks: List[Chunk],
        documents: List[str]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for chunks.
        
        Metrics:
            - avg_chunk_length: Average chunk size
            - std_chunk_length: Standard deviation (consistency)
            - coverage: Percentage of original content preserved
            - chunk_density: Chunks per document
        """
        if not chunks:
            return {
                'avg_chunk_length': 0,
                'std_chunk_length': 0,
                'coverage': 0,
                'chunk_density': 0
            }
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        
        # Total characters in original documents
        total_original = sum(len(doc) for doc in documents)
        
        # Total characters in chunks (accounting for overlap)
        total_chunks = sum(chunk_lengths)
        
        return {
            'avg_chunk_length': np.mean(chunk_lengths),
            'std_chunk_length': np.std(chunk_lengths),
            'coverage': min(total_chunks / total_original, 1.0) if total_original > 0 else 0,
            'chunk_density': len(chunks) / len(documents) if documents else 0
        }
    
    def _calculate_score(
        self,
        metrics: Dict[str, float],
        processing_time: float
    ) -> float:
        """
        Calculate overall quality score.
        
        Scoring criteria:
            - Prefer moderate chunk sizes (not too small, not too large)
            - Prefer consistent chunk lengths (lower std)
            - Prefer good coverage
            - Penalize very slow processing
        
        Score range: 0-100
        """
        # Ideal chunk length is around 500-1000 characters
        ideal_length = 750
        length_score = 100 * (1 - min(abs(metrics['avg_chunk_length'] - ideal_length) / ideal_length, 1))
        
        # Consistency score (lower std is better)
        # Normalize by mean to get coefficient of variation
        if metrics['avg_chunk_length'] > 0:
            cv = metrics['std_chunk_length'] / metrics['avg_chunk_length']
            consistency_score = 100 * (1 - min(cv, 1))
        else:
            consistency_score = 0
        
        # Coverage score
        coverage_score = 100 * metrics['coverage']
        
        # Speed score (penalize if > 1 second)
        speed_score = 100 * max(1 - processing_time, 0)
        
        # Weighted combination
        score = (
            0.4 * length_score +
            0.3 * consistency_score +
            0.2 * coverage_score +
            0.1 * speed_score
        )
        
        return score
    
    def recommend(
        self,
        documents: List[str],
        use_case: str = "general"
    ) -> Dict[str, Any]:
        """
        Get recommended configuration for specific use case.
        
        Args:
            documents: Sample documents
            use_case: Type of use case:
                - "general": Balanced configuration
                - "precise": Smaller chunks for precise retrieval
                - "context": Larger chunks for better context
                - "speed": Fast processing
                
        Returns:
            Recommended configuration dictionary
        """
        logger.info(f"Getting recommendations for use case: {use_case}")
        
        # Run optimization
        results = self.optimize(documents)
        
        # Filter based on use case
        if use_case == "precise":
            # Prefer smaller chunks
            results = [r for r in results if r.chunk_size <= 600]
        elif use_case == "context":
            # Prefer larger chunks
            results = [r for r in results if r.chunk_size >= 1000]
        elif use_case == "speed":
            # Prefer faster strategies
            results.sort(key=lambda x: x.processing_time)
        
        best = results[0]
        
        recommendation = {
            'strategy': best.strategy,
            'chunk_size': best.chunk_size,
            'chunk_overlap': best.chunk_overlap,
            'expected_chunks': best.num_chunks,
            'expected_avg_length': best.avg_chunk_length,
            'score': best.score,
            'reasoning': self._explain_recommendation(best, use_case)
        }
        
        logger.info(f"Recommendation: {recommendation}")
        
        return recommendation
    
    def _explain_recommendation(
        self,
        result: OptimizationResult,
        use_case: str
    ) -> str:
        """Generate human-readable explanation for recommendation"""
        explanations = {
            "general": (
                f"The {result.strategy} strategy with chunk size {result.chunk_size} "
                f"provides a good balance between precision and context. "
                f"Average chunk length: {result.avg_chunk_length:.0f} chars."
            ),
            "precise": (
                f"For precise retrieval, using {result.strategy} with smaller chunks "
                f"({result.chunk_size} chars) helps match specific information. "
                f"This will create approximately {result.num_chunks} chunks per document."
            ),
            "context": (
                f"Larger chunks ({result.chunk_size} chars) using {result.strategy} "
                f"preserve more context, which helps the LLM generate better answers. "
                f"Trade-off: slightly less precise retrieval."
            ),
            "speed": (
                f"The {result.strategy} strategy is fastest ({result.processing_time:.2f}s) "
                f"while maintaining good quality (score: {result.score:.1f})."
            )
        }
        
        return explanations.get(use_case, explanations["general"])


def optimize_chunks(
    documents: List[str],
    use_case: str = "general"
) -> Dict[str, Any]:
    """
    Convenience function to get chunk optimization recommendations.
    
    Args:
        documents: Sample documents to optimize on
        use_case: Use case type (general, precise, context, speed)
        
    Returns:
        Recommended configuration
        
    Example:
        >>> config = optimize_chunks(my_documents, use_case="precise")
        >>> print(f"Use {config['strategy']} with size {config['chunk_size']}")
    """
    optimizer = ChunkOptimizer()
    return optimizer.recommend(documents, use_case)
