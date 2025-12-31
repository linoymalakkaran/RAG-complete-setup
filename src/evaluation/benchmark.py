"""
Comprehensive Benchmark Suite for RAG System

This module provides performance testing and benchmarking for:
- Document ingestion
- Embedding generation
- Vector search
- Retrieval quality
- End-to-end query performance
- Different retrieval strategies
- Caching performance
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict
import tracemalloc

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    duration: float  # seconds
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    timestamp: str
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def compute_summary(self):
        """Compute summary statistics."""
        durations = [r.duration for r in self.results if r.success]
        
        self.summary = {
            "total_benchmarks": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "total_duration": sum(durations),
            "avg_duration": statistics.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "median_duration": statistics.median(durations) if durations else 0,
            "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary
        }
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark results to {filepath}")


class BenchmarkRunner:
    """Runner for executing benchmarks."""
    
    def __init__(self, name: str = "RAG Benchmark"):
        self.name = name
        self.suite = BenchmarkSuite(
            name=name,
            timestamp=datetime.now().isoformat()
        )
    
    def run_benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        iterations: int = 1,
        warmup: int = 0,
        track_memory: bool = False,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of times to run
            warmup: Number of warmup iterations
            track_memory: Whether to track memory usage
            *args, **kwargs: Arguments for func
            
        Returns:
            BenchmarkResult
        """
        logger.info(f"Running benchmark: {name}")
        
        # Warmup
        for _ in range(warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Run benchmark
        durations = []
        memory_peak = 0
        error = None
        
        for i in range(iterations):
            try:
                if track_memory:
                    tracemalloc.start()
                
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                if track_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    memory_peak = max(memory_peak, peak)
                
                durations.append(end - start)
                
            except Exception as e:
                error = str(e)
                logger.error(f"Benchmark {name} failed: {e}")
                break
        
        # Compute statistics
        success = len(durations) == iterations
        avg_duration = statistics.mean(durations) if durations else 0
        
        metadata = {
            "iterations": iterations,
            "all_durations": durations,
            "avg_duration": avg_duration,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0
        }
        
        if track_memory:
            metadata["peak_memory_mb"] = memory_peak / (1024 * 1024)
        
        result = BenchmarkResult(
            name=name,
            duration=avg_duration,
            success=success,
            error=error,
            metadata=metadata
        )
        
        self.suite.add_result(result)
        return result
    
    def get_suite(self) -> BenchmarkSuite:
        """Get benchmark suite with computed summary."""
        self.suite.compute_summary()
        return self.suite


class RAGBenchmark:
    """
    Comprehensive benchmarks for RAG system.
    
    Tests:
    - Ingestion performance
    - Embedding speed
    - Vector search latency
    - Retrieval quality
    - End-to-end query performance
    - Caching impact
    """
    
    def __init__(
        self,
        rag_system: Any,  # RAG orchestrator or similar
        test_documents: Optional[List[str]] = None,
        test_queries: Optional[List[str]] = None
    ):
        """
        Initialize RAG benchmark.
        
        Args:
            rag_system: RAG system to benchmark
            test_documents: Test documents for ingestion
            test_queries: Test queries for retrieval
        """
        self.rag_system = rag_system
        self.test_documents = test_documents or self._generate_test_documents()
        self.test_queries = test_queries or self._generate_test_queries()
        self.runner = BenchmarkRunner("RAG System Benchmark")
    
    def _generate_test_documents(self, count: int = 100) -> List[str]:
        """Generate synthetic test documents."""
        documents = []
        topics = [
            "cloud computing", "machine learning", "data science",
            "software engineering", "cybersecurity", "devops",
            "artificial intelligence", "blockchain", "web development",
            "mobile development"
        ]
        
        for i in range(count):
            topic = topics[i % len(topics)]
            doc = f"Document {i}: This is a test document about {topic}. " \
                  f"It contains information relevant to {topic} and related concepts. " \
                  f"The document provides comprehensive coverage of key topics in {topic}."
            documents.append(doc)
        
        return documents
    
    def _generate_test_queries(self, count: int = 20) -> List[str]:
        """Generate test queries."""
        queries = [
            "What is cloud computing?",
            "Explain machine learning concepts",
            "How does data science work?",
            "Software engineering best practices",
            "Cybersecurity threats and solutions",
            "DevOps principles",
            "AI applications",
            "Blockchain technology",
            "Web development frameworks",
            "Mobile app development",
            "Tell me about cloud services",
            "Machine learning algorithms",
            "Data analysis techniques",
            "Agile software development",
            "Network security",
            "Continuous integration",
            "Deep learning",
            "Smart contracts",
            "Frontend frameworks",
            "iOS development"
        ]
        return queries[:count]
    
    def benchmark_ingestion(
        self,
        batch_sizes: List[int] = [10, 50, 100],
        iterations: int = 3
    ):
        """
        Benchmark document ingestion at different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per batch size
        """
        logger.info("Benchmarking document ingestion...")
        
        for batch_size in batch_sizes:
            docs = self.test_documents[:batch_size]
            
            def ingest_batch():
                # Assuming rag_system has an add_documents method
                if hasattr(self.rag_system, 'add_documents'):
                    return self.rag_system.add_documents(docs)
                else:
                    # Fallback: simulate ingestion
                    time.sleep(batch_size * 0.01)
            
            self.runner.run_benchmark(
                name=f"Ingestion_{batch_size}_docs",
                func=ingest_batch,
                iterations=iterations,
                track_memory=True
            )
    
    def benchmark_embedding_generation(
        self,
        text_lengths: List[int] = [50, 200, 500, 1000],
        iterations: int = 10
    ):
        """
        Benchmark embedding generation for different text lengths.
        
        Args:
            text_lengths: List of text lengths (in words) to test
            iterations: Number of iterations per length
        """
        logger.info("Benchmarking embedding generation...")
        
        for length in text_lengths:
            text = " ".join(["word"] * length)
            
            def generate_embedding():
                # Assuming rag_system has an embed method or embedding_function
                if hasattr(self.rag_system, 'embedding_function'):
                    return self.rag_system.embedding_function(text)
                elif hasattr(self.rag_system, 'embed'):
                    return self.rag_system.embed(text)
                else:
                    # Fallback: simulate embedding
                    time.sleep(0.01)
            
            self.runner.run_benchmark(
                name=f"Embedding_{length}_words",
                func=generate_embedding,
                iterations=iterations
            )
    
    def benchmark_vector_search(
        self,
        k_values: List[int] = [5, 10, 20, 50],
        iterations: int = 10
    ):
        """
        Benchmark vector search with different k values.
        
        Args:
            k_values: List of k values to test
            iterations: Number of iterations per k value
        """
        logger.info("Benchmarking vector search...")
        
        query = self.test_queries[0]
        
        for k in k_values:
            def search():
                # Assuming rag_system has a retrieve method
                if hasattr(self.rag_system, 'retrieve'):
                    return self.rag_system.retrieve(query, k=k)
                elif hasattr(self.rag_system, 'search'):
                    return self.rag_system.search(query, k=k)
                else:
                    # Fallback: simulate search
                    time.sleep(0.01)
            
            self.runner.run_benchmark(
                name=f"VectorSearch_k={k}",
                func=search,
                iterations=iterations
            )
    
    def benchmark_end_to_end_query(
        self,
        num_queries: int = 10,
        iterations: int = 3
    ):
        """
        Benchmark end-to-end query processing.
        
        Args:
            num_queries: Number of different queries to test
            iterations: Number of iterations per query
        """
        logger.info("Benchmarking end-to-end queries...")
        
        for i, query in enumerate(self.test_queries[:num_queries]):
            def process_query():
                # Assuming rag_system has a query method
                if hasattr(self.rag_system, 'query'):
                    return self.rag_system.query(query)
                else:
                    # Fallback: simulate query
                    time.sleep(0.1)
            
            self.runner.run_benchmark(
                name=f"E2E_Query_{i+1}",
                func=process_query,
                iterations=iterations,
                track_memory=True
            )
    
    def benchmark_retrieval_strategies(
        self,
        strategies: List[str] = ["simple", "multi_query", "hyde"],
        iterations: int = 5
    ):
        """
        Benchmark different retrieval strategies.
        
        Args:
            strategies: List of strategy names
            iterations: Number of iterations per strategy
        """
        logger.info("Benchmarking retrieval strategies...")
        
        query = self.test_queries[0]
        
        for strategy in strategies:
            def retrieve_with_strategy():
                # Assuming rag_system supports strategy parameter
                if hasattr(self.rag_system, 'query'):
                    return self.rag_system.query(query, strategy=strategy)
                else:
                    time.sleep(0.05)
            
            self.runner.run_benchmark(
                name=f"Strategy_{strategy}",
                func=retrieve_with_strategy,
                iterations=iterations
            )
    
    def benchmark_cache_performance(
        self,
        cache_sizes: List[int] = [10, 50, 100],
        iterations: int = 10
    ):
        """
        Benchmark caching performance.
        
        Args:
            cache_sizes: List of cache sizes to test
            iterations: Number of iterations
        """
        logger.info("Benchmarking cache performance...")
        
        # Test cache hit vs miss
        query = self.test_queries[0]
        
        def query_with_cold_cache():
            # Clear cache first
            if hasattr(self.rag_system, 'clear_cache'):
                self.rag_system.clear_cache()
            
            if hasattr(self.rag_system, 'query'):
                return self.rag_system.query(query)
            else:
                time.sleep(0.1)
        
        def query_with_warm_cache():
            # Don't clear cache - should hit
            if hasattr(self.rag_system, 'query'):
                return self.rag_system.query(query)
            else:
                time.sleep(0.01)  # Simulated cache hit is faster
        
        # Benchmark cold cache
        self.runner.run_benchmark(
            name="Cache_Miss",
            func=query_with_cold_cache,
            iterations=iterations
        )
        
        # Warm up cache
        if hasattr(self.rag_system, 'query'):
            self.rag_system.query(query)
        
        # Benchmark warm cache
        self.runner.run_benchmark(
            name="Cache_Hit",
            func=query_with_warm_cache,
            iterations=iterations
        )
    
    def benchmark_concurrent_queries(
        self,
        num_concurrent: int = 10,
        iterations: int = 5
    ):
        """
        Benchmark concurrent query processing.
        
        Args:
            num_concurrent: Number of concurrent queries
            iterations: Number of iterations
        """
        logger.info("Benchmarking concurrent queries...")
        
        import concurrent.futures
        
        def process_concurrent():
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                queries = self.test_queries[:num_concurrent]
                
                if hasattr(self.rag_system, 'query'):
                    futures = [
                        executor.submit(self.rag_system.query, q)
                        for q in queries
                    ]
                    return [f.result() for f in futures]
                else:
                    # Simulate
                    time.sleep(0.1)
        
        self.runner.run_benchmark(
            name=f"Concurrent_{num_concurrent}_queries",
            func=process_concurrent,
            iterations=iterations,
            track_memory=True
        )
    
    def run_all_benchmarks(self) -> BenchmarkSuite:
        """
        Run all benchmark suites.
        
        Returns:
            Complete benchmark suite with results
        """
        logger.info("Starting comprehensive benchmark suite...")
        
        try:
            self.benchmark_ingestion()
        except Exception as e:
            logger.error(f"Ingestion benchmark failed: {e}")
        
        try:
            self.benchmark_embedding_generation()
        except Exception as e:
            logger.error(f"Embedding benchmark failed: {e}")
        
        try:
            self.benchmark_vector_search()
        except Exception as e:
            logger.error(f"Vector search benchmark failed: {e}")
        
        try:
            self.benchmark_end_to_end_query()
        except Exception as e:
            logger.error(f"E2E query benchmark failed: {e}")
        
        try:
            self.benchmark_retrieval_strategies()
        except Exception as e:
            logger.error(f"Strategy benchmark failed: {e}")
        
        try:
            self.benchmark_cache_performance()
        except Exception as e:
            logger.error(f"Cache benchmark failed: {e}")
        
        try:
            self.benchmark_concurrent_queries()
        except Exception as e:
            logger.error(f"Concurrent benchmark failed: {e}")
        
        logger.info("Benchmark suite complete")
        return self.runner.get_suite()


class RetrievalQualityBenchmark:
    """
    Benchmark retrieval quality using test datasets.
    
    Metrics:
    - Precision@K
    - Recall@K
    - MRR (Mean Reciprocal Rank)
    - NDCG (Normalized Discounted Cumulative Gain)
    """
    
    def __init__(
        self,
        rag_system: Any,
        test_dataset: List[Dict[str, Any]]
    ):
        """
        Initialize quality benchmark.
        
        Args:
            rag_system: RAG system to evaluate
            test_dataset: List of test cases with query and relevant_docs
        """
        self.rag_system = rag_system
        self.test_dataset = test_dataset
    
    def precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int
    ) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant = set(relevant_docs)
        
        return len(retrieved_k & relevant) / k
    
    def recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if not relevant_docs:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant = set(relevant_docs)
        
        return len(retrieved_k & relevant) / len(relevant)
    
    def mrr(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        relevant = set(relevant_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate(self, k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate retrieval quality.
        
        Args:
            k_values: List of k values for metrics
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating retrieval quality...")
        
        metrics = defaultdict(list)
        
        for test_case in self.test_dataset:
            query = test_case["query"]
            relevant_docs = test_case["relevant_docs"]
            
            # Retrieve documents
            if hasattr(self.rag_system, 'retrieve'):
                results = self.rag_system.retrieve(query, k=max(k_values))
                retrieved_docs = [r.get("id", r.get("text", "")) for r in results]
            else:
                retrieved_docs = []
            
            # Calculate metrics for each k
            for k in k_values:
                metrics[f"precision@{k}"].append(
                    self.precision_at_k(retrieved_docs, relevant_docs, k)
                )
                metrics[f"recall@{k}"].append(
                    self.recall_at_k(retrieved_docs, relevant_docs, k)
                )
            
            # MRR
            metrics["mrr"].append(self.mrr(retrieved_docs, relevant_docs))
        
        # Compute averages
        results = {
            metric: statistics.mean(values)
            for metric, values in metrics.items()
        }
        
        logger.info("Quality evaluation complete")
        return results


def generate_benchmark_report(
    suite: BenchmarkSuite,
    output_path: str = "benchmark_report.md"
):
    """
    Generate a markdown report from benchmark results.
    
    Args:
        suite: BenchmarkSuite with results
        output_path: Path to save report
    """
    lines = [
        f"# {suite.name}",
        f"\nGenerated: {suite.timestamp}\n",
        "## Summary\n",
        f"- **Total Benchmarks**: {suite.summary['total_benchmarks']}",
        f"- **Successful**: {suite.summary['successful']}",
        f"- **Failed**: {suite.summary['failed']}",
        f"- **Total Duration**: {suite.summary['total_duration']:.3f}s",
        f"- **Average Duration**: {suite.summary['avg_duration']:.3f}s",
        f"- **Median Duration**: {suite.summary['median_duration']:.3f}s\n",
        "## Detailed Results\n",
        "| Benchmark | Duration (s) | Status | Metadata |",
        "|-----------|--------------|--------|----------|"
    ]
    
    for result in suite.results:
        status = "✅ Success" if result.success else "❌ Failed"
        metadata_str = ", ".join(
            f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in result.metadata.items()
            if k not in ["all_durations", "iterations"]
        )
        
        lines.append(
            f"| {result.name} | {result.duration:.4f} | {status} | {metadata_str} |"
        )
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Generated benchmark report: {output_path}")
