# Advanced Features Documentation

This document covers the advanced features added to the RAG system for 100% completion.

## Table of Contents

1. [Video Processing with Whisper](#video-processing)
2. [FAISS Vector Store](#faiss-vector-store)
3. [Benchmark Suite](#benchmark-suite)
4. [MLflow Integration](#mlflow-integration)

---

## Video Processing

### Overview

The video processing module enables the RAG system to ingest and search video content by:
- Extracting audio from video files
- Transcribing audio using OpenAI Whisper
- Creating timestamped, searchable chunks
- Integrating transcripts into the vector store

### Installation

```bash
pip install openai-whisper moviepy ffmpeg-python
```

### Basic Usage

```python
from src.processing.video_processor import VideoProcessor, process_video_for_rag

# Initialize processor
processor = VideoProcessor(
    whisper_model="base",  # Options: tiny, base, small, medium, large
    chunk_duration=30,     # Seconds per chunk
    device="cpu"           # or "cuda" for GPU
)

# Process a single video
video_doc = processor.process_video(
    video_path="path/to/video.mp4",
    title="My Video",
    keep_audio=False
)

# Or use convenience function
video_doc, rag_docs = process_video_for_rag(
    video_path="path/to/video.mp4",
    whisper_model="base"
)
```

### Processing Multiple Videos

```python
# Process entire directory
video_docs = processor.process_video_directory(
    directory="data/videos",
    extensions=(".mp4", ".avi", ".mov")
)
```

### RAG Integration

```python
from src.processing.video_processor import VideoRAGIntegration

# Convert to RAG documents
rag_docs = VideoRAGIntegration.video_to_documents(
    video_doc,
    include_timestamps=True
)

# Add to vector store
vector_store.add_documents([doc["text"] for doc in rag_docs])

# Format video context for retrieval
context = VideoRAGIntegration.format_video_context(
    retrieved_docs=retrieved_docs,
    max_segments=5
)
```

### Features

- **Multiple Model Sizes**: Choose from tiny (fastest) to large (most accurate)
- **Timestamped Segments**: Each chunk includes start/end timestamps
- **Metadata Rich**: Stores video path, duration, language, segment info
- **Search Support**: Full-text search across transcript segments
- **GPU Support**: Use CUDA for faster transcription

### Example Output

```json
{
  "title": "Company Training Video",
  "segments": [
    {
      "text": "Welcome to our new employee orientation...",
      "start_timestamp": "00:00:15",
      "end_timestamp": "00:00:45",
      "duration": 30.0
    }
  ],
  "language": "en",
  "duration": 1800.0
}
```

---

## FAISS Vector Store

### Overview

FAISS (Facebook AI Similarity Search) provides an alternative to ChromaDB with:
- Faster similarity search for large datasets
- Multiple index types (Flat, IVF, HNSW)
- GPU support
- Lower memory footprint
- Efficient batch operations

### Installation

```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### Index Types

1. **Flat** - Exact search (best for <100K vectors)
2. **IVF** - Inverted File Index (fast for >100K vectors)
3. **HNSW** - Hierarchical Navigable Small World (very fast, low recall)

### Basic Usage

```python
from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig

# Create configuration
config = FAISSConfig(
    index_type="HNSW",     # Flat, IVF, or HNSW
    dimension=768,          # Embedding dimension
    metric="L2",            # L2 or IP (inner product)
    use_gpu=False,
    normalize_embeddings=True
)

# Initialize store
vector_store = FAISSVectorStore(
    config=config,
    persist_directory="data/faiss_index"
)

# Add documents
ids = vector_store.add(
    texts=["Document 1", "Document 2"],
    embeddings=[emb1, emb2],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)

# Search
results = vector_store.search(
    query_embedding=query_emb,
    k=5,
    filter_func=lambda doc: doc.metadata.get("type") == "policy"
)

# Save and load
vector_store.save()
vector_store.load()
```

### Advanced Features

#### IVF Configuration

```python
config = FAISSConfig(
    index_type="IVF",
    nlist=100,      # Number of clusters
    nprobe=10       # Number of clusters to search
)
```

#### HNSW Configuration

```python
config = FAISSConfig(
    index_type="HNSW",
    m=32,                  # Number of connections
    ef_construction=40,    # Build quality
    ef_search=16          # Search quality
)
```

#### Metadata Filtering

```python
# Search with metadata filter
def filter_func(doc):
    return (
        doc.metadata.get("category") == "HR" and
        doc.metadata.get("year") >= 2023
    )

results = vector_store.search(query_emb, k=10, filter_func=filter_func)

# Search by metadata directly
docs = vector_store.search_by_metadata(
    metadata_filter={"category": "HR", "year": 2023},
    k=10
)
```

#### RAG Retriever Interface

```python
from src.vectorstore.faiss_store import FAISSRetriever

retriever = FAISSRetriever(
    vector_store=vector_store,
    embedding_function=embed_text,
    k=5
)

# Retrieve documents
results = retriever.retrieve(
    query="What is the vacation policy?",
    metadata_filter={"category": "HR"}
)
```

### Performance Comparison

| Index Type | Build Time | Query Time | Recall | Use Case |
|------------|------------|------------|--------|----------|
| Flat | Fast | Slow | 100% | <100K vectors |
| IVF | Medium | Fast | 95-99% | >100K vectors |
| HNSW | Slow | Very Fast | 90-95% | Real-time search |

---

## Benchmark Suite

### Overview

Comprehensive performance testing framework for:
- Ingestion throughput
- Embedding generation speed
- Vector search latency
- End-to-end query performance
- Retrieval strategy comparison
- Cache effectiveness
- Quality metrics (Precision@K, Recall@K, MRR)

### Installation

No additional dependencies required (uses standard library).

### Basic Usage

```python
from src.evaluation.benchmark import RAGBenchmark, generate_benchmark_report

# Initialize benchmark
benchmark = RAGBenchmark(
    rag_system=my_rag_system,
    test_documents=test_docs,
    test_queries=test_queries
)

# Run all benchmarks
suite = benchmark.run_all_benchmarks()

# Generate report
generate_benchmark_report(suite, "benchmark_report.md")

# Save results
suite.save("benchmark_results.json")
```

### Individual Benchmarks

#### Ingestion Performance

```python
benchmark.benchmark_ingestion(
    batch_sizes=[10, 50, 100],
    iterations=3
)
```

#### Embedding Generation

```python
benchmark.benchmark_embedding_generation(
    text_lengths=[50, 200, 500, 1000],  # words
    iterations=10
)
```

#### Vector Search

```python
benchmark.benchmark_vector_search(
    k_values=[5, 10, 20, 50],
    iterations=10
)
```

#### End-to-End Queries

```python
benchmark.benchmark_end_to_end_query(
    num_queries=10,
    iterations=3
)
```

#### Retrieval Strategies

```python
benchmark.benchmark_retrieval_strategies(
    strategies=["simple", "multi_query", "hyde"],
    iterations=5
)
```

#### Cache Performance

```python
benchmark.benchmark_cache_performance(
    cache_sizes=[10, 50, 100],
    iterations=10
)
```

#### Concurrent Queries

```python
benchmark.benchmark_concurrent_queries(
    num_concurrent=10,
    iterations=5
)
```

### Quality Metrics

```python
from src.evaluation.benchmark import RetrievalQualityBenchmark

# Prepare test dataset
test_dataset = [
    {
        "query": "What is the vacation policy?",
        "relevant_docs": ["doc_123", "doc_456"]
    },
    # ... more test cases
]

# Initialize quality benchmark
quality_bench = RetrievalQualityBenchmark(
    rag_system=my_rag_system,
    test_dataset=test_dataset
)

# Evaluate
results = quality_bench.evaluate(k_values=[1, 5, 10])

print(f"Precision@5: {results['precision@5']:.3f}")
print(f"Recall@5: {results['recall@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
```

### Custom Benchmarks

```python
from src.evaluation.benchmark import BenchmarkRunner

runner = BenchmarkRunner("Custom Benchmark")

def my_function():
    # Your code here
    pass

result = runner.run_benchmark(
    name="my_custom_test",
    func=my_function,
    iterations=10,
    warmup=2,
    track_memory=True
)

suite = runner.get_suite()
```

### Example Report

```markdown
# RAG System Benchmark

Generated: 2024-01-15T10:30:00

## Summary

- **Total Benchmarks**: 25
- **Successful**: 25
- **Failed**: 0
- **Total Duration**: 125.456s
- **Average Duration**: 5.018s
- **Median Duration**: 2.341s

## Detailed Results

| Benchmark | Duration (s) | Status | Metadata |
|-----------|--------------|--------|----------|
| Ingestion_10_docs | 0.1234 | ✅ Success | avg_duration: 0.123 |
| VectorSearch_k=5 | 0.0456 | ✅ Success | avg_duration: 0.046 |
| E2E_Query_1 | 1.2345 | ✅ Success | peak_memory_mb: 245.3 |
```

---

## MLflow Integration

### Overview

MLflow integration provides:
- Experiment tracking
- Parameter logging
- Metric visualization
- Artifact storage
- Run comparison
- Model versioning

### Installation

```bash
pip install mlflow
```

### Basic Usage

```python
from src.tracking.mlflow_integration import MLflowTracker, ExperimentConfig

# Create tracker
config = ExperimentConfig(
    experiment_name="RAG_Experiments",
    tracking_uri="./mlruns",
    tags={"project": "company_rag"}
)

tracker = MLflowTracker(config=config)

# Start a run
tracker.start_run(
    run_name="test_retrieval_v1",
    tags={"type": "retrieval"},
    description="Testing new retrieval strategy"
)

# Log parameters
tracker.log_params({
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 512,
    "top_k": 5,
    "strategy": "hybrid"
})

# Log metrics
tracker.log_metrics({
    "precision@5": 0.85,
    "recall@5": 0.78,
    "latency_ms": 234.5
})

# Log artifacts
tracker.log_dict(
    {"config": "value"},
    "config.json"
)

# End run
tracker.end_run()
```

### RAG Experiment Tracking

```python
from src.tracking.mlflow_integration import RAGExperimentTracker

# Initialize
experiment_tracker = RAGExperimentTracker(
    experiment_name="RAG_Experiments"
)

# Track retrieval experiment
experiment_tracker.track_retrieval_experiment(
    run_name="hybrid_retrieval_v2",
    retrieval_config={
        "strategy": "hybrid",
        "top_k": 10,
        "rerank": True
    },
    embedding_model="all-mpnet-base-v2",
    metrics={
        "precision@5": 0.87,
        "recall@5": 0.82,
        "mrr": 0.91,
        "latency_ms": 189.3
    },
    queries=test_queries,
    results=retrieval_results
)
```

### Track Individual Queries

```python
experiment_tracker.track_rag_query(
    query="What is the vacation policy?",
    response="Our vacation policy allows...",
    retrieved_docs=retrieved_docs,
    metrics={
        "latency_seconds": 1.23,
        "num_retrieved_docs": 5,
        "relevance_score": 0.89
    },
    config={
        "strategy": "hybrid",
        "model": "gpt-4"
    }
)
```

### Track Benchmarks

```python
experiment_tracker.track_benchmark(
    benchmark_name="end_to_end_performance",
    benchmark_results=suite.to_dict(),
    config={
        "test_queries": 100,
        "iterations": 10
    }
)
```

### Compare Runs

```python
# Compare multiple runs
comparison = experiment_tracker.compare_runs(
    run_ids=["run_123", "run_456", "run_789"],
    metrics=["precision@5", "latency_ms"]
)

print(comparison["metric_comparison"]["precision@5"])
# {'run_123': 0.85, 'run_456': 0.87, 'run_789': 0.82}
```

### Find Best Run

```python
best_run = experiment_tracker.get_best_run(
    metric_name="precision@5",
    maximize=True,
    filter_string="tags.type = 'retrieval'"
)

print(f"Best run: {best_run['run_name']}")
print(f"Precision@5: {best_run['metric_value']}")
print(f"Parameters: {best_run['params']}")
```

### Automatic Callbacks

```python
from src.tracking.mlflow_integration import MLflowCallback

# Create callback
callback = MLflowCallback(experiment_tracker)

# In your RAG orchestrator
class RAGOrchestrator:
    def query(self, query: str, config: dict):
        # Callback: query start
        callback.on_query_start(query, config)
        
        # Process query...
        response = self._generate_response(query)
        retrieved_docs = self._retrieve(query)
        
        # Callback: query end
        callback.on_query_end(
            response=response,
            retrieved_docs=retrieved_docs,
            additional_metrics={"relevance": 0.89}
        )
        
        return response
```

### Function Decorator

```python
from src.tracking.mlflow_integration import mlflow_track_function

@mlflow_track_function(
    tracker=tracker,
    run_name="embedding_generation",
    log_params=True,
    log_result=True
)
def generate_embeddings(texts: list, model: str):
    # Your embedding logic
    return embeddings
```

### MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000

# Access at http://localhost:5000
```

### Features

- **Automatic Logging**: Track all parameters and metrics automatically
- **Artifact Storage**: Save models, configs, results
- **Visualization**: Compare runs with built-in charts
- **Model Registry**: Version and deploy models
- **Reproducibility**: Track everything needed to reproduce results
- **Collaboration**: Share experiments with team

### Example UI Workflow

1. **View Experiments**: See all runs organized by experiment
2. **Compare Runs**: Side-by-side comparison of metrics
3. **Visualize Metrics**: Charts showing metric trends
4. **Download Artifacts**: Access saved configs and results
5. **Register Models**: Promote best models to registry

---

## Integration Examples

### Complete Video + FAISS + MLflow Pipeline

```python
from src.processing.video_processor import VideoProcessor
from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
from src.tracking.mlflow_integration import RAGExperimentTracker

# Initialize components
tracker = RAGExperimentTracker()
processor = VideoProcessor(whisper_model="base")

config = FAISSConfig(index_type="HNSW", dimension=768)
vector_store = FAISSVectorStore(config=config)

# Start MLflow run
tracker.tracker.start_run(run_name="video_ingestion_experiment")

# Process video
video_doc = processor.process_video("training_video.mp4")
rag_docs = VideoRAGIntegration.video_to_documents(video_doc)

# Add to FAISS
embeddings = [embed(doc["text"]) for doc in rag_docs]
vector_store.add(
    texts=[doc["text"] for doc in rag_docs],
    embeddings=embeddings,
    metadatas=[doc["metadata"] for doc in rag_docs]
)

# Log to MLflow
tracker.tracker.log_params({
    "video_path": "training_video.mp4",
    "whisper_model": "base",
    "faiss_index": "HNSW",
    "num_segments": len(rag_docs)
})

tracker.tracker.log_metrics({
    "video_duration": video_doc.metadata["duration"],
    "num_vectors": vector_store.index.ntotal
})

tracker.tracker.end_run()
```

### Benchmark with MLflow Tracking

```python
from src.evaluation.benchmark import RAGBenchmark

# Run benchmarks
benchmark = RAGBenchmark(rag_system=orchestrator)
suite = benchmark.run_all_benchmarks()

# Track in MLflow
experiment_tracker.track_benchmark(
    benchmark_name="full_system_benchmark",
    benchmark_results=suite.to_dict(),
    config={
        "system_version": "2.0",
        "index_type": "FAISS_HNSW"
    }
)
```

---

## Best Practices

### Video Processing

1. **Model Selection**: Use `base` for most cases, `small`/`medium` for better accuracy
2. **Chunk Duration**: 30-60 seconds provides good granularity
3. **GPU Usage**: Enable for faster processing of large videos
4. **Cleanup**: Set `keep_audio=False` to save disk space

### FAISS

1. **Index Selection**: 
   - Flat for <100K vectors (exact search)
   - IVF for 100K-10M vectors (fast approximate)
   - HNSW for real-time search (very fast)
2. **Normalization**: Enable for cosine similarity
3. **Persistence**: Save index regularly
4. **Rebuild**: Periodically rebuild after deletions

### Benchmarking

1. **Warmup**: Use warmup iterations for accurate timing
2. **Iterations**: Run 3-10 iterations for statistical significance
3. **Memory Tracking**: Enable for resource-intensive operations
4. **Comparison**: Baseline before optimizations

### MLflow

1. **Naming**: Use descriptive run names
2. **Tags**: Tag runs by type, version, purpose
3. **Metrics**: Log all relevant metrics (latency, accuracy, etc.)
4. **Artifacts**: Save configs, results, models
5. **Cleanup**: Archive old experiments periodically

---

## Troubleshooting

### Video Processing

**Issue**: "Whisper not installed"
```bash
pip install openai-whisper
```

**Issue**: "MoviePy error"
```bash
# Install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

### FAISS

**Issue**: "FAISS not installed"
```bash
pip install faiss-cpu  # or faiss-gpu
```

**Issue**: "Index not trained"
```python
# IVF indices need training
if config.index_type == "IVF":
    # Add enough documents for training (>nlist * 39)
    pass
```

### Benchmarking

**Issue**: "ReferenceError: weakly-referenced object no longer exists"
```python
# Ensure objects persist during benchmark
benchmark.rag_system = my_rag_system  # Keep reference
```

### MLflow

**Issue**: "Cannot connect to tracking server"
```bash
# Use local tracking
export MLFLOW_TRACKING_URI=./mlruns
```

**Issue**: "Experiment already exists"
```python
# It's fine - MLflow will use existing experiment
# Or delete: mlflow experiments delete --experiment-id <id>
```

---

## API Reference

Complete API documentation available in docstrings. Key classes:

- `VideoProcessor`: Process video files
- `FAISSVectorStore`: FAISS vector database
- `RAGBenchmark`: Performance benchmarking
- `MLflowTracker`: Experiment tracking
- `RAGExperimentTracker`: RAG-specific tracking

For detailed API docs:
```python
help(VideoProcessor)
help(FAISSVectorStore)
help(RAGBenchmark)
help(MLflowTracker)
```
