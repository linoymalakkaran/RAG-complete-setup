# Quick Reference: Advanced Features

Fast reference for using the advanced RAG features.

---

## 🎥 Video Processing

### Quick Start
```python
from src.processing.video_processor import VideoProcessor

processor = VideoProcessor(whisper_model="base")
video_doc = processor.process_video("video.mp4")
```

### Common Operations
```python
# Process directory
videos = processor.process_video_directory("videos/")

# RAG integration
from src.processing.video_processor import VideoRAGIntegration
rag_docs = VideoRAGIntegration.video_to_documents(video_doc)

# Search transcript
results = video_doc.search_text("vacation policy")
segment = video_doc.get_segment_by_time(45.0)
```

### Model Options
- `tiny` - Fastest, lowest accuracy
- `base` - **Recommended** for most use cases
- `small` - Better accuracy, slower
- `medium` - High accuracy, much slower
- `large` - Best accuracy, very slow

---

## 🚀 FAISS Vector Store

### Quick Start
```python
from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig

config = FAISSConfig(index_type="HNSW", dimension=768)
store = FAISSVectorStore(config, persist_directory="data/faiss")
```

### Index Types
```python
# Exact search (small datasets)
FAISSConfig(index_type="Flat")

# Fast search (medium datasets)
FAISSConfig(index_type="IVF", nlist=100, nprobe=10)

# Very fast search (large datasets)
FAISSConfig(index_type="HNSW", m=32, ef_search=16)
```

### Common Operations
```python
# Add documents
ids = store.add(texts, embeddings, metadatas)

# Search
results = store.search(query_emb, k=5)

# Metadata filtering
hr_docs = store.search_by_metadata({"category": "HR"})

# Save/Load
store.save()
store.load()
```

---

## 📊 Benchmark Suite

### Quick Start
```python
from src.evaluation.benchmark import RAGBenchmark

benchmark = RAGBenchmark(rag_system)
suite = benchmark.run_all_benchmarks()
```

### Individual Benchmarks
```python
# Ingestion speed
benchmark.benchmark_ingestion(batch_sizes=[10, 50, 100])

# Search latency
benchmark.benchmark_vector_search(k_values=[5, 10, 20])

# End-to-end queries
benchmark.benchmark_end_to_end_query(num_queries=10)

# Strategy comparison
benchmark.benchmark_retrieval_strategies(
    strategies=["simple", "multi_query", "hyde"]
)
```

### Generate Report
```python
from src.evaluation.benchmark import generate_benchmark_report

generate_benchmark_report(suite, "report.md")
```

### Quality Metrics
```python
from src.evaluation.benchmark import RetrievalQualityBenchmark

quality = RetrievalQualityBenchmark(rag_system, test_dataset)
metrics = quality.evaluate(k_values=[1, 5, 10])

print(f"Precision@5: {metrics['precision@5']}")
print(f"Recall@5: {metrics['recall@5']}")
print(f"MRR: {metrics['mrr']}")
```

---

## 📈 MLflow Tracking

### Quick Start
```python
from src.tracking.mlflow_integration import RAGExperimentTracker

tracker = RAGExperimentTracker(experiment_name="My_Experiments")
```

### Track Experiments
```python
# Track retrieval experiment
tracker.track_retrieval_experiment(
    run_name="hybrid_v1",
    retrieval_config={
        "strategy": "hybrid",
        "top_k": 10,
        "alpha": 0.5
    },
    embedding_model="all-mpnet-base-v2",
    metrics={
        "precision@5": 0.87,
        "latency_ms": 189.3
    }
)

# Track individual query
tracker.track_rag_query(
    query="What is the vacation policy?",
    response="Our policy allows...",
    retrieved_docs=docs,
    metrics={"latency": 1.23},
    config={"model": "gpt-4"}
)

# Track benchmark
tracker.track_benchmark(
    benchmark_name="performance_test",
    benchmark_results=suite.to_dict(),
    config={"version": "2.0"}
)
```

### Compare & Analyze
```python
# Find best run
best = tracker.get_best_run(
    metric_name="precision@5",
    maximize=True
)

# Compare runs
comparison = tracker.compare_runs(
    run_ids=["run_1", "run_2", "run_3"],
    metrics=["precision@5", "latency_ms"]
)
```

### MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000
```

---

## 🔗 Integration Examples

### Video → FAISS → MLflow
```python
from src.processing.video_processor import VideoProcessor
from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
from src.tracking.mlflow_integration import RAGExperimentTracker

# Process video
processor = VideoProcessor()
video_doc = processor.process_video("training.mp4")

# Add to FAISS
config = FAISSConfig(index_type="HNSW")
store = FAISSVectorStore(config)
store.add(texts, embeddings, metadatas)

# Track in MLflow
tracker = RAGExperimentTracker()
tracker.tracker.start_run("video_ingestion")
tracker.tracker.log_params({"video": "training.mp4"})
tracker.tracker.log_metrics({"num_segments": len(video_doc.transcript_segments)})
tracker.tracker.end_run()
```

### Benchmark → MLflow
```python
from src.evaluation.benchmark import RAGBenchmark
from src.tracking.mlflow_integration import RAGExperimentTracker

# Run benchmarks
benchmark = RAGBenchmark(rag_system)
suite = benchmark.run_all_benchmarks()

# Track results
tracker = RAGExperimentTracker()
tracker.track_benchmark(
    "full_system_benchmark",
    suite.to_dict(),
    config={"version": "1.0"}
)
```

---

## 🛠️ Common Tasks

### Task: Process videos and make searchable
```python
from src.processing.video_processor import VideoProcessor, VideoRAGIntegration

processor = VideoProcessor(whisper_model="base")
videos = processor.process_video_directory("data/videos")

for video in videos:
    rag_docs = VideoRAGIntegration.video_to_documents(video)
    # Add to your vector store
    vector_store.add_documents(rag_docs)
```

### Task: Switch from ChromaDB to FAISS
```python
from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig

# Create FAISS store
config = FAISSConfig(index_type="HNSW", dimension=768)
faiss_store = FAISSVectorStore(config, persist_directory="data/faiss")

# Migrate from ChromaDB (assuming you have docs and embeddings)
faiss_store.add(texts, embeddings, metadatas)
faiss_store.save()
```

### Task: Benchmark your RAG system
```python
from src.evaluation.benchmark import RAGBenchmark, generate_benchmark_report

benchmark = RAGBenchmark(
    rag_system=orchestrator,
    test_queries=["query1", "query2", "query3"]
)

suite = benchmark.run_all_benchmarks()
generate_benchmark_report(suite, "my_benchmark.md")
```

### Task: Track experiments over time
```python
from src.tracking.mlflow_integration import RAGExperimentTracker

tracker = RAGExperimentTracker()

# Experiment 1
tracker.track_retrieval_experiment(
    "baseline",
    {"strategy": "simple"},
    "model-v1",
    {"precision@5": 0.75}
)

# Experiment 2
tracker.track_retrieval_experiment(
    "improved",
    {"strategy": "hybrid"},
    "model-v2",
    {"precision@5": 0.87}
)

# Compare
best = tracker.get_best_run("precision@5", maximize=True)
```

---

## 📝 Configuration Examples

### Video Processing Config
```python
VideoProcessor(
    whisper_model="base",     # Model size
    chunk_duration=30,        # Seconds per chunk
    device="cpu",             # cpu or cuda
    language="en"            # Force language (or None)
)
```

### FAISS Config
```python
# For small datasets (<100K)
FAISSConfig(index_type="Flat", dimension=768)

# For medium datasets (100K-1M)
FAISSConfig(
    index_type="IVF",
    dimension=768,
    nlist=100,
    nprobe=10
)

# For large datasets (>1M)
FAISSConfig(
    index_type="HNSW",
    dimension=768,
    m=32,
    ef_construction=40,
    ef_search=16
)
```

### MLflow Config
```python
ExperimentConfig(
    experiment_name="RAG_Experiments",
    tracking_uri="./mlruns",
    artifact_location="./artifacts",
    tags={"project": "company_rag"}
)
```

---

## 🚨 Troubleshooting

### Video Processing
**Error**: "Whisper not found"
```bash
pip install openai-whisper
```

**Error**: "MoviePy error"
```bash
# Install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

### FAISS
**Error**: "FAISS not found"
```bash
pip install faiss-cpu  # or faiss-gpu
```

**Error**: "Index not trained"
```python
# IVF indices need training data
# Add at least nlist * 39 documents before use
```

### MLflow
**Error**: "Cannot connect"
```bash
# Use local tracking
export MLFLOW_TRACKING_URI=./mlruns
```

---

## 📚 More Resources

- **Full Documentation**: `docs/ADVANCED_FEATURES.md`
- **Examples**: `examples/advanced_features_demo.py`
- **Tests**: `tests/test_advanced_features.py`
- **API Reference**: Docstrings in source files
- **Completion Guide**: `COMPLETION_100_PERCENT.md`

---

*Quick Reference v1.0 - Updated January 1, 2026*
