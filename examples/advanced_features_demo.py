"""
Advanced Features Example Usage

Demonstrates how to use all advanced features:
- Video Processing with Whisper
- FAISS Vector Store
- Benchmark Suite
- MLflow Integration
"""

import os
import numpy as np
from pathlib import Path


# ============================================================================
# 1. Video Processing Example
# ============================================================================

def example_video_processing():
    """Example: Process videos and add to RAG system."""
    print("\n" + "="*60)
    print("VIDEO PROCESSING EXAMPLE")
    print("="*60)
    
    try:
        from src.processing.video_processor import (
            VideoProcessor,
            VideoRAGIntegration,
            process_video_for_rag
        )
        
        # Initialize processor
        processor = VideoProcessor(
            whisper_model="base",  # Options: tiny, base, small, medium, large
            chunk_duration=30,     # 30-second chunks
            device="cpu"           # Use "cuda" for GPU
        )
        
        # Example 1: Process single video
        print("\n1. Processing single video...")
        
        # Note: Replace with actual video path
        video_path = "data/videos/sample_video.mp4"
        
        if os.path.exists(video_path):
            video_doc = processor.process_video(
                video_path=video_path,
                title="Sample Training Video",
                keep_audio=False  # Delete audio after transcription
            )
            
            print(f"✓ Processed: {video_doc.title}")
            print(f"  Duration: {video_doc.metadata['duration']:.1f}s")
            print(f"  Segments: {len(video_doc.transcript_segments)}")
            print(f"  Language: {video_doc.language}")
            
            # Show first segment
            if video_doc.transcript_segments:
                seg = video_doc.transcript_segments[0]
                print(f"\n  First segment ({seg.start_timestamp} - {seg.end_timestamp}):")
                print(f"  {seg.text[:100]}...")
        else:
            print(f"⚠ Video not found: {video_path}")
            print("  Skipping video processing demo")
        
        # Example 2: Convert to RAG documents
        print("\n2. Converting to RAG documents...")
        
        # Simulate video document
        from src.processing.video_processor import VideoDocument, TranscriptSegment
        
        segments = [
            TranscriptSegment(
                text="Welcome to the company onboarding program. Today we'll cover policies.",
                start_time=0.0,
                end_time=30.0
            ),
            TranscriptSegment(
                text="Our vacation policy allows 20 days per year for full-time employees.",
                start_time=30.0,
                end_time=60.0
            )
        ]
        
        demo_video = VideoDocument(
            video_path="demo.mp4",
            title="HR Onboarding",
            transcript_segments=segments,
            metadata={"category": "HR", "year": 2024},
            full_transcript=" ".join(s.text for s in segments)
        )
        
        rag_docs = VideoRAGIntegration.video_to_documents(
            demo_video,
            include_timestamps=True
        )
        
        print(f"✓ Created {len(rag_docs)} RAG documents")
        print(f"\n  Sample document:")
        print(f"  Text: {rag_docs[0]['text'][:100]}...")
        print(f"  Metadata: {list(rag_docs[0]['metadata'].keys())}")
        
        print("\n✓ Video processing example completed")
        
    except ImportError as e:
        print(f"⚠ Video processing not available: {e}")
        print("  Install with: pip install openai-whisper moviepy")


# ============================================================================
# 2. FAISS Vector Store Example
# ============================================================================

def example_faiss_vector_store():
    """Example: Using FAISS for vector storage and search."""
    print("\n" + "="*60)
    print("FAISS VECTOR STORE EXAMPLE")
    print("="*60)
    
    try:
        from src.vectorstore.faiss_store import (
            FAISSVectorStore,
            FAISSConfig,
            FAISSRetriever
        )
        
        # Example 1: Create FAISS store
        print("\n1. Creating FAISS vector store...")
        
        config = FAISSConfig(
            index_type="HNSW",    # Fast approximate search
            dimension=768,         # Embedding dimension
            metric="L2",           # Distance metric
            normalize_embeddings=True,
            use_gpu=False
        )
        
        vector_store = FAISSVectorStore(
            config=config,
            persist_directory="data/faiss_index"
        )
        
        print(f"✓ Created FAISS store")
        print(f"  Index type: {config.index_type}")
        print(f"  Dimension: {config.dimension}")
        
        # Example 2: Add documents
        print("\n2. Adding documents...")
        
        # Sample data
        texts = [
            "Our vacation policy provides 20 days of paid time off annually.",
            "Healthcare benefits include medical, dental, and vision coverage.",
            "The 401k retirement plan offers 5% company matching.",
            "Remote work policy allows 3 days per week from home.",
            "Professional development budget is $2000 per year."
        ]
        
        # Generate random embeddings (in real use, use actual embedding model)
        np.random.seed(42)
        embeddings = [np.random.rand(768).astype('float32') for _ in texts]
        
        metadatas = [
            {"category": "HR", "topic": "vacation"},
            {"category": "HR", "topic": "healthcare"},
            {"category": "HR", "topic": "retirement"},
            {"category": "IT", "topic": "remote_work"},
            {"category": "HR", "topic": "development"}
        ]
        
        ids = vector_store.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(ids)} documents")
        
        # Example 3: Search
        print("\n3. Searching documents...")
        
        query_emb = np.random.rand(768).astype('float32')
        results = vector_store.search(query_emb, k=3)
        
        print(f"✓ Found {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n  Result {i} (score: {score:.4f}):")
            print(f"  {doc.text[:80]}...")
            print(f"  Category: {doc.metadata.get('category')}")
        
        # Example 4: Metadata filtering
        print("\n4. Searching with metadata filter...")
        
        hr_docs = vector_store.search_by_metadata(
            metadata_filter={"category": "HR"},
            k=10
        )
        
        print(f"✓ Found {len(hr_docs)} HR documents")
        
        # Example 5: Save and stats
        print("\n5. Saving and getting stats...")
        
        vector_store.save()
        stats = vector_store.get_stats()
        
        print(f"✓ Saved to disk")
        print(f"  Stats: {stats}")
        
        print("\n✓ FAISS vector store example completed")
        
    except ImportError as e:
        print(f"⚠ FAISS not available: {e}")
        print("  Install with: pip install faiss-cpu")


# ============================================================================
# 3. Benchmark Suite Example
# ============================================================================

def example_benchmark_suite():
    """Example: Running performance benchmarks."""
    print("\n" + "="*60)
    print("BENCHMARK SUITE EXAMPLE")
    print("="*60)
    
    from src.evaluation.benchmark import (
        BenchmarkRunner,
        RAGBenchmark,
        RetrievalQualityBenchmark,
        generate_benchmark_report
    )
    
    # Example 1: Simple function benchmark
    print("\n1. Simple function benchmark...")
    
    runner = BenchmarkRunner("Function Benchmark")
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    result = runner.run_benchmark(
        name="fibonacci_10",
        func=fibonacci,
        n=10,
        iterations=5,
        warmup=1
    )
    
    print(f"✓ Benchmark completed")
    print(f"  Average duration: {result.duration:.6f}s")
    print(f"  Min: {result.metadata['min_duration']:.6f}s")
    print(f"  Max: {result.metadata['max_duration']:.6f}s")
    
    # Example 2: Mock RAG benchmark
    print("\n2. RAG system benchmark...")
    
    # Create mock RAG system
    class MockRAGSystem:
        def query(self, query):
            import time
            time.sleep(0.01)  # Simulate processing
            return "Mock response"
        
        def retrieve(self, query, k=5):
            import time
            time.sleep(0.005)
            return [{"text": f"Doc {i}"} for i in range(k)]
    
    rag_system = MockRAGSystem()
    
    benchmark = RAGBenchmark(
        rag_system=rag_system,
        test_documents=["doc1", "doc2", "doc3"],
        test_queries=["query1", "query2"]
    )
    
    # Run individual benchmarks
    print("  Running vector search benchmark...")
    benchmark.benchmark_vector_search(
        k_values=[5, 10],
        iterations=3
    )
    
    print("  Running E2E query benchmark...")
    benchmark.benchmark_end_to_end_query(
        num_queries=2,
        iterations=2
    )
    
    # Get results
    suite = benchmark.runner.get_suite()
    
    print(f"\n✓ Benchmark suite completed")
    print(f"  Total benchmarks: {suite.summary['total_benchmarks']}")
    print(f"  Successful: {suite.summary['successful']}")
    print(f"  Total duration: {suite.summary['total_duration']:.3f}s")
    
    # Example 3: Generate report
    print("\n3. Generating benchmark report...")
    
    report_path = "benchmark_report.md"
    generate_benchmark_report(suite, report_path)
    
    if os.path.exists(report_path):
        print(f"✓ Report saved to: {report_path}")
        
        # Show first few lines
        with open(report_path, 'r') as f:
            lines = f.readlines()[:10]
            print("\n  Report preview:")
            for line in lines:
                print(f"  {line.rstrip()}")
    
    # Example 4: Quality metrics
    print("\n4. Quality metrics example...")
    
    test_dataset = [
        {
            "query": "What is the vacation policy?",
            "relevant_docs": ["doc_vacation_1", "doc_vacation_2"]
        }
    ]
    
    quality_bench = RetrievalQualityBenchmark(
        rag_system=rag_system,
        test_dataset=test_dataset
    )
    
    # Calculate metrics
    precision = quality_bench.precision_at_k(
        retrieved_docs=["doc_vacation_1", "doc_other", "doc_vacation_2"],
        relevant_docs=["doc_vacation_1", "doc_vacation_2"],
        k=3
    )
    
    print(f"✓ Precision@3: {precision:.3f}")
    
    print("\n✓ Benchmark suite example completed")


# ============================================================================
# 4. MLflow Integration Example
# ============================================================================

def example_mlflow_integration():
    """Example: Using MLflow for experiment tracking."""
    print("\n" + "="*60)
    print("MLFLOW INTEGRATION EXAMPLE")
    print("="*60)
    
    try:
        from src.tracking.mlflow_integration import (
            MLflowTracker,
            RAGExperimentTracker,
            ExperimentConfig,
            MLflowCallback
        )
        
        # Example 1: Basic MLflow tracking
        print("\n1. Basic MLflow tracking...")
        
        config = ExperimentConfig(
            experiment_name="Demo_RAG_Experiments",
            tracking_uri="./mlruns",
            tags={"project": "demo"}
        )
        
        tracker = MLflowTracker(config=config)
        
        # Start a run
        tracker.start_run(
            run_name="demo_run_1",
            tags={"type": "demo"},
            description="Demo of MLflow tracking"
        )
        
        # Log parameters
        tracker.log_params({
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "top_k": 5,
            "temperature": 0.7
        })
        
        # Simulate some work and log metrics
        import time
        start = time.time()
        time.sleep(0.1)  # Simulate processing
        end = time.time()
        
        tracker.log_metrics({
            "latency": end - start,
            "precision@5": 0.85,
            "recall@5": 0.78
        })
        
        # Log artifact
        config_dict = {"model": "gpt-4", "temperature": 0.7}
        tracker.log_dict(config_dict, "config.json")
        
        # End run
        tracker.end_run()
        
        print("✓ MLflow run completed")
        
        # Example 2: RAG experiment tracking
        print("\n2. RAG experiment tracking...")
        
        exp_tracker = RAGExperimentTracker(
            experiment_name="Demo_RAG_Experiments"
        )
        
        exp_tracker.track_retrieval_experiment(
            run_name="hybrid_retrieval",
            retrieval_config={
                "strategy": "hybrid",
                "top_k": 10,
                "rerank": True,
                "alpha": 0.5
            },
            embedding_model="all-mpnet-base-v2",
            metrics={
                "precision@5": 0.87,
                "recall@5": 0.82,
                "mrr": 0.91,
                "latency_ms": 189.3
            },
            queries=["query1", "query2"]
        )
        
        print("✓ Retrieval experiment tracked")
        
        # Example 3: Track a query
        print("\n3. Tracking individual query...")
        
        exp_tracker.track_rag_query(
            query="What is the vacation policy?",
            response="Our company provides 20 days of paid vacation...",
            retrieved_docs=[
                {"text": "Vacation policy doc", "score": 0.95},
                {"text": "HR policies doc", "score": 0.87}
            ],
            metrics={
                "latency_seconds": 1.23,
                "num_retrieved_docs": 2,
                "relevance_score": 0.91
            },
            config={"model": "gpt-4", "strategy": "hybrid"}
        )
        
        print("✓ Query tracked")
        
        # Example 4: Compare runs (simulated)
        print("\n4. Run comparison example...")
        
        print("✓ To compare runs:")
        print("  1. Start MLflow UI: mlflow ui --backend-store-uri ./mlruns")
        print("  2. Open browser: http://localhost:5000")
        print("  3. Select runs and click 'Compare'")
        
        print("\n✓ MLflow integration example completed")
        
    except ImportError as e:
        print(f"⚠ MLflow not available: {e}")
        print("  Install with: pip install mlflow")


# ============================================================================
# 5. Full Integration Example
# ============================================================================

def example_full_integration():
    """Example: Using all features together."""
    print("\n" + "="*60)
    print("FULL INTEGRATION EXAMPLE")
    print("="*60)
    
    print("\nThis example shows how to use all features together:")
    print("1. Process videos → Extract transcripts")
    print("2. Store in FAISS → Fast vector search")
    print("3. Benchmark performance → Measure speed & quality")
    print("4. Track with MLflow → Monitor experiments")
    
    print("\nExample workflow:")
    print("""
    # 1. Process video
    from src.processing.video_processor import VideoProcessor
    processor = VideoProcessor()
    video_doc = processor.process_video("training.mp4")
    
    # 2. Store in FAISS
    from src.vectorstore.faiss_store import FAISSVectorStore
    vector_store = FAISSVectorStore()
    vector_store.add(texts, embeddings, metadatas)
    
    # 3. Run benchmarks
    from src.evaluation.benchmark import RAGBenchmark
    benchmark = RAGBenchmark(rag_system)
    suite = benchmark.run_all_benchmarks()
    
    # 4. Track with MLflow
    from src.tracking.mlflow_integration import RAGExperimentTracker
    tracker = RAGExperimentTracker()
    tracker.track_benchmark("performance", suite.to_dict())
    """)
    
    print("\n✓ Full integration example completed")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES EXAMPLES")
    print("="*60)
    print("\nDemonstrating all advanced RAG features:")
    print("- Video Processing with Whisper")
    print("- FAISS Vector Store")
    print("- Benchmark Suite")
    print("- MLflow Integration")
    
    # Run examples
    example_video_processing()
    example_faiss_vector_store()
    example_benchmark_suite()
    example_mlflow_integration()
    example_full_integration()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Check docs/ADVANCED_FEATURES.md for detailed documentation")
    print("2. Run tests: pytest tests/test_advanced_features.py")
    print("3. Start MLflow UI: mlflow ui --backend-store-uri ./mlruns")
    print("4. Explore example notebooks in notebooks/")
    
    print("\n✓ RAG system is now 100% complete!")


if __name__ == "__main__":
    main()
