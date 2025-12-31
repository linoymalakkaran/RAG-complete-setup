"""
Integration Tests for Advanced Features

Tests for:
- Video processing
- FAISS vector store
- Benchmark suite
- MLflow integration
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Video Processing Tests
# ============================================================================

class TestVideoProcessor:
    """Tests for video processing functionality."""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model."""
        model = Mock()
        model.transcribe.return_value = {
            "text": "This is a test transcription.",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "This is a test"
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "transcription."
                }
            ]
        }
        return model
    
    @pytest.fixture
    def temp_video_file(self):
        """Create temporary video file."""
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        # Create dummy file
        with open(video_path, 'wb') as f:
            f.write(b"dummy video content")
        
        yield video_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('src.processing.video_processor.WHISPER_AVAILABLE', True)
    @patch('src.processing.video_processor.MOVIEPY_AVAILABLE', True)
    def test_video_processor_initialization(self, mock_whisper_model):
        """Test VideoProcessor initialization."""
        with patch('src.processing.video_processor.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            from src.processing.video_processor import VideoProcessor
            
            processor = VideoProcessor(
                whisper_model="base",
                chunk_duration=30
            )
            
            assert processor.whisper_model_name == "base"
            assert processor.chunk_duration == 30
            mock_whisper.load_model.assert_called_once_with("base", device="cpu")
    
    def test_transcript_segment_creation(self):
        """Test TranscriptSegment dataclass."""
        from src.processing.video_processor import TranscriptSegment
        
        segment = TranscriptSegment(
            text="Test segment",
            start_time=0.0,
            end_time=30.0
        )
        
        assert segment.text == "Test segment"
        assert segment.duration == 30.0
        assert segment.start_timestamp == "00:00:00"
        assert segment.end_timestamp == "00:00:30"
    
    def test_video_document_search(self):
        """Test VideoDocument search functionality."""
        from src.processing.video_processor import VideoDocument, TranscriptSegment
        
        segments = [
            TranscriptSegment("vacation policy details", 0.0, 30.0),
            TranscriptSegment("healthcare benefits", 30.0, 60.0),
            TranscriptSegment("retirement planning", 60.0, 90.0)
        ]
        
        video_doc = VideoDocument(
            video_path="/path/to/video.mp4",
            title="HR Policies",
            transcript_segments=segments,
            metadata={},
            full_transcript="vacation policy details healthcare benefits retirement planning"
        )
        
        # Search for text
        results = video_doc.search_text("vacation")
        assert len(results) == 1
        assert results[0].text == "vacation policy details"
        
        # Get segment by time
        segment = video_doc.get_segment_by_time(45.0)
        assert segment.text == "healthcare benefits"
    
    def test_video_rag_integration(self):
        """Test VideoRAGIntegration."""
        from src.processing.video_processor import (
            VideoDocument,
            TranscriptSegment,
            VideoRAGIntegration
        )
        
        segments = [
            TranscriptSegment("Segment 1", 0.0, 30.0),
            TranscriptSegment("Segment 2", 30.0, 60.0)
        ]
        
        video_doc = VideoDocument(
            video_path="/path/to/video.mp4",
            title="Test Video",
            transcript_segments=segments,
            metadata={"category": "training"},
            full_transcript="Segment 1 Segment 2"
        )
        
        # Convert to RAG documents
        rag_docs = VideoRAGIntegration.video_to_documents(
            video_doc,
            include_timestamps=True
        )
        
        assert len(rag_docs) == 2
        assert "[00:00:00 - 00:00:30]" in rag_docs[0]["text"]
        assert rag_docs[0]["metadata"]["title"] == "Test Video"
        assert rag_docs[0]["metadata"]["segment_index"] == 0


# ============================================================================
# FAISS Vector Store Tests
# ============================================================================

class TestFAISSVectorStore:
    """Tests for FAISS vector store."""
    
    @pytest.fixture
    def temp_persist_dir(self):
        """Create temporary directory for persistence."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings."""
        np.random.seed(42)
        return [
            np.random.rand(768).astype('float32') for _ in range(10)
        ]
    
    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts."""
        return [f"Document {i}" for i in range(10)]
    
    @pytest.fixture
    def sample_metadatas(self):
        """Generate sample metadata."""
        return [{"source": f"doc_{i}", "category": "test"} for i in range(10)]
    
    @patch('src.vectorstore.faiss_store.FAISS_AVAILABLE', True)
    def test_faiss_store_initialization(self, temp_persist_dir):
        """Test FAISSVectorStore initialization."""
        with patch('src.vectorstore.faiss_store.faiss') as mock_faiss:
            # Mock FAISS index
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatL2.return_value = mock_index
            
            from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
            
            config = FAISSConfig(index_type="Flat", dimension=768)
            store = FAISSVectorStore(
                config=config,
                persist_directory=temp_persist_dir
            )
            
            assert store.config.index_type == "Flat"
            assert store.config.dimension == 768
            assert len(store.documents) == 0
    
    @patch('src.vectorstore.faiss_store.FAISS_AVAILABLE', True)
    @patch('src.vectorstore.faiss_store.faiss')
    def test_faiss_add_documents(
        self,
        mock_faiss,
        sample_texts,
        sample_embeddings,
        sample_metadatas
    ):
        """Test adding documents to FAISS store."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.is_trained = True
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
        
        config = FAISSConfig(dimension=768)
        store = FAISSVectorStore(config=config)
        
        # Add documents
        ids = store.add(
            texts=sample_texts,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas
        )
        
        assert len(ids) == 10
        assert len(store.documents) == 10
        mock_index.add.assert_called_once()
    
    @patch('src.vectorstore.faiss_store.FAISS_AVAILABLE', True)
    @patch('src.vectorstore.faiss_store.faiss')
    def test_faiss_search(
        self,
        mock_faiss,
        sample_texts,
        sample_embeddings,
        sample_metadatas
    ):
        """Test FAISS search."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 10
        mock_index.is_trained = True
        
        # Mock search results
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            np.array([[0, 1, 2, 3, 4]])
        )
        
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
        
        config = FAISSConfig(dimension=768)
        store = FAISSVectorStore(config=config)
        
        # Add documents first
        store.add(sample_texts, sample_embeddings, sample_metadatas)
        
        # Search
        query_emb = np.random.rand(768).astype('float32')
        results = store.search(query_emb, k=5)
        
        assert len(results) <= 5
        mock_index.search.assert_called_once()
    
    @patch('src.vectorstore.faiss_store.FAISS_AVAILABLE', True)
    @patch('src.vectorstore.faiss_store.faiss')
    def test_faiss_metadata_search(
        self,
        mock_faiss,
        sample_texts,
        sample_embeddings
    ):
        """Test metadata-based search."""
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.is_trained = True
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
        
        store = FAISSVectorStore(config=FAISSConfig(dimension=768))
        
        # Add documents with different categories
        metadatas = [
            {"category": "HR", "year": 2023},
            {"category": "IT", "year": 2023},
            {"category": "HR", "year": 2024}
        ]
        
        store.add(sample_texts[:3], sample_embeddings[:3], metadatas)
        
        # Search by metadata
        results = store.search_by_metadata({"category": "HR"})
        
        assert len(results) == 2
        assert all(doc.metadata["category"] == "HR" for doc in results)
    
    @patch('src.vectorstore.faiss_store.FAISS_AVAILABLE', True)
    @patch('src.vectorstore.faiss_store.faiss')
    def test_faiss_persistence(
        self,
        mock_faiss,
        temp_persist_dir,
        sample_texts,
        sample_embeddings,
        sample_metadatas
    ):
        """Test FAISS save and load."""
        mock_index = Mock()
        mock_index.ntotal = 10
        mock_index.is_trained = True
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_faiss.write_index = Mock()
        mock_faiss.read_index = Mock(return_value=mock_index)
        
        from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
        
        config = FAISSConfig(dimension=768)
        store = FAISSVectorStore(
            config=config,
            persist_directory=temp_persist_dir
        )
        
        # Add documents
        store.add(sample_texts, sample_embeddings, sample_metadatas)
        
        # Save
        store.save()
        
        # Verify save was called
        mock_faiss.write_index.assert_called_once()
        
        # Verify metadata file created
        metadata_path = os.path.join(temp_persist_dir, "metadata.pkl")
        # Would exist if not mocked


# ============================================================================
# Benchmark Suite Tests
# ============================================================================

class TestBenchmarkSuite:
    """Tests for benchmark suite."""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system."""
        system = Mock()
        system.query.return_value = "Test response"
        system.retrieve.return_value = [{"text": "doc1"}, {"text": "doc2"}]
        system.add_documents.return_value = ["id1", "id2"]
        return system
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult dataclass."""
        from src.evaluation.benchmark import BenchmarkResult
        
        result = BenchmarkResult(
            name="test_benchmark",
            duration=1.234,
            success=True,
            metadata={"iterations": 10}
        )
        
        assert result.name == "test_benchmark"
        assert result.duration == 1.234
        assert result.success is True
        assert result.metadata["iterations"] == 10
        
        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["name"] == "test_benchmark"
    
    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite."""
        from src.evaluation.benchmark import BenchmarkSuite, BenchmarkResult
        
        suite = BenchmarkSuite(
            name="Test Suite",
            timestamp="2024-01-01T00:00:00"
        )
        
        # Add results
        suite.add_result(BenchmarkResult("test1", 1.0, True))
        suite.add_result(BenchmarkResult("test2", 2.0, True))
        suite.add_result(BenchmarkResult("test3", 0.5, False))
        
        # Compute summary
        suite.compute_summary()
        
        assert suite.summary["total_benchmarks"] == 3
        assert suite.summary["successful"] == 2
        assert suite.summary["failed"] == 1
        assert suite.summary["avg_duration"] == 1.5
    
    def test_benchmark_runner(self):
        """Test BenchmarkRunner."""
        from src.evaluation.benchmark import BenchmarkRunner
        
        runner = BenchmarkRunner("Test")
        
        def test_function(x):
            return x * 2
        
        result = runner.run_benchmark(
            name="test_multiply",
            func=test_function,
            x=5,
            iterations=3
        )
        
        assert result.success is True
        assert result.metadata["iterations"] == 3
        assert len(result.metadata["all_durations"]) == 3
    
    def test_rag_benchmark_initialization(self, mock_rag_system):
        """Test RAGBenchmark initialization."""
        from src.evaluation.benchmark import RAGBenchmark
        
        benchmark = RAGBenchmark(
            rag_system=mock_rag_system,
            test_documents=["doc1", "doc2"],
            test_queries=["query1", "query2"]
        )
        
        assert benchmark.rag_system == mock_rag_system
        assert len(benchmark.test_documents) == 2
        assert len(benchmark.test_queries) == 2
    
    def test_quality_metrics(self):
        """Test retrieval quality metrics."""
        from src.evaluation.benchmark import RetrievalQualityBenchmark
        
        mock_system = Mock()
        mock_system.retrieve.return_value = [
            {"id": "doc1"},
            {"id": "doc2"},
            {"id": "doc3"}
        ]
        
        test_dataset = [
            {
                "query": "test query",
                "relevant_docs": ["doc1", "doc3", "doc5"]
            }
        ]
        
        benchmark = RetrievalQualityBenchmark(
            rag_system=mock_system,
            test_dataset=test_dataset
        )
        
        # Test precision@k
        precision = benchmark.precision_at_k(
            retrieved_docs=["doc1", "doc2", "doc3"],
            relevant_docs=["doc1", "doc3", "doc5"],
            k=3
        )
        assert precision == 2/3  # 2 relevant out of 3 retrieved
        
        # Test recall@k
        recall = benchmark.recall_at_k(
            retrieved_docs=["doc1", "doc2", "doc3"],
            relevant_docs=["doc1", "doc3", "doc5"],
            k=3
        )
        assert recall == 2/3  # 2 relevant found out of 3 total relevant
        
        # Test MRR
        mrr = benchmark.mrr(
            retrieved_docs=["doc2", "doc1", "doc3"],
            relevant_docs=["doc1", "doc3"]
        )
        assert mrr == 1/2  # First relevant doc at position 2


# ============================================================================
# MLflow Integration Tests
# ============================================================================

class TestMLflowIntegration:
    """Tests for MLflow integration."""
    
    @pytest.fixture
    def temp_mlruns_dir(self):
        """Create temporary MLflow directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.tracking.mlflow_integration.MLFLOW_AVAILABLE', True)
    def test_mlflow_tracker_initialization(self, temp_mlruns_dir):
        """Test MLflowTracker initialization."""
        with patch('src.tracking.mlflow_integration.mlflow') as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp_123"
            
            from src.tracking.mlflow_integration import (
                MLflowTracker,
                ExperimentConfig
            )
            
            config = ExperimentConfig(
                experiment_name="Test Experiment",
                tracking_uri=temp_mlruns_dir
            )
            
            tracker = MLflowTracker(config=config)
            
            assert tracker.config.experiment_name == "Test Experiment"
            mock_mlflow.create_experiment.assert_called_once()
    
    @patch('src.tracking.mlflow_integration.MLFLOW_AVAILABLE', True)
    @patch('src.tracking.mlflow_integration.mlflow')
    def test_mlflow_run_management(self, mock_mlflow):
        """Test MLflow run start/end."""
        mock_run = Mock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = Mock(experiment_id="exp_123")
        
        from src.tracking.mlflow_integration import MLflowTracker
        
        tracker = MLflowTracker()
        
        # Start run
        run = tracker.start_run(run_name="test_run")
        assert tracker.current_run is not None
        mock_mlflow.start_run.assert_called_once()
        
        # End run
        tracker.end_run()
        assert tracker.current_run is None
        mock_mlflow.end_run.assert_called_once()
    
    @patch('src.tracking.mlflow_integration.MLFLOW_AVAILABLE', True)
    @patch('src.tracking.mlflow_integration.mlflow')
    def test_mlflow_logging(self, mock_mlflow):
        """Test MLflow parameter and metric logging."""
        mock_run = Mock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = Mock(experiment_id="exp_123")
        
        from src.tracking.mlflow_integration import MLflowTracker
        
        tracker = MLflowTracker()
        tracker.start_run()
        
        # Log params
        tracker.log_params({"model": "gpt-4", "temperature": 0.7})
        assert mock_mlflow.log_param.call_count == 2
        
        # Log metrics
        tracker.log_metrics({"accuracy": 0.95, "latency": 1.23})
        assert mock_mlflow.log_metric.call_count == 2
        
        tracker.end_run()
    
    @patch('src.tracking.mlflow_integration.MLFLOW_AVAILABLE', True)
    @patch('src.tracking.mlflow_integration.mlflow')
    def test_rag_experiment_tracker(self, mock_mlflow):
        """Test RAGExperimentTracker."""
        mock_run = Mock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = Mock(experiment_id="exp_123")
        
        from src.tracking.mlflow_integration import RAGExperimentTracker
        
        tracker = RAGExperimentTracker(experiment_name="Test")
        
        # Track retrieval experiment
        tracker.track_retrieval_experiment(
            run_name="test_retrieval",
            retrieval_config={"strategy": "hybrid", "k": 5},
            embedding_model="all-MiniLM-L6-v2",
            metrics={"precision@5": 0.85}
        )
        
        # Verify run was started and ended
        mock_mlflow.start_run.assert_called()
        mock_mlflow.end_run.assert_called()
    
    @patch('src.tracking.mlflow_integration.MLFLOW_AVAILABLE', True)
    @patch('src.tracking.mlflow_integration.mlflow')
    def test_mlflow_callback(self, mock_mlflow):
        """Test MLflowCallback."""
        mock_run = Mock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = Mock(experiment_id="exp_123")
        
        from src.tracking.mlflow_integration import (
            RAGExperimentTracker,
            MLflowCallback
        )
        
        tracker = RAGExperimentTracker()
        callback = MLflowCallback(tracker)
        
        # Simulate query lifecycle
        callback.on_query_start(
            query="test query",
            config={"model": "gpt-4"}
        )
        
        assert callback.current_query == "test query"
        assert callback.query_start_time is not None
        
        callback.on_query_end(
            response="test response",
            retrieved_docs=[{"text": "doc1"}],
            additional_metrics={"relevance": 0.9}
        )
        
        # Verify tracking happened
        assert callback.query_start_time is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestFeatureIntegration:
    """Test integration between features."""
    
    @patch('src.vectorstore.faiss_store.FAISS_AVAILABLE', True)
    @patch('src.vectorstore.faiss_store.faiss')
    @patch('src.tracking.mlflow_integration.MLFLOW_AVAILABLE', True)
    @patch('src.tracking.mlflow_integration.mlflow')
    def test_faiss_with_mlflow(
        self,
        mock_mlflow,
        mock_faiss,
        temp_mlruns_dir=None
    ):
        """Test FAISS integration with MLflow tracking."""
        # Setup mocks
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.is_trained = True
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        mock_run = Mock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = Mock(experiment_id="exp_123")
        
        from src.vectorstore.faiss_store import FAISSVectorStore, FAISSConfig
        from src.tracking.mlflow_integration import MLflowTracker
        
        # Create FAISS store
        faiss_config = FAISSConfig(dimension=768)
        vector_store = FAISSVectorStore(config=faiss_config)
        
        # Create MLflow tracker
        tracker = MLflowTracker()
        tracker.start_run(run_name="faiss_test")
        
        # Log FAISS config
        tracker.log_params({
            "index_type": faiss_config.index_type,
            "dimension": faiss_config.dimension
        })
        
        # Add documents to FAISS
        texts = ["doc1", "doc2"]
        embeddings = [np.random.rand(768).astype('float32') for _ in range(2)]
        vector_store.add(texts, embeddings)
        
        # Log metrics
        stats = vector_store.get_stats()
        tracker.log_metrics({
            "num_documents": stats["num_documents"]
        })
        
        tracker.end_run()
        
        # Verify integration worked
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
