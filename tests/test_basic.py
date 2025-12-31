"""
Basic unit tests for RAG components
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.chunking.chunking_strategies import (
    FixedSizeChunking,
    RecursiveChunking,
    ChunkerFactory
)


class TestChunking:
    """Test chunking strategies"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_text = """
        This is a test document.
        
        It has multiple paragraphs.
        
        Each paragraph should be handled properly by the chunking strategy.
        We want to ensure that chunks are created correctly.
        """
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking"""
        chunker = FixedSizeChunking(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(self.sample_text, metadata={'test': 'value'})
        
        assert len(chunks) > 0
        assert all(len(c.content) <= 60 for c in chunks)  # Allow some variance
        assert all(c.metadata['test'] == 'value' for c in chunks)
    
    def test_recursive_chunking(self):
        """Test recursive chunking"""
        chunker = RecursiveChunking(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(self.sample_text, metadata={'doc_id': 'test1'})
        
        assert len(chunks) > 0
        assert all(c.chunk_id is not None for c in chunks)
        assert all('doc_id' in c.metadata for c in chunks)
    
    def test_chunker_factory(self):
        """Test chunker factory"""
        # Test fixed strategy
        fixed = ChunkerFactory.create('fixed', chunk_size=100)
        assert isinstance(fixed, FixedSizeChunking)
        
        # Test recursive strategy
        recursive = ChunkerFactory.create('recursive', chunk_size=200)
        assert isinstance(recursive, RecursiveChunking)
        
        # Test invalid strategy
        with pytest.raises(ValueError):
            ChunkerFactory.create('invalid_strategy')
    
    def test_chunk_metadata(self):
        """Test chunk metadata propagation"""
        chunker = RecursiveChunking()
        metadata = {
            'source': 'test.txt',
            'category': 'test',
            'doc_id': '123'
        }
        
        chunks = chunker.chunk(self.sample_text, metadata=metadata)
        
        for chunk in chunks:
            assert chunk.metadata['source'] == 'test.txt'
            assert chunk.metadata['category'] == 'test'
            assert chunk.metadata['doc_id'] == '123'
            assert 'chunk_index' in chunk.metadata


class TestEmbeddings:
    """Test embedding providers"""
    
    def test_embedding_factory(self):
        """Test embedding factory creation"""
        from src.embeddings.providers.embedding_providers import EmbeddingFactory
        
        # Test local model creation (doesn't require API key)
        embedder = EmbeddingFactory.create('local', model='all-MiniLM-L6-v2')
        
        assert embedder.dimension > 0
        assert embedder.model_name is not None
    
    def test_local_embedding(self):
        """Test local embedding generation"""
        from src.embeddings.providers.embedding_providers import LocalEmbedding
        
        embedder = LocalEmbedding(model='all-MiniLM-L6-v2')
        
        # Test single embedding
        text = "This is a test"
        embedding = embedder.embed_text(text)
        
        assert len(embedding) == embedder.dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embedding(self):
        """Test batch embedding"""
        from src.embeddings.providers.embedding_providers import LocalEmbedding
        
        embedder = LocalEmbedding(model='all-MiniLM-L6-v2')
        
        texts = ["Test 1", "Test 2", "Test 3"]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(e) == embedder.dimension for e in embeddings)


class TestHybridSearch:
    """Test hybrid search functionality"""
    
    def test_bm25_retriever(self):
        """Test BM25 retriever"""
        from src.embeddings.hybrid import BM25Retriever
        
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a great programming language",
            "Machine learning is a subset of artificial intelligence"
        ]
        doc_ids = ["doc1", "doc2", "doc3"]
        metadata = [{'id': i} for i in range(3)]
        
        retriever = BM25Retriever(documents, doc_ids, metadata)
        
        # Test search
        results = retriever.search("python programming", top_k=2)
        
        assert len(results) <= 2
        assert all(r.doc_id in doc_ids for r in results)
        assert all(r.score > 0 for r in results)


def test_config_manager():
    """Test configuration manager"""
    from src.utils.config_manager import ConfigManager
    
    config = ConfigManager()
    
    # Test getting values
    assert config.get('app.name') is not None
    assert config.get('embeddings.dimension') is not None
    
    # Test default values
    assert config.get('nonexistent.key', 'default') == 'default'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
