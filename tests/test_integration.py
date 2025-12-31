"""
Unit Tests for Integration Components

Tests for:
- RAG Orchestrator
- Response Cache
- Streaming
"""

import pytest
from unittest.mock import Mock, MagicMock
import time

from src.integration import (
    ResponseCache,
    RAGConfig,
    RetrievalStrategy,
    StreamEvent,
    StreamEventType
)


class TestResponseCache:
    """Tests for ResponseCache."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = ResponseCache(max_size=100, ttl=3600)
        
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert len(cache.cache) == 0
    
    def test_set_and_get(self):
        """Test setting and getting from cache."""
        cache = ResponseCache(max_size=10, ttl=3600)
        
        query = "What is RAG?"
        response = {"answer": "RAG is..."}
        
        # Set cache
        cache.set(query, response)
        
        # Get from cache
        cached = cache.get(query)
        
        assert cached is not None
        assert cached["answer"] == "RAG is..."
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = ResponseCache(max_size=10, ttl=3600)
        
        cached = cache.get("Non-existent query")
        
        assert cached is None
        assert cache.misses == 1
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = ResponseCache(max_size=10, ttl=1)  # 1 second TTL
        
        query = "Test query"
        response = {"answer": "Test"}
        
        cache.set(query, response)
        
        # Should be in cache
        assert cache.get(query) is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired
        cached = cache.get(query)
        assert cached is None
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = ResponseCache(max_size=3, ttl=0)  # No TTL
        
        # Fill cache
        cache.set("query1", {"answer": "1"})
        cache.set("query2", {"answer": "2"})
        cache.set("query3", {"answer": "3"})
        
        # Add one more (should evict oldest)
        cache.set("query4", {"answer": "4"})
        
        # query1 should be evicted
        assert cache.get("query1") is None
        assert cache.get("query4") is not None
        assert cache.evictions == 1
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = ResponseCache(max_size=10, ttl=0)
        
        cache.set("query1", {"answer": "1"})
        cache.set("query2", {"answer": "2"})
        
        # Invalidate specific query
        cache.invalidate(query="query1")
        
        assert cache.get("query1") is None
        assert cache.get("query2") is not None
    
    def test_clear_cache(self):
        """Test clearing cache."""
        cache = ResponseCache(max_size=10, ttl=0)
        
        cache.set("query1", {"answer": "1"})
        cache.set("query2", {"answer": "2"})
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.current_memory_bytes == 0
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = ResponseCache(max_size=10, ttl=0)
        
        cache.set("query1", {"answer": "1"})
        cache.get("query1")  # Hit
        cache.get("query2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestRAGConfig:
    """Tests for RAGConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RAGConfig()
        
        assert config.retrieval_strategy == RetrievalStrategy.SIMPLE
        assert config.top_k == 5
        assert config.use_reranking == True
        assert config.use_conversation_memory == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RAGConfig(
            retrieval_strategy=RetrievalStrategy.MULTI_QUERY,
            top_k=10,
            num_queries=5
        )
        
        assert config.retrieval_strategy == RetrievalStrategy.MULTI_QUERY
        assert config.top_k == 10
        assert config.num_queries == 5


class TestStreamEvent:
    """Tests for StreamEvent."""
    
    def test_stream_event_creation(self):
        """Test creating stream event."""
        event = StreamEvent(
            type=StreamEventType.START,
            data={"query": "Test"}
        )
        
        assert event.type == StreamEventType.START
        assert event.data["query"] == "Test"
        assert event.timestamp is not None
    
    def test_stream_event_to_json(self):
        """Test converting event to JSON."""
        event = StreamEvent(
            type=StreamEventType.TOKEN,
            data="Hello"
        )
        
        json_str = event.to_json()
        
        assert "TOKEN" in json_str
        assert "Hello" in json_str
    
    def test_stream_event_to_sse(self):
        """Test converting event to SSE format."""
        event = StreamEvent(
            type=StreamEventType.END,
            data={}
        )
        
        sse_str = event.to_sse()
        
        assert sse_str.startswith("data:")
        assert sse_str.endswith("\n\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
