"""
Integration Module - Orchestration, caching, and streaming

Provides integration components:
- RAG orchestrator for complete pipeline
- Response caching (LRU and semantic)
- Streaming responses
"""

from src.integration.orchestrator import (
    RAGOrchestrator,
    RAGConfig,
    RetrievalStrategy
)

from src.integration.cache import (
    ResponseCache,
    SemanticCache,
    CacheEntry
)

from src.integration.streaming import (
    StreamingRAG,
    StreamEvent,
    StreamEventType,
    SSEFormatter,
    StreamBuffer
)


__all__ = [
    # Orchestrator
    "RAGOrchestrator",
    "RAGConfig",
    "RetrievalStrategy",
    
    # Cache
    "ResponseCache",
    "SemanticCache",
    "CacheEntry",
    
    # Streaming
    "StreamingRAG",
    "StreamEvent",
    "StreamEventType",
    "SSEFormatter",
    "StreamBuffer"
]
