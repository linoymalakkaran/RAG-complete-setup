"""
Response Cache - Cache RAG responses for performance

Caches query responses to avoid redundant:
- LLM calls
- Vector searches
- Document retrievals

Supports:
- TTL-based expiration
- Size limits
- Cache invalidation
- Hit/miss statistics
"""

from typing import Dict, Any, Optional, Callable
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from collections import OrderedDict
import logging

from src.utils.logging_config import RAGLogger


@dataclass
class CacheEntry:
    """Cached response entry."""
    
    query_hash: str
    response: Dict[str, Any]
    timestamp: float
    hit_count: int = 0
    size_bytes: int = 0


class ResponseCache:
    """
    LRU cache for RAG responses with TTL.
    
    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Size-based limits
    - Query normalization
    - Hit/miss statistics
    
    Example:
        >>> cache = ResponseCache(max_size=100, ttl=3600)
        >>> 
        >>> # Check cache
        >>> cached = cache.get("What is RAG?")
        >>> if cached:
        >>>     return cached
        >>> 
        >>> # ... generate response ...
        >>> 
        >>> # Store in cache
        >>> cache.set("What is RAG?", response)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,  # 1 hour
        max_memory_mb: int = 100
    ):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live in seconds (0 = no expiration)
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl = ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory_bytes = 0
        
        self.logger = RAGLogger.get_logger("response_cache")
        
        self.logger.info(
            f"Initialized cache (max_size={max_size}, ttl={ttl}s, "
            f"max_memory={max_memory_mb}MB)"
        )
    
    def get(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Args:
            query: User query
            conversation_id: Optional conversation context
            metadata: Optional query metadata
            
        Returns:
            Cached response or None if not found/expired
        """
        # Generate cache key
        cache_key = self._generate_key(query, conversation_id, metadata)
        
        # Check if exists
        if cache_key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[cache_key]
        
        # Check expiration
        if self.ttl > 0 and (time.time() - entry.timestamp) > self.ttl:
            # Expired - remove
            self._remove_entry(cache_key)
            self.misses += 1
            self.logger.debug(f"Cache expired for query: {query[:50]}...")
            return None
        
        # Cache hit - move to end (most recently used)
        self.cache.move_to_end(cache_key)
        entry.hit_count += 1
        self.hits += 1
        
        self.logger.debug(
            f"Cache HIT (hit_count={entry.hit_count}): {query[:50]}..."
        )
        
        return entry.response
    
    def set(
        self,
        query: str,
        response: Dict[str, Any],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache response.
        
        Args:
            query: User query
            response: Response to cache
            conversation_id: Optional conversation context
            metadata: Optional query metadata
        """
        # Generate cache key
        cache_key = self._generate_key(query, conversation_id, metadata)
        
        # Calculate size
        size_bytes = len(json.dumps(response).encode('utf-8'))
        
        # Check if too large
        if size_bytes > self.max_memory_bytes:
            self.logger.warning(
                f"Response too large to cache ({size_bytes} bytes): {query[:50]}..."
            )
            return
        
        # Evict if necessary
        while (
            len(self.cache) >= self.max_size or
            self.current_memory_bytes + size_bytes > self.max_memory_bytes
        ):
            if not self.cache:
                break
            self._evict_lru()
        
        # Create entry
        entry = CacheEntry(
            query_hash=cache_key,
            response=response,
            timestamp=time.time(),
            hit_count=0,
            size_bytes=size_bytes
        )
        
        # Add to cache
        self.cache[cache_key] = entry
        self.current_memory_bytes += size_bytes
        
        self.logger.debug(f"Cached response ({size_bytes} bytes): {query[:50]}...")
    
    def invalidate(
        self,
        query: Optional[str] = None,
        conversation_id: Optional[str] = None,
        pattern: Optional[str] = None
    ):
        """
        Invalidate cache entries.
        
        Args:
            query: Specific query to invalidate
            conversation_id: Invalidate all for conversation
            pattern: Invalidate matching pattern (prefix)
        """
        if query:
            # Invalidate specific query
            cache_key = self._generate_key(query, conversation_id)
            if cache_key in self.cache:
                self._remove_entry(cache_key)
                self.logger.info(f"Invalidated cache for: {query[:50]}...")
        
        elif conversation_id:
            # Invalidate all for conversation
            keys_to_remove = [
                k for k in self.cache.keys()
                if conversation_id in k
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
            
            self.logger.info(
                f"Invalidated {len(keys_to_remove)} entries for conversation: {conversation_id}"
            )
        
        elif pattern:
            # Invalidate by pattern
            keys_to_remove = [
                k for k in self.cache.keys()
                if k.startswith(pattern)
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
            
            self.logger.info(
                f"Invalidated {len(keys_to_remove)} entries matching: {pattern}"
            )
    
    def clear(self):
        """Clear entire cache."""
        count = len(self.cache)
        self.cache.clear()
        self.current_memory_bytes = 0
        
        self.logger.info(f"Cleared cache ({count} entries)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        # Get top queries by hit count
        top_queries = sorted(
            self.cache.values(),
            key=lambda x: x.hit_count,
            reverse=True
        )[:5]
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_mb": self.current_memory_bytes / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "top_queries": [
                {
                    "hash": entry.query_hash[:16],
                    "hit_count": entry.hit_count,
                    "age_seconds": int(time.time() - entry.timestamp)
                }
                for entry in top_queries
            ]
        }
    
    def _generate_key(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key from query and context."""
        
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Include conversation context if provided
        if conversation_id:
            key_data = f"{normalized_query}|{conversation_id}"
        else:
            key_data = normalized_query
        
        # Include relevant metadata
        if metadata:
            # Only include metadata that affects response
            relevant_metadata = {
                k: v for k, v in metadata.items()
                if k in ['model', 'temperature', 'top_k']
            }
            if relevant_metadata:
                key_data += f"|{json.dumps(relevant_metadata, sort_keys=True)}"
        
        # Hash for fixed-length key
        return hashlib.sha256(key_data.encode('utf-8')).hexdigest()
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Remove first item (least recently used)
        key, entry = self.cache.popitem(last=False)
        self.current_memory_bytes -= entry.size_bytes
        self.evictions += 1
        
        self.logger.debug(
            f"Evicted LRU entry (hit_count={entry.hit_count}, "
            f"age={int(time.time() - entry.timestamp)}s)"
        )
    
    def _remove_entry(self, key: str):
        """Remove specific entry."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory_bytes -= entry.size_bytes


class SemanticCache(ResponseCache):
    """
    Semantic cache using embedding similarity.
    
    Instead of exact query matching, finds semantically similar
    cached queries using embeddings.
    
    Requires: embedding function for query similarity
    """
    
    def __init__(
        self,
        embedding_function: Callable[[str], list],
        similarity_threshold: float = 0.95,
        **kwargs
    ):
        """
        Initialize semantic cache.
        
        Args:
            embedding_function: Function to generate query embeddings
            similarity_threshold: Minimum similarity for cache hit (0-1)
            **kwargs: Additional args for ResponseCache
        """
        super().__init__(**kwargs)
        
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        
        # Store embeddings for cached queries
        self.embeddings: Dict[str, list] = {}
        
        self.logger.info(
            f"Initialized semantic cache "
            f"(threshold={similarity_threshold})"
        )
    
    def get(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response using semantic similarity.
        
        Args:
            query: User query
            conversation_id: Optional conversation context
            metadata: Optional query metadata
            
        Returns:
            Cached response or None
        """
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Find most similar cached query
        best_similarity = 0.0
        best_key = None
        
        for cache_key, cached_embedding in self.embeddings.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(
                query_embedding,
                cached_embedding
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = cache_key
        
        # Check if above threshold
        if best_similarity >= self.similarity_threshold and best_key:
            # Semantic cache hit
            entry = self.cache.get(best_key)
            
            if entry:
                # Check expiration
                if self.ttl > 0 and (time.time() - entry.timestamp) > self.ttl:
                    self._remove_entry(best_key)
                    self.misses += 1
                    return None
                
                # Move to end and increment hit count
                self.cache.move_to_end(best_key)
                entry.hit_count += 1
                self.hits += 1
                
                self.logger.debug(
                    f"Semantic cache HIT (similarity={best_similarity:.3f}): "
                    f"{query[:50]}..."
                )
                
                return entry.response
        
        self.misses += 1
        return None
    
    def set(
        self,
        query: str,
        response: Dict[str, Any],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Cache response with query embedding."""
        
        # Generate cache key and embedding
        cache_key = self._generate_key(query, conversation_id, metadata)
        query_embedding = self.embedding_function(query)
        
        # Store embedding
        self.embeddings[cache_key] = query_embedding
        
        # Call parent set
        super().set(query, response, conversation_id, metadata)
    
    def _remove_entry(self, key: str):
        """Remove entry and its embedding."""
        super()._remove_entry(key)
        if key in self.embeddings:
            del self.embeddings[key]
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":
    # Example usage
    cache = ResponseCache(max_size=100, ttl=3600)
    
    # Simulate caching
    query1 = "What is RAG?"
    response1 = {"answer": "RAG stands for Retrieval Augmented Generation"}
    
    # Set cache
    cache.set(query1, response1)
    
    # Get from cache
    cached = cache.get(query1)
    print(f"Cached response: {cached}")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
