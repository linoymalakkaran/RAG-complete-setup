"""
FastAPI Server - Production RAG API

Provides REST API endpoints for:
- Query processing
- Conversation management
- Document upload
- Metrics and monitoring
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime

from src.integration.orchestrator import RAGOrchestrator, RAGConfig, RetrievalStrategy
from src.integration.cache import ResponseCache
from src.integration.streaming import StreamingRAG, SSEFormatter
from src.utils.logging_config import RAGLogger


# Pydantic models for API
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="User query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional config overrides")
    use_cache: bool = Field(True, description="Whether to use cache")
    stream: bool = Field(False, description="Stream response")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: Optional[str]
    metadata: Dict[str, Any]
    cached: bool = False


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    conversation_id: str
    messages: List[Dict[str, Any]]
    message_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class StatsResponse(BaseModel):
    """Statistics response."""
    cache_stats: Dict[str, Any]
    total_queries: int
    active_conversations: int


# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="Production RAG API with advanced features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
orchestrator: Optional[RAGOrchestrator] = None
cache: Optional[ResponseCache] = None
streaming_rag: Optional[StreamingRAG] = None
logger: Optional[RAGLogger] = None

# Statistics
total_queries = 0


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global orchestrator, cache, streaming_rag, logger
    
    logger = RAGLogger.get_logger("api")
    logger.info("Starting RAG API server...")
    
    # Create config
    config = RAGConfig(
        retrieval_strategy=RetrievalStrategy.MULTI_QUERY,
        use_reranking=True,
        use_conversation_memory=True
    )
    
    # Note: In production, initialize with actual vector store
    # orchestrator = RAGOrchestrator(vector_store, config)
    
    # Initialize cache
    cache = ResponseCache(max_size=1000, ttl=3600)
    
    # Note: Uncomment when orchestrator is initialized
    # streaming_rag = StreamingRAG(orchestrator)
    
    logger.info("RAG API server started successfully")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process RAG query.
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and sources
    """
    global total_queries
    
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG orchestrator not initialized. Please configure vector store."
        )
    
    try:
        total_queries += 1
        
        # Check cache if enabled
        cached_response = None
        if request.use_cache and cache:
            cached_response = cache.get(
                request.query,
                conversation_id=request.conversation_id
            )
        
        if cached_response:
            logger.info(f"Cache hit for query: {request.query[:50]}...")
            return QueryResponse(
                **cached_response,
                cached=True
            )
        
        # Process query
        response = orchestrator.query(
            query=request.query,
            conversation_id=request.conversation_id,
            metadata=request.config
        )
        
        # Cache response
        if request.use_cache and cache:
            cache.set(
                request.query,
                response,
                conversation_id=request.conversation_id
            )
        
        return QueryResponse(**response, cached=False)
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream RAG query response.
    
    Args:
        request: Query request
        
    Returns:
        Server-Sent Events stream
    """
    if streaming_rag is None:
        raise HTTPException(
            status_code=503,
            detail="Streaming not available. Please configure orchestrator."
        )
    
    try:
        # Create event stream
        events = streaming_rag.stream_query(
            query=request.query,
            conversation_id=request.conversation_id,
            metadata=request.config
        )
        
        # Format as SSE
        sse_stream = SSEFormatter.format_stream(events)
        
        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(conversation_id: str):
    """
    Get conversation history.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Conversation history
    """
    if orchestrator is None or not orchestrator.memory:
        raise HTTPException(
            status_code=503,
            detail="Conversation memory not available"
        )
    
    try:
        messages = orchestrator.get_conversation_history(conversation_id)
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            message_count=len(messages)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Success message
    """
    if orchestrator is None or not orchestrator.memory:
        raise HTTPException(
            status_code=503,
            detail="Conversation memory not available"
        )
    
    try:
        orchestrator.memory.delete_conversation(conversation_id)
        
        # Invalidate cache for this conversation
        if cache:
            cache.invalidate(conversation_id=conversation_id)
        
        return {"message": f"Conversation {conversation_id} deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and index document.
    
    Args:
        file: Document file
        background_tasks: Background tasks
        
    Returns:
        Upload status
    """
    # Note: Implement document processing
    # This would:
    # 1. Extract text from file
    # 2. Chunk the text
    # 3. Generate embeddings
    # 4. Add to vector store
    
    raise HTTPException(
        status_code=501,
        detail="Document upload not yet implemented"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get API statistics.
    
    Returns:
        Statistics including cache stats
    """
    cache_stats = cache.get_stats() if cache else {}
    
    active_conversations = 0
    if orchestrator and orchestrator.memory:
        active_conversations = len(orchestrator.memory.conversations)
    
    return StatsResponse(
        cache_stats=cache_stats,
        total_queries=total_queries,
        active_conversations=active_conversations
    )


@app.post("/cache/clear")
async def clear_cache():
    """Clear response cache."""
    if cache:
        cache.clear()
        return {"message": "Cache cleared"}
    
    raise HTTPException(status_code=503, detail="Cache not available")


@app.post("/cache/invalidate")
async def invalidate_cache(
    query: Optional[str] = None,
    conversation_id: Optional[str] = None,
    pattern: Optional[str] = None
):
    """
    Invalidate cache entries.
    
    Args:
        query: Specific query to invalidate
        conversation_id: Invalidate all for conversation
        pattern: Invalidate matching pattern
        
    Returns:
        Success message
    """
    if not cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    cache.invalidate(
        query=query,
        conversation_id=conversation_id,
        pattern=pattern
    )
    
    return {"message": "Cache invalidated"}


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
