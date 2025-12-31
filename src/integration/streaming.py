"""
Streaming Response Handler - Stream RAG responses in real-time

Provides streaming capabilities for:
- Incremental response generation
- Real-time UI updates
- Server-sent events (SSE)
- WebSocket streaming
"""

from typing import Iterator, Dict, Any, Optional, Callable
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

from src.llm.openai_client import OpenAIClient
from src.utils.logging_config import RAGLogger


class StreamEventType(Enum):
    """Types of streaming events."""
    START = "start"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    GENERATION_START = "generation_start"
    TOKEN = "token"
    SOURCES = "sources"
    END = "end"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Streaming event."""
    
    type: StreamEventType
    data: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        })
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        return f"data: {self.to_json()}\n\n"


class StreamingRAG:
    """
    Streaming RAG pipeline.
    
    Streams RAG responses in real-time with events for:
    - Query processing started
    - Documents retrieved
    - Context built
    - Response generation started
    - Each generated token
    - Sources included
    - Query completed
    
    Example:
        >>> streaming_rag = StreamingRAG(orchestrator)
        >>> 
        >>> for event in streaming_rag.stream_query("What is RAG?"):
        >>>     if event.type == StreamEventType.TOKEN:
        >>>         print(event.data, end="", flush=True)
        >>>     elif event.type == StreamEventType.SOURCES:
        >>>         print(f"\\nSources: {event.data}")
    """
    
    def __init__(
        self,
        orchestrator: Any,
        llm_client: Optional[OpenAIClient] = None
    ):
        """
        Initialize streaming RAG.
        
        Args:
            orchestrator: RAG orchestrator instance
            llm_client: OpenAI client for streaming generation
        """
        self.orchestrator = orchestrator
        self.llm_client = llm_client or OpenAIClient()
        
        self.logger = RAGLogger.get_logger("streaming_rag")
    
    def stream_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[StreamEvent]:
        """
        Stream RAG query processing.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID
            metadata: Optional metadata
            
        Yields:
            StreamEvent objects
        """
        try:
            # Event: Start
            yield StreamEvent(
                type=StreamEventType.START,
                data={"query": query}
            )
            
            # Create conversation if needed
            if conversation_id is None and self.orchestrator.config.use_conversation_memory:
                conversation_id = self.orchestrator.memory.create_conversation()
            
            # Step 1: Query enhancement
            enhanced_query = self.orchestrator._enhance_query(query)
            
            # Step 2: Retrieve documents
            retrieved_docs = self.orchestrator._retrieve_documents(enhanced_query)
            
            # Event: Retrieval complete
            yield StreamEvent(
                type=StreamEventType.RETRIEVAL,
                data={
                    "num_documents": len(retrieved_docs),
                    "query": enhanced_query
                }
            )
            
            # Step 3: Rerank
            if self.orchestrator.config.use_reranking and self.orchestrator.reranker:
                reranked = self.orchestrator._rerank_documents(
                    enhanced_query,
                    retrieved_docs
                )
            else:
                reranked = retrieved_docs
            
            # Step 4: Build context
            context = self.orchestrator._build_context(
                enhanced_query,
                reranked,
                conversation_id
            )
            
            # Event: Context built
            yield StreamEvent(
                type=StreamEventType.CONTEXT,
                data={
                    "context_length": len(context),
                    "num_sources": len(reranked[:self.orchestrator.config.top_k])
                }
            )
            
            # Step 5: Stream generation
            yield StreamEvent(
                type=StreamEventType.GENERATION_START,
                data={}
            )
            
            # Build prompt
            prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {enhanced_query}

Answer:"""
            
            system_prompt = """You are a helpful AI assistant. Answer based on the provided context.
If the context doesn't contain enough information, say so clearly."""
            
            # Stream tokens
            full_response = ""
            for token in self._stream_generation(prompt, system_prompt):
                full_response += token
                
                yield StreamEvent(
                    type=StreamEventType.TOKEN,
                    data=token
                )
            
            # Step 6: Send sources
            sources = self.orchestrator._format_sources(
                reranked[:self.orchestrator.config.top_k]
            )
            
            yield StreamEvent(
                type=StreamEventType.SOURCES,
                data=sources
            )
            
            # Step 7: Update conversation memory
            if conversation_id and self.orchestrator.memory:
                self.orchestrator.memory.add_message(
                    conversation_id,
                    "user",
                    query
                )
                self.orchestrator.memory.add_message(
                    conversation_id,
                    "assistant",
                    full_response
                )
            
            # Event: End
            yield StreamEvent(
                type=StreamEventType.END,
                data={
                    "conversation_id": conversation_id,
                    "total_tokens": len(full_response.split())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)}
            )
    
    def _stream_generation(
        self,
        prompt: str,
        system_prompt: str
    ) -> Iterator[str]:
        """
        Stream LLM generation.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Yields:
            Generated tokens
        """
        # Check if LLM client supports streaming
        if hasattr(self.llm_client, 'stream_generate'):
            # Use native streaming
            for token in self.llm_client.stream_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.orchestrator.config.model,
                temperature=self.orchestrator.config.temperature,
                max_tokens=self.orchestrator.config.max_tokens
            ):
                yield token
        else:
            # Fallback: Generate full response and simulate streaming
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.orchestrator.config.model,
                temperature=self.orchestrator.config.temperature,
                max_tokens=self.orchestrator.config.max_tokens
            )
            
            # Simulate token-by-token streaming
            words = response.split()
            for word in words:
                yield word + " "
                time.sleep(0.01)  # Small delay for realism


class SSEFormatter:
    """
    Server-Sent Events formatter for streaming.
    
    Formats StreamEvent objects as SSE for web clients.
    """
    
    @staticmethod
    def format_event(event: StreamEvent) -> str:
        """Format event as SSE."""
        return event.to_sse()
    
    @staticmethod
    def format_stream(
        events: Iterator[StreamEvent]
    ) -> Iterator[str]:
        """
        Format event stream as SSE.
        
        Args:
            events: Iterator of StreamEvent objects
            
        Yields:
            SSE-formatted strings
        """
        for event in events:
            yield SSEFormatter.format_event(event)


class StreamBuffer:
    """
    Buffer for streaming responses.
    
    Buffers tokens and emits when:
    - Buffer reaches threshold
    - Sentence boundary detected
    - Timeout reached
    """
    
    def __init__(
        self,
        buffer_size: int = 5,
        timeout_ms: int = 100
    ):
        """
        Initialize stream buffer.
        
        Args:
            buffer_size: Number of tokens to buffer
            timeout_ms: Max wait time in milliseconds
        """
        self.buffer_size = buffer_size
        self.timeout_ms = timeout_ms
        
        self.buffer = []
        self.last_emit = time.time()
    
    def add_token(self, token: str) -> Optional[str]:
        """
        Add token to buffer.
        
        Args:
            token: Token to add
            
        Returns:
            Buffered text if ready to emit, None otherwise
        """
        self.buffer.append(token)
        
        # Check if should emit
        should_emit = (
            len(self.buffer) >= self.buffer_size or
            self._is_sentence_boundary(token) or
            (time.time() - self.last_emit) * 1000 >= self.timeout_ms
        )
        
        if should_emit:
            return self.flush()
        
        return None
    
    def flush(self) -> str:
        """Flush buffer and return contents."""
        if not self.buffer:
            return ""
        
        text = "".join(self.buffer)
        self.buffer = []
        self.last_emit = time.time()
        
        return text
    
    def _is_sentence_boundary(self, token: str) -> bool:
        """Check if token is sentence boundary."""
        return token.strip() in {'.', '!', '?', '\n'}


if __name__ == "__main__":
    # Example usage
    print("Streaming RAG - Example\n")
    print("Note: Requires orchestrator to be initialized")
    print("\nExample:")
    print("  streaming_rag = StreamingRAG(orchestrator)")
    print("  for event in streaming_rag.stream_query('What is RAG?'):")
    print("      if event.type == StreamEventType.TOKEN:")
    print("          print(event.data, end='', flush=True)")
    print("      elif event.type == StreamEventType.SOURCES:")
    print("          print(f'\\nSources: {event.data}')")
