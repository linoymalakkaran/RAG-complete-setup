"""
Conversation Buffer - Token-aware context window management

Manages conversation buffers with:
- Token counting
- Window sliding
- Lost-in-middle mitigation
- Smart truncation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from src.context.memory import Message, ConversationMemory
from src.utils.logging_config import RAGLogger


@dataclass
class BufferConfig:
    """Configuration for conversation buffer."""
    
    max_tokens: int = 4000
    reserve_tokens: int = 1000  # Reserve for response
    include_system_prompt: bool = True
    truncation_strategy: str = "sliding"  # 'sliding', 'summarize', 'middle'
    preserve_recent: int = 3  # Always keep N most recent messages


class ConversationBuffer:
    """
    Token-aware conversation buffer management.
    
    Manages context window with awareness of token limits:
    - Counts tokens in conversation
    - Slides window when limit reached
    - Implements lost-in-middle mitigation
    - Smart truncation strategies
    
    Example:
        >>> config = BufferConfig(max_tokens=4000)
        >>> buffer = ConversationBuffer(memory, config)
        >>> 
        >>> # Get buffered context
        >>> context = buffer.get_buffered_context(conv_id)
        >>> 
        >>> # Check if room for new message
        >>> can_add = buffer.can_add_message(conv_id, new_message)
    """
    
    def __init__(
        self,
        memory: ConversationMemory,
        config: Optional[BufferConfig] = None
    ):
        """
        Initialize conversation buffer.
        
        Args:
            memory: ConversationMemory instance
            config: Buffer configuration
        """
        self.memory = memory
        self.config = config or BufferConfig()
        
        self.logger = RAGLogger.get_logger("conversation_buffer")
        
        self.logger.info(
            f"Initialized conversation buffer "
            f"(max_tokens={self.config.max_tokens}, "
            f"strategy={self.config.truncation_strategy})"
        )
    
    def get_buffered_context(
        self,
        conversation_id: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get conversation context within token limits.
        
        Args:
            conversation_id: Conversation ID
            system_prompt: Optional system prompt
            
        Returns:
            Dict with:
                - messages: List of messages that fit in window
                - tokens_used: Total tokens
                - truncated: Whether truncation occurred
                - strategy_used: Truncation strategy applied
        """
        messages = self.memory.get_messages(conversation_id)
        
        if not messages:
            return {
                "messages": [],
                "tokens_used": 0,
                "truncated": False,
                "strategy_used": None
            }
        
        # Calculate available tokens
        available_tokens = self.config.max_tokens - self.config.reserve_tokens
        
        # Account for system prompt
        system_tokens = 0
        if system_prompt and self.config.include_system_prompt:
            system_tokens = self._estimate_tokens(system_prompt)
            available_tokens -= system_tokens
        
        # Apply truncation strategy
        if self.config.truncation_strategy == "sliding":
            selected_messages, truncated = self._sliding_window(messages, available_tokens)
        elif self.config.truncation_strategy == "summarize":
            selected_messages, truncated = self._summarize_old(messages, available_tokens)
        elif self.config.truncation_strategy == "middle":
            selected_messages, truncated = self._keep_ends(messages, available_tokens)
        else:
            selected_messages = messages
            truncated = False
        
        total_tokens = sum(self._estimate_tokens(m.content) for m in selected_messages)
        total_tokens += system_tokens
        
        return {
            "messages": selected_messages,
            "tokens_used": total_tokens,
            "truncated": truncated,
            "strategy_used": self.config.truncation_strategy if truncated else None
        }
    
    def can_add_message(
        self,
        conversation_id: str,
        message_content: str
    ) -> bool:
        """
        Check if new message can be added without exceeding limits.
        
        Args:
            conversation_id: Conversation ID
            message_content: New message content
            
        Returns:
            True if message fits in window
        """
        buffered = self.get_buffered_context(conversation_id)
        current_tokens = buffered["tokens_used"]
        new_tokens = self._estimate_tokens(message_content)
        
        # Account for reserve tokens
        total_tokens = current_tokens + new_tokens + self.config.reserve_tokens
        
        return total_tokens <= self.config.max_tokens
    
    def get_token_stats(self, conversation_id: str) -> Dict[str, int]:
        """
        Get token usage statistics.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dict with token statistics
        """
        messages = self.memory.get_messages(conversation_id)
        
        if not messages:
            return {
                "total_tokens": 0,
                "message_count": 0,
                "avg_tokens_per_message": 0,
                "available_tokens": self.config.max_tokens - self.config.reserve_tokens
            }
        
        message_tokens = [self._estimate_tokens(m.content) for m in messages]
        total_tokens = sum(message_tokens)
        
        return {
            "total_tokens": total_tokens,
            "message_count": len(messages),
            "avg_tokens_per_message": total_tokens // len(messages) if messages else 0,
            "available_tokens": self.config.max_tokens - self.config.reserve_tokens - total_tokens,
            "max_tokens": self.config.max_tokens,
            "reserve_tokens": self.config.reserve_tokens
        }
    
    def _sliding_window(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> tuple[List[Message], bool]:
        """
        Sliding window: Keep most recent messages.
        
        Args:
            messages: All messages
            max_tokens: Maximum tokens allowed
            
        Returns:
            (selected_messages, was_truncated)
        """
        # Always preserve recent messages
        preserve_count = min(self.config.preserve_recent, len(messages))
        
        # Work backwards from most recent
        selected = []
        current_tokens = 0
        
        for message in reversed(messages):
            msg_tokens = self._estimate_tokens(message.content)
            
            if current_tokens + msg_tokens <= max_tokens:
                selected.insert(0, message)
                current_tokens += msg_tokens
            elif len(selected) < preserve_count:
                # Force include if in preserve window
                selected.insert(0, message)
                current_tokens += msg_tokens
            else:
                # Can't fit more
                break
        
        truncated = len(selected) < len(messages)
        
        if truncated:
            self.logger.debug(
                f"Sliding window: kept {len(selected)}/{len(messages)} messages "
                f"({current_tokens} tokens)"
            )
        
        return selected, truncated
    
    def _summarize_old(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> tuple[List[Message], bool]:
        """
        Summarize old messages, keep recent ones.
        
        Args:
            messages: All messages
            max_tokens: Maximum tokens allowed
            
        Returns:
            (selected_messages, was_truncated)
        """
        # Keep last N messages as-is
        preserve_count = self.config.preserve_recent
        recent_messages = messages[-preserve_count:]
        old_messages = messages[:-preserve_count]
        
        # Calculate tokens for recent messages
        recent_tokens = sum(self._estimate_tokens(m.content) for m in recent_messages)
        available_for_old = max_tokens - recent_tokens
        
        if not old_messages or available_for_old <= 0:
            return recent_messages, len(old_messages) > 0
        
        # In production, use LLM to summarize old messages
        # For now, create a simple summary
        summary_text = f"[Summary of {len(old_messages)} earlier messages]"
        summary_tokens = self._estimate_tokens(summary_text)
        
        if summary_tokens <= available_for_old:
            # Create summary message
            summary_msg = Message(
                role="system",
                content=summary_text,
                metadata={"is_summary": True, "summarized_count": len(old_messages)}
            )
            return [summary_msg] + recent_messages, True
        else:
            # Can't fit summary, just return recent
            return recent_messages, True
    
    def _keep_ends(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> tuple[List[Message], bool]:
        """
        Lost-in-middle mitigation: Keep first and last messages.
        
        This addresses the "lost in the middle" problem where LLMs
        pay less attention to middle parts of context.
        
        Args:
            messages: All messages
            max_tokens: Maximum tokens allowed
            
        Returns:
            (selected_messages, was_truncated)
        """
        if len(messages) <= self.config.preserve_recent * 2:
            # Too few messages to apply strategy
            return self._sliding_window(messages, max_tokens)
        
        # Keep first N and last N messages
        first_n = self.config.preserve_recent
        last_n = self.config.preserve_recent
        
        first_messages = messages[:first_n]
        last_messages = messages[-last_n:]
        
        # Calculate tokens
        first_tokens = sum(self._estimate_tokens(m.content) for m in first_messages)
        last_tokens = sum(self._estimate_tokens(m.content) for m in last_messages)
        total_tokens = first_tokens + last_tokens
        
        if total_tokens <= max_tokens:
            # Add middle indicator
            middle_msg = Message(
                role="system",
                content=f"[{len(messages) - first_n - last_n} messages omitted from middle]",
                metadata={"is_middle_marker": True}
            )
            
            return first_messages + [middle_msg] + last_messages, True
        else:
            # Can't even fit ends, fallback to sliding
            return self._sliding_window(messages, max_tokens)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses rough approximation: ~4 characters per token.
        In production, use tiktoken or actual tokenizer.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough approximation
        return len(text) // 4 + 1


if __name__ == "__main__":
    # Example usage
    from src.context.memory import ConversationMemory
    
    # Create memory and buffer
    memory = ConversationMemory()
    config = BufferConfig(max_tokens=100, preserve_recent=2)
    buffer = ConversationBuffer(memory, config)
    
    # Create conversation
    conv_id = memory.create_conversation()
    
    # Add many messages to trigger truncation
    for i in range(10):
        memory.add_message(
            conv_id,
            "user" if i % 2 == 0 else "assistant",
            f"This is message number {i} with some content to use tokens."
        )
    
    # Get buffered context
    context = buffer.get_buffered_context(conv_id)
    
    print(f"Total messages: {len(memory.get_messages(conv_id))}")
    print(f"Buffered messages: {len(context['messages'])}")
    print(f"Tokens used: {context['tokens_used']}")
    print(f"Truncated: {context['truncated']}")
    print(f"Strategy: {context['strategy_used']}")
    
    # Get stats
    stats = buffer.get_token_stats(conv_id)
    print(f"\nToken stats: {stats}")
