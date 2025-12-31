"""
Conversation Memory - Multi-turn conversation support

Manages conversation history for RAG systems with:
- Message history tracking
- Context summarization
- Memory persistence
- Conversation threading
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

from src.utils.logging_config import RAGLogger


@dataclass
class Message:
    """Represents a single message in a conversation."""
    
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """Represents a conversation thread."""
    
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get conversation messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    Manages conversation history and context.
    
    Provides multi-turn conversation support with:
    - Message history tracking
    - Context window management
    - Conversation summarization
    - Memory persistence
    
    Example:
        >>> memory = ConversationMemory(max_messages=10)
        >>> 
        >>> # Start new conversation
        >>> conv_id = memory.create_conversation()
        >>> 
        >>> # Add messages
        >>> memory.add_message(conv_id, "user", "What is RAG?")
        >>> memory.add_message(conv_id, "assistant", "RAG stands for...")
        >>> 
        >>> # Get context
        >>> context = memory.get_conversation_context(conv_id)
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 4000,
        summarize_threshold: int = 15,
        persist_path: Optional[str] = None
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to keep in memory
            max_tokens: Maximum tokens for context window
            summarize_threshold: Summarize when messages exceed this
            persist_path: Path to persist conversations (optional)
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self.persist_path = persist_path
        
        self.logger = RAGLogger.get_logger("conversation_memory")
        
        # In-memory storage
        self.conversations: Dict[str, Conversation] = {}
        
        # Load persisted conversations if path provided
        if persist_path:
            self._load_conversations()
        
        self.logger.info(
            f"Initialized conversation memory (max_messages={max_messages}, "
            f"max_tokens={max_tokens})"
        )
    
    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional ID (generated if not provided)
            metadata: Optional metadata
            
        Returns:
            Conversation ID
        """
        if conversation_id is None:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        conversation = Conversation(
            conversation_id=conversation_id,
            metadata=metadata or {}
        )
        
        self.conversations[conversation_id] = conversation
        
        self.logger.info(f"Created conversation: {conversation_id}")
        
        if self.persist_path:
            self._persist_conversation(conversation_id)
        
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        conversation.add_message(role, content, metadata)
        
        self.logger.debug(
            f"Added {role} message to {conversation_id} "
            f"({len(conversation.messages)} total)"
        )
        
        # Check if summarization needed
        if len(conversation.messages) > self.summarize_threshold:
            self._maybe_summarize(conversation_id)
        
        # Trim if exceeds max
        if len(conversation.messages) > self.max_messages:
            self._trim_messages(conversation_id)
        
        if self.persist_path:
            self._persist_conversation(conversation_id)
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None
    ) -> str:
        """
        Get conversation context as formatted string.
        
        Args:
            conversation_id: Conversation ID
            max_messages: Max recent messages to include
            
        Returns:
            Formatted conversation context
        """
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return ""
        
        messages = conversation.get_messages(limit=max_messages)
        
        context_parts = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get messages from a conversation."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return []
        
        return conversation.get_messages(limit=limit)
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.logger.info(f"Deleted conversation: {conversation_id}")
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self.conversations.keys())
    
    def _maybe_summarize(self, conversation_id: str):
        """Summarize old messages if conversation is long."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return
        
        # Check if summary already exists
        if "summary" in conversation.metadata:
            return
        
        # Get older messages (exclude last 5)
        old_messages = conversation.messages[:-5]
        
        if len(old_messages) >= self.summarize_threshold:
            # Create summary (in production, use LLM)
            summary = f"[Summary of {len(old_messages)} earlier messages]"
            conversation.metadata["summary"] = summary
            
            # Keep only recent messages + summary
            conversation.messages = conversation.messages[-5:]
            
            self.logger.info(f"Summarized {conversation_id}")
    
    def _trim_messages(self, conversation_id: str):
        """Trim messages to max_messages."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return
        
        if len(conversation.messages) > self.max_messages:
            # Keep only most recent messages
            conversation.messages = conversation.messages[-self.max_messages:]
            self.logger.info(f"Trimmed {conversation_id} to {self.max_messages} messages")
    
    def _persist_conversation(self, conversation_id: str):
        """Persist conversation to disk."""
        if not self.persist_path:
            return
        
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return
        
        try:
            import os
            os.makedirs(self.persist_path, exist_ok=True)
            
            filepath = os.path.join(self.persist_path, f"{conversation_id}.json")
            with open(filepath, 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
            
            self.logger.debug(f"Persisted {conversation_id}")
        
        except Exception as e:
            self.logger.error(f"Error persisting conversation: {e}")
    
    def _load_conversations(self):
        """Load persisted conversations from disk."""
        if not self.persist_path:
            return
        
        try:
            import os
            if not os.path.exists(self.persist_path):
                return
            
            for filename in os.listdir(self.persist_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.persist_path, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        conversation = Conversation.from_dict(data)
                        self.conversations[conversation.conversation_id] = conversation
            
            self.logger.info(f"Loaded {len(self.conversations)} conversations")
        
        except Exception as e:
            self.logger.error(f"Error loading conversations: {e}")


if __name__ == "__main__":
    # Example usage
    memory = ConversationMemory(max_messages=10)
    
    # Create conversation
    conv_id = memory.create_conversation()
    
    # Add messages
    memory.add_message(conv_id, "user", "What is RAG?")
    memory.add_message(conv_id, "assistant", "RAG stands for Retrieval Augmented Generation...")
    memory.add_message(conv_id, "user", "How does it work?")
    memory.add_message(conv_id, "assistant", "RAG works by first retrieving relevant documents...")
    
    # Get context
    context = memory.get_conversation_context(conv_id)
    print("Conversation Context:")
    print(context)
    
    # Get messages
    messages = memory.get_messages(conv_id, limit=2)
    print(f"\nLast 2 messages: {len(messages)}")
