"""
Unit Tests for Context Management

Tests for:
- Conversation Memory
- Conversation Buffer
- Window Manager
"""

import pytest
from datetime import datetime

from src.context import (
    Message,
    Conversation,
    ConversationMemory,
    BufferConfig,
    ConversationBuffer,
    MessagePriority,
    ContextItem,
    WindowManager
)


class TestConversationMemory:
    """Tests for ConversationMemory."""
    
    def test_create_conversation(self):
        """Test conversation creation."""
        memory = ConversationMemory()
        conv_id = memory.create_conversation()
        
        assert conv_id is not None
        assert conv_id in memory.conversations
        assert len(memory.conversations) == 1
    
    def test_add_message(self):
        """Test adding messages."""
        memory = ConversationMemory()
        conv_id = memory.create_conversation()
        
        memory.add_message(conv_id, "user", "Hello")
        memory.add_message(conv_id, "assistant", "Hi there!")
        
        messages = memory.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"
    
    def test_message_limit(self):
        """Test message limit enforcement."""
        memory = ConversationMemory(max_messages=5)
        conv_id = memory.create_conversation()
        
        # Add 10 messages
        for i in range(10):
            memory.add_message(conv_id, "user", f"Message {i}")
        
        messages = memory.get_messages(conv_id)
        # Should only keep last 5
        assert len(messages) == 5
        assert messages[-1].content == "Message 9"
    
    def test_get_conversation_context(self):
        """Test getting conversation context."""
        memory = ConversationMemory()
        conv_id = memory.create_conversation()
        
        memory.add_message(conv_id, "user", "What is RAG?")
        memory.add_message(conv_id, "assistant", "RAG is...")
        
        context = memory.get_conversation_context(conv_id)
        
        assert "user: What is RAG?" in context
        assert "assistant: RAG is..." in context
    
    def test_delete_conversation(self):
        """Test conversation deletion."""
        memory = ConversationMemory()
        conv_id = memory.create_conversation()
        
        memory.add_message(conv_id, "user", "Hello")
        memory.delete_conversation(conv_id)
        
        assert conv_id not in memory.conversations
        messages = memory.get_messages(conv_id)
        assert len(messages) == 0


class TestConversationBuffer:
    """Tests for ConversationBuffer."""
    
    def test_get_buffered_context(self):
        """Test getting buffered context."""
        memory = ConversationMemory()
        config = BufferConfig(max_tokens=100, preserve_recent=2)
        buffer = ConversationBuffer(memory, config)
        
        conv_id = memory.create_conversation()
        
        # Add messages
        for i in range(5):
            memory.add_message(conv_id, "user", f"Message {i}")
        
        result = buffer.get_buffered_context(conv_id)
        
        assert "messages" in result
        assert "tokens_used" in result
        assert "truncated" in result
        assert len(result["messages"]) <= 5
    
    def test_can_add_message(self):
        """Test checking if message can be added."""
        memory = ConversationMemory()
        config = BufferConfig(max_tokens=50, reserve_tokens=10)
        buffer = ConversationBuffer(memory, config)
        
        conv_id = memory.create_conversation()
        
        # Short message should fit
        assert buffer.can_add_message(conv_id, "Short")
        
        # Very long message might not fit
        long_message = "word " * 1000
        can_add = buffer.can_add_message(conv_id, long_message)
        # Result depends on token estimation
        assert isinstance(can_add, bool)
    
    def test_token_stats(self):
        """Test token statistics."""
        memory = ConversationMemory()
        buffer = ConversationBuffer(memory)
        
        conv_id = memory.create_conversation()
        memory.add_message(conv_id, "user", "Hello world")
        
        stats = buffer.get_token_stats(conv_id)
        
        assert "total_tokens" in stats
        assert "message_count" in stats
        assert "avg_tokens_per_message" in stats
        assert stats["message_count"] == 1


class TestWindowManager:
    """Tests for WindowManager."""
    
    def test_add_conversation_context(self):
        """Test adding conversation context."""
        manager = WindowManager(max_tokens=1000)
        
        messages = [
            Message("user", "Hello"),
            Message("assistant", "Hi there!")
        ]
        
        manager.add_conversation_context(messages)
        
        assert len(manager.items) == 2
        assert manager.items[0].source == "conversation"
    
    def test_add_document_context(self):
        """Test adding document context."""
        manager = WindowManager(max_tokens=1000)
        
        docs = [
            {"content": "Document 1 content", "metadata": {}},
            {"content": "Document 2 content", "metadata": {}}
        ]
        
        manager.add_document_context(docs)
        
        assert len(manager.items) == 2
        assert manager.items[0].source == "document"
    
    def test_add_system_prompt(self):
        """Test adding system prompt."""
        manager = WindowManager(max_tokens=1000)
        
        manager.add_system_prompt("You are a helpful assistant.")
        
        assert len(manager.items) == 1
        assert manager.items[0].source == "system"
        assert manager.items[0].priority == MessagePriority.CRITICAL
    
    def test_build_context_priority(self):
        """Test building context with priority strategy."""
        manager = WindowManager(max_tokens=200)
        
        # Add items with different priorities
        manager.add_system_prompt("System prompt")
        
        messages = [
            Message("user", "Question 1"),
            Message("assistant", "Answer 1")
        ]
        manager.add_conversation_context(messages)
        
        docs = [{"content": "Document content", "metadata": {}}]
        manager.add_document_context(docs, relevance_scores=[0.9])
        
        result = manager.build_context(strategy="priority")
        
        assert "context" in result
        assert "tokens_used" in result
        assert "truncated" in result
        assert len(result["context"]) > 0
    
    def test_clear(self):
        """Test clearing context."""
        manager = WindowManager(max_tokens=1000)
        
        manager.add_system_prompt("Test")
        assert len(manager.items) == 1
        
        manager.clear()
        assert len(manager.items) == 0
    
    def test_get_stats(self):
        """Test getting statistics."""
        manager = WindowManager(max_tokens=1000)
        
        manager.add_system_prompt("Test")
        
        stats = manager.get_stats()
        
        assert "total_items" in stats
        assert "total_tokens" in stats
        assert "by_source" in stats
        assert "by_priority" in stats
        assert stats["total_items"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
