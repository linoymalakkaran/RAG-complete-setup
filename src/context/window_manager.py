"""
Context Window Manager - Advanced context management

Provides sophisticated context window management:
- Dynamic window sizing
- Priority-based message selection
- Context compression
- Multi-document context balancing
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from src.context.memory import Message
from src.utils.logging_config import RAGLogger


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 4  # System prompts, user queries
    HIGH = 3  # Recent messages
    MEDIUM = 2  # Relevant history
    LOW = 1  # Old context


@dataclass
class ContextItem:
    """Represents an item in the context window."""
    
    content: str
    priority: MessagePriority
    tokens: int
    metadata: Dict[str, Any]
    source: str  # 'conversation', 'document', 'system'


class WindowManager:
    """
    Advanced context window management.
    
    Manages context window with sophisticated strategies:
    - Priority-based selection
    - Dynamic window sizing
    - Lost-in-middle mitigation
    - Multi-source context balancing
    
    Example:
        >>> manager = WindowManager(max_tokens=4000)
        >>> 
        >>> # Add conversation messages
        >>> manager.add_conversation_context(messages)
        >>> 
        >>> # Add document context
        >>> manager.add_document_context(retrieved_docs)
        >>> 
        >>> # Build final context
        >>> context = manager.build_context()
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        reserve_tokens: int = 1000,
        conversation_ratio: float = 0.3,
        document_ratio: float = 0.6,
        system_ratio: float = 0.1,
        priority_boost: float = 1.5
    ):
        """
        Initialize window manager.
        
        Args:
            max_tokens: Maximum total tokens
            reserve_tokens: Tokens reserved for response
            conversation_ratio: Ratio of tokens for conversation
            document_ratio: Ratio of tokens for documents
            system_ratio: Ratio of tokens for system prompts
            priority_boost: Multiplier for high-priority items
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.conversation_ratio = conversation_ratio
        self.document_ratio = document_ratio
        self.system_ratio = system_ratio
        self.priority_boost = priority_boost
        
        self.logger = RAGLogger.get_logger("window_manager")
        
        # Context items
        self.items: List[ContextItem] = []
        
        # Calculate token budgets
        available = max_tokens - reserve_tokens
        self.conversation_budget = int(available * conversation_ratio)
        self.document_budget = int(available * document_ratio)
        self.system_budget = int(available * system_ratio)
        
        self.logger.info(
            f"Initialized window manager: conversation={self.conversation_budget}, "
            f"documents={self.document_budget}, system={self.system_budget} tokens"
        )
    
    def add_conversation_context(
        self,
        messages: List[Message],
        prioritize_recent: bool = True
    ):
        """
        Add conversation messages to context.
        
        Args:
            messages: List of conversation messages
            prioritize_recent: Give higher priority to recent messages
        """
        for i, message in enumerate(messages):
            # Determine priority
            if prioritize_recent:
                # More recent = higher priority
                recency_factor = i / len(messages)
                if recency_factor > 0.8:
                    priority = MessagePriority.HIGH
                elif recency_factor > 0.5:
                    priority = MessagePriority.MEDIUM
                else:
                    priority = MessagePriority.LOW
            else:
                priority = MessagePriority.MEDIUM
            
            # Estimate tokens
            tokens = self._estimate_tokens(message.content)
            
            item = ContextItem(
                content=message.content,
                priority=priority,
                tokens=tokens,
                metadata={"role": message.role, "index": i},
                source="conversation"
            )
            
            self.items.append(item)
        
        self.logger.debug(f"Added {len(messages)} conversation messages")
    
    def add_document_context(
        self,
        documents: List[Dict[str, Any]],
        relevance_scores: Optional[List[float]] = None
    ):
        """
        Add retrieved documents to context.
        
        Args:
            documents: Retrieved documents
            relevance_scores: Optional relevance scores for prioritization
        """
        for i, doc in enumerate(documents):
            content = doc.get("content", doc.get("page_content", ""))
            
            # Determine priority based on relevance
            if relevance_scores and i < len(relevance_scores):
                score = relevance_scores[i]
                if score > 0.8:
                    priority = MessagePriority.CRITICAL
                elif score > 0.6:
                    priority = MessagePriority.HIGH
                elif score > 0.4:
                    priority = MessagePriority.MEDIUM
                else:
                    priority = MessagePriority.LOW
            else:
                # First documents are usually most relevant
                if i < 2:
                    priority = MessagePriority.HIGH
                elif i < 5:
                    priority = MessagePriority.MEDIUM
                else:
                    priority = MessagePriority.LOW
            
            tokens = self._estimate_tokens(content)
            
            item = ContextItem(
                content=content,
                priority=priority,
                tokens=tokens,
                metadata=doc.get("metadata", {}),
                source="document"
            )
            
            self.items.append(item)
        
        self.logger.debug(f"Added {len(documents)} documents")
    
    def add_system_prompt(self, prompt: str):
        """Add system prompt (highest priority)."""
        item = ContextItem(
            content=prompt,
            priority=MessagePriority.CRITICAL,
            tokens=self._estimate_tokens(prompt),
            metadata={},
            source="system"
        )
        
        self.items.append(item)
    
    def build_context(
        self,
        strategy: str = "priority"
    ) -> Dict[str, Any]:
        """
        Build final context within token limits.
        
        Args:
            strategy: Selection strategy ('priority', 'balanced', 'lost_in_middle')
            
        Returns:
            Dict with:
                - context: Final context string
                - items_included: List of included items
                - tokens_used: Total tokens
                - truncated: Whether truncation occurred
        """
        if strategy == "priority":
            selected = self._priority_selection()
        elif strategy == "balanced":
            selected = self._balanced_selection()
        elif strategy == "lost_in_middle":
            selected = self._lost_in_middle_selection()
        else:
            selected = self.items
        
        # Build context string
        context_parts = []
        
        # System prompts first
        for item in selected:
            if item.source == "system":
                context_parts.append(item.content)
        
        # Then documents
        for i, item in enumerate(selected):
            if item.source == "document":
                context_parts.append(f"Document {i+1}:\n{item.content}")
        
        # Then conversation
        for item in selected:
            if item.source == "conversation":
                role = item.metadata.get("role", "unknown")
                context_parts.append(f"{role.title()}: {item.content}")
        
        context = "\n\n".join(context_parts)
        total_tokens = sum(item.tokens for item in selected)
        
        return {
            "context": context,
            "items_included": selected,
            "tokens_used": total_tokens,
            "truncated": len(selected) < len(self.items),
            "items_total": len(self.items),
            "items_selected": len(selected)
        }
    
    def _priority_selection(self) -> List[ContextItem]:
        """Select items based on priority."""
        # Sort by priority (descending) and tokens (ascending for same priority)
        sorted_items = sorted(
            self.items,
            key=lambda x: (x.priority.value, -x.tokens),
            reverse=True
        )
        
        selected = []
        tokens_used = 0
        
        for item in sorted_items:
            if tokens_used + item.tokens <= (self.max_tokens - self.reserve_tokens):
                selected.append(item)
                tokens_used += item.tokens
            elif item.priority == MessagePriority.CRITICAL:
                # Always include critical items
                selected.append(item)
                tokens_used += item.tokens
        
        return selected
    
    def _balanced_selection(self) -> List[ContextItem]:
        """Balance tokens across different sources."""
        selected = []
        
        # Separate by source
        by_source = {
            "system": [],
            "conversation": [],
            "document": []
        }
        
        for item in self.items:
            by_source[item.source].append(item)
        
        # Select from each source within budget
        # System (always include)
        for item in by_source["system"]:
            selected.append(item)
        
        # Conversation
        conv_selected = self._select_within_budget(
            by_source["conversation"],
            self.conversation_budget
        )
        selected.extend(conv_selected)
        
        # Documents
        doc_selected = self._select_within_budget(
            by_source["document"],
            self.document_budget
        )
        selected.extend(doc_selected)
        
        return selected
    
    def _lost_in_middle_selection(self) -> List[ContextItem]:
        """
        Mitigate lost-in-middle problem.
        
        Keep high-priority items at start and end of context.
        """
        # Sort by priority
        sorted_items = sorted(
            self.items,
            key=lambda x: x.priority.value,
            reverse=True
        )
        
        # Calculate how many we can fit
        available = self.max_tokens - self.reserve_tokens
        
        selected_start = []
        selected_end = []
        tokens_used = 0
        
        # Add critical items first
        critical = [i for i in sorted_items if i.priority == MessagePriority.CRITICAL]
        for item in critical:
            selected_start.append(item)
            tokens_used += item.tokens
        
        # Add remaining items, alternating between start and end
        remaining = [i for i in sorted_items if i.priority != MessagePriority.CRITICAL]
        add_to_start = True
        
        for item in remaining:
            if tokens_used + item.tokens > available:
                break
            
            if add_to_start:
                selected_start.append(item)
            else:
                selected_end.insert(0, item)
            
            tokens_used += item.tokens
            add_to_start = not add_to_start
        
        return selected_start + selected_end
    
    def _select_within_budget(
        self,
        items: List[ContextItem],
        budget: int
    ) -> List[ContextItem]:
        """Select items within token budget."""
        # Sort by priority
        sorted_items = sorted(
            items,
            key=lambda x: x.priority.value,
            reverse=True
        )
        
        selected = []
        tokens_used = 0
        
        for item in sorted_items:
            if tokens_used + item.tokens <= budget:
                selected.append(item)
                tokens_used += item.tokens
        
        return selected
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars ≈ 1 token)."""
        return len(text) // 4 + 1
    
    def clear(self):
        """Clear all context items."""
        self.items = []
        self.logger.debug("Cleared context")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        by_source = {"system": 0, "conversation": 0, "document": 0}
        by_priority = {p: 0 for p in MessagePriority}
        total_tokens = 0
        
        for item in self.items:
            by_source[item.source] += 1
            by_priority[item.priority] += 1
            total_tokens += item.tokens
        
        return {
            "total_items": len(self.items),
            "total_tokens": total_tokens,
            "by_source": by_source,
            "by_priority": {p.name: c for p, c in by_priority.items()},
            "budgets": {
                "conversation": self.conversation_budget,
                "document": self.document_budget,
                "system": self.system_budget,
                "reserve": self.reserve_tokens
            }
        }


if __name__ == "__main__":
    # Example usage
    manager = WindowManager(max_tokens=200)
    
    # Add system prompt
    manager.add_system_prompt("You are a helpful assistant.")
    
    # Add conversation
    from src.context.memory import Message
    messages = [
        Message("user", "What is RAG?"),
        Message("assistant", "RAG stands for Retrieval Augmented Generation."),
        Message("user", "How does it work?")
    ]
    manager.add_conversation_context(messages)
    
    # Add documents
    docs = [
        {"content": "RAG combines retrieval and generation...", "metadata": {}},
        {"content": "The retrieval component finds relevant docs...", "metadata": {}}
    ]
    manager.add_document_context(docs, relevance_scores=[0.9, 0.7])
    
    # Build context
    result = manager.build_context(strategy="priority")
    
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Items included: {result['items_selected']}/{result['items_total']}")
    print(f"Truncated: {result['truncated']}")
    print(f"\nContext preview:\n{result['context'][:200]}...")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nStats: {stats}")
