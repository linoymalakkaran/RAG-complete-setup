"""
Document chunking strategies for RAG.

Implements multiple chunking approaches:
1. Fixed-size chunking: Simple character-based splitting
2. Recursive chunking: Split on hierarchical separators
3. Semantic chunking: Group sentences by semantic similarity
4. Parent-document chunking: Small chunks with parent context
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import re
from dataclasses import dataclass

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.utils.logging_config import setup_logging

logger = setup_logging("rag.chunking")


@dataclass
class Chunk:
    """
    Represents a document chunk with metadata.
    
    Attributes:
        content: The text content of the chunk
        metadata: Associated metadata (source, page, position, etc.)
        chunk_id: Unique identifier for this chunk
        embedding: Optional pre-computed embedding
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def __len__(self):
        return len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id,
            'embedding': self.embedding
        }


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    
    All chunking strategies should inherit from this class and implement
    the chunk() method.
    """
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def _add_metadata(self, chunks: List[str], base_metadata: Dict[str, Any]) -> List[Chunk]:
        """Helper to convert text chunks to Chunk objects with metadata"""
        result = []
        for idx, chunk_text in enumerate(chunks):
            metadata = {**base_metadata, 'chunk_index': idx}
            result.append(Chunk(
                content=chunk_text,
                metadata=metadata,
                chunk_id=f"{base_metadata.get('doc_id', 'unknown')}_{idx}"
            ))
        return result


class FixedSizeChunking(ChunkingStrategy):
    """
    Fixed-size chunking strategy.
    
    Splits text into chunks of fixed character length with overlap.
    
    Pros:
        - Simple and predictable
        - Consistent chunk sizes
        - Fast processing
    
    Cons:
        - May split sentences/paragraphs awkwardly
        - No semantic awareness
        
    Best for:
        - Uniform documents
        - When consistent chunk size is important
        - Quick prototyping
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Initialized FixedSizeChunking: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into fixed-size chunks"""
        metadata = metadata or {}
        
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        
        chunks = splitter.split_text(text)
        logger.debug(f"Created {len(chunks)} fixed-size chunks")
        
        return self._add_metadata(chunks, metadata)


class RecursiveChunking(ChunkingStrategy):
    """
    Recursive chunking strategy.
    
    Tries to split on hierarchical separators (paragraphs → sentences → words).
    This preserves document structure better than fixed-size chunking.
    
    Separators hierarchy:
        1. Double newline (paragraphs)
        2. Single newline
        3. Period + space (sentences)
        4. Space (words)
        5. Character-level (fallback)
    
    Pros:
        - Respects document structure
        - More natural chunk boundaries
        - Configurable separators
    
    Cons:
        - Variable chunk sizes
        - Slightly more complex
        
    Best for:
        - Most use cases (recommended default)
        - Documents with clear structure
        - When semantic coherence matters
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target characters per chunk
            chunk_overlap: Number of overlapping characters
            separators: List of separators to try (in order)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        logger.info(
            f"Initialized RecursiveChunking: size={chunk_size}, "
            f"overlap={chunk_overlap}, separators={len(self.separators)}"
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text using recursive strategy"""
        metadata = metadata or {}
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        logger.debug(f"Created {len(chunks)} recursive chunks")
        
        return self._add_metadata(chunks, metadata)


class SemanticChunking(ChunkingStrategy):
    """
    Semantic chunking strategy.
    
    Groups sentences based on semantic similarity rather than fixed size.
    Chunks are formed by grouping consecutive sentences that are semantically related.
    
    Algorithm:
        1. Split text into sentences
        2. Embed each sentence
        3. Calculate similarity between consecutive sentences
        4. Create chunk boundary when similarity drops below threshold
    
    Pros:
        - Semantically coherent chunks
        - Adaptive to content
        - Better for topic-focused retrieval
    
    Cons:
        - Slower (requires embedding)
        - Variable chunk sizes
        - Requires embedding model
        
    Best for:
        - Documents with topic shifts
        - When semantic coherence is critical
        - Educational/technical content
    """
    
    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1500,
        similarity_threshold: float = 0.5,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize semantic chunker.
        
        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            similarity_threshold: Similarity threshold for grouping (0-1)
            embedding_model: Sentence transformer model name
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        
        # Load embedding model for semantic similarity
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        logger.info(
            f"Initialized SemanticChunking: min={min_chunk_size}, "
            f"max={max_chunk_size}, threshold={similarity_threshold}"
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text using semantic similarity"""
        metadata = metadata or {}
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Embed sentences
        logger.debug(f"Embedding {len(sentences)} sentences")
        embeddings = self.model.encode(sentences)
        
        # Group semantically similar sentences
        chunks = self._group_sentences(sentences, embeddings)
        
        logger.debug(f"Created {len(chunks)} semantic chunks")
        
        return self._add_metadata(chunks, metadata)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[str]:
        """
        Group sentences into chunks based on semantic similarity.
        
        Creates a new chunk when:
            - Similarity between consecutive sentences drops below threshold
            - Max chunk size is reached
        """
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i-1:i],
                embeddings[i:i+1]
            )[0][0]
            
            sentence = sentences[i]
            sentence_len = len(sentence)
            
            # Decide whether to add to current chunk or start new one
            should_split = (
                similarity < self.similarity_threshold or
                current_size + sentence_len > self.max_chunk_size
            )
            
            if should_split and current_size >= self.min_chunk_size:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_len
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_size += sentence_len
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class ParentDocumentChunking(ChunkingStrategy):
    """
    Parent-document chunking strategy.
    
    Creates small chunks for retrieval but maintains reference to larger
    parent chunks for context. This solves the "context window" problem:
        - Small chunks: Better for precise retrieval
        - Large parents: Better for LLM context
    
    Structure:
        Parent (2000 chars)
        ├── Child 1 (400 chars)
        ├── Child 2 (400 chars)
        └── Child 3 (400 chars)
    
    At retrieval time:
        1. Search over child chunks (precise matching)
        2. Return parent chunks (rich context)
    
    Pros:
        - Best of both worlds (precision + context)
        - Better for complex queries
        - Preserves surrounding context
    
    Cons:
        - More storage (children + parents)
        - More complex retrieval logic
        - Higher token usage
        
    Best for:
        - Long documents
        - When context is crucial
        - Complex question answering
    """
    
    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 400,
        child_overlap: int = 50
    ):
        """
        Initialize parent-document chunker.
        
        Args:
            parent_size: Size of parent chunks
            child_size: Size of child chunks
            child_overlap: Overlap between child chunks
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        
        logger.info(
            f"Initialized ParentDocumentChunking: parent={parent_size}, "
            f"child={child_size}, overlap={child_overlap}"
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Create parent-child chunk hierarchy.
        
        Returns child chunks with parent_id in metadata for linking.
        """
        metadata = metadata or {}
        
        # Create parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        parent_texts = parent_splitter.split_text(text)
        
        # Create child chunks for each parent
        child_chunks = []
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = f"{metadata.get('doc_id', 'unknown')}_parent_{parent_idx}"
            
            # Split parent into children
            child_texts = child_splitter.split_text(parent_text)
            
            for child_idx, child_text in enumerate(child_texts):
                child_metadata = {
                    **metadata,
                    'parent_id': parent_id,
                    'parent_content': parent_text,  # Store parent for retrieval
                    'chunk_index': f"{parent_idx}_{child_idx}"
                }
                
                child_chunks.append(Chunk(
                    content=child_text,
                    metadata=child_metadata,
                    chunk_id=f"{parent_id}_child_{child_idx}"
                ))
        
        logger.debug(
            f"Created {len(child_chunks)} child chunks from "
            f"{len(parent_texts)} parent chunks"
        )
        
        return child_chunks


class ChunkerFactory:
    """
    Factory to create chunking strategies based on configuration.
    
    Usage:
        >>> chunker = ChunkerFactory.create("recursive", chunk_size=1000)
        >>> chunks = chunker.chunk(text, metadata)
    """
    
    _strategies = {
        'fixed': FixedSizeChunking,
        'recursive': RecursiveChunking,
        'semantic': SemanticChunking,
        'parent_document': ParentDocumentChunking
    }
    
    @classmethod
    def create(cls, strategy: str, **kwargs) -> ChunkingStrategy:
        """
        Create chunking strategy instance.
        
        Args:
            strategy: Strategy name (fixed, recursive, semantic, parent_document)
            **kwargs: Strategy-specific parameters
            
        Returns:
            ChunkingStrategy instance
        """
        if strategy not in cls._strategies:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        
        strategy_class = cls._strategies[strategy]
        return strategy_class(**kwargs)
    
    @classmethod
    def available_strategies(cls) -> List[str]:
        """Get list of available strategies"""
        return list(cls._strategies.keys())


def chunk_document(
    text: str,
    strategy: str = "recursive",
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[Chunk]:
    """
    Convenience function to chunk text with specified strategy.
    
    Args:
        text: Input text
        strategy: Chunking strategy name
        metadata: Metadata to attach to chunks
        **kwargs: Strategy-specific parameters
        
    Returns:
        List of Chunk objects
        
    Example:
        >>> chunks = chunk_document(
        ...     text=document_content,
        ...     strategy="recursive",
        ...     chunk_size=1000,
        ...     metadata={"doc_id": "policy_001", "source": "hr_manual.pdf"}
        ... )
    """
    chunker = ChunkerFactory.create(strategy, **kwargs)
    return chunker.chunk(text, metadata)
