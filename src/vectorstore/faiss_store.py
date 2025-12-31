"""
FAISS Vector Store Implementation

Alternative to ChromaDB using FAISS (Facebook AI Similarity Search).

Features:
- Fast similarity search with FAISS indices
- Multiple index types (Flat, IVF, HNSW)
- Persistence to disk
- Metadata filtering
- Batch operations
- GPU support (if available)
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")


@dataclass
class FAISSConfig:
    """Configuration for FAISS vector store."""
    index_type: str = "Flat"  # Flat, IVF, HNSW
    dimension: int = 768  # Embedding dimension
    metric: str = "L2"  # L2, IP (inner product)
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search in IVF
    m: int = 32  # Number of connections for HNSW
    ef_construction: int = 40  # Construction parameter for HNSW
    ef_search: int = 16  # Search parameter for HNSW
    use_gpu: bool = False  # Use GPU if available
    normalize_embeddings: bool = True  # Normalize for cosine similarity


@dataclass
class Document:
    """Document with embedding and metadata."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            # Embedding saved separately
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> 'Document':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            embedding=embedding,
            metadata=data.get("metadata", {})
        )


class FAISSVectorStore:
    """
    FAISS-based vector store for similarity search.
    
    Supports:
    - Multiple index types
    - Metadata filtering
    - Persistence
    - Batch operations
    """
    
    def __init__(
        self,
        config: Optional[FAISSConfig] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            config: FAISS configuration
            persist_directory: Directory to persist index and metadata
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        self.config = config or FAISSConfig()
        self.persist_directory = persist_directory
        
        # Initialize index
        self.index = self._create_index()
        
        # Storage for documents and metadata
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        # Load existing data if persist directory exists
        if persist_directory and os.path.exists(persist_directory):
            self.load()
        
        logger.info(f"Initialized FAISS vector store with {self.config.index_type} index")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        dimension = self.config.dimension
        
        if self.config.index_type == "Flat":
            # Exact search (brute force)
            if self.config.metric == "IP":
                index = faiss.IndexFlatIP(dimension)
            else:
                index = faiss.IndexFlatL2(dimension)
        
        elif self.config.index_type == "IVF":
            # Inverted file index (faster for large datasets)
            if self.config.metric == "IP":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(
                    quantizer,
                    dimension,
                    self.config.nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(
                    quantizer,
                    dimension,
                    self.config.nlist,
                    faiss.METRIC_L2
                )
        
        elif self.config.index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast)
            index = faiss.IndexHNSWFlat(dimension, self.config.m)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
        
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        # Move to GPU if requested and available
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"Could not move to GPU: {e}")
        
        return index
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
        return embedding
    
    def add(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text content
            embeddings: List of embeddings
            metadatas: List of metadata dicts
            ids: List of document IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Normalize embeddings
        embeddings_array = np.array([
            self._normalize_embedding(emb) for emb in embeddings
        ]).astype('float32')
        
        # Train IVF index if needed
        if self.config.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings_array)
            logger.info("IVF index trained")
        
        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
        # Store documents and mappings
        for i, (doc_id, text, embedding, metadata) in enumerate(
            zip(ids, texts, embeddings, metadatas)
        ):
            idx = start_idx + i
            
            # Create document
            doc = Document(
                id=doc_id,
                text=text,
                embedding=embedding,
                metadata=metadata
            )
            
            # Update storage
            self.documents[doc_id] = doc
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
        
        logger.info(f"Added {len(texts)} documents. Total: {len(self.documents)}")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_func: Optional[Callable[[Document], bool]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_func: Optional function to filter results by metadata
            
        Returns:
            List of (Document, score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query_emb = self._normalize_embedding(query_embedding).astype('float32')
        query_emb = query_emb.reshape(1, -1)
        
        # Set search parameters for IVF
        if self.config.index_type == "IVF":
            self.index.nprobe = self.config.nprobe
        
        # Search (retrieve more if filtering)
        search_k = k * 10 if filter_func else k
        distances, indices = self.index.search(query_emb, search_k)
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            doc_id = self.index_to_id.get(idx)
            if doc_id is None:
                continue
            
            doc = self.documents.get(doc_id)
            if doc is None:
                continue
            
            # Apply filter if provided
            if filter_func and not filter_func(doc):
                continue
            
            # Convert distance to similarity score
            if self.config.metric == "IP":
                score = float(dist)  # Higher is better for IP
            else:
                score = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity
            
            results.append((doc, score))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search documents by metadata.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs
            k: Maximum number of results (None for all)
            
        Returns:
            List of matching documents
        """
        results = []
        
        for doc in self.documents.values():
            match = all(
                doc.metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
            
            if match:
                results.append(doc)
                if k and len(results) >= k:
                    break
        
        return results
    
    def get(self, ids: List[str]) -> List[Optional[Document]]:
        """Get documents by IDs."""
        return [self.documents.get(doc_id) for doc_id in ids]
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Note: FAISS doesn't support deletion, so we remove from metadata
        but not from the index. Consider rebuilding index periodically.
        """
        deleted = []
        for doc_id in ids:
            if doc_id in self.documents:
                doc = self.documents.pop(doc_id)
                idx = self.id_to_index.pop(doc_id, None)
                if idx is not None:
                    self.index_to_id.pop(idx, None)
                deleted.append(doc_id)
        
        if deleted:
            logger.info(f"Deleted {len(deleted)} documents (metadata only)")
            logger.warning("FAISS index not rebuilt. Consider calling rebuild_index()")
        
        return len(deleted) > 0
    
    def rebuild_index(self):
        """
        Rebuild FAISS index from current documents.
        
        Use this after deletions to actually remove vectors from index.
        """
        logger.info("Rebuilding FAISS index...")
        
        # Create new index
        self.index = self._create_index()
        
        # Reset mappings
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        # Re-add all documents
        if self.documents:
            texts = []
            embeddings = []
            metadatas = []
            ids = []
            
            for doc in self.documents.values():
                texts.append(doc.text)
                embeddings.append(doc.embedding)
                metadatas.append(doc.metadata)
                ids.append(doc.id)
            
            self.add(texts, embeddings, metadatas, ids)
        
        logger.info("Index rebuild complete")
    
    def save(self, directory: Optional[str] = None):
        """
        Save index and metadata to disk.
        
        Args:
            directory: Directory to save to (uses persist_directory if not provided)
        """
        save_dir = directory or self.persist_directory
        if not save_dir:
            raise ValueError("No save directory specified")
        
        # Create directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss_index.bin")
        
        # Move index to CPU if on GPU
        index_to_save = self.index
        if self.config.use_gpu:
            try:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            except Exception:
                pass
        
        faiss.write_index(index_to_save, index_path)
        
        # Save documents and metadata
        metadata_path = os.path.join(save_dir, "metadata.pkl")
        metadata = {
            "config": self.config,
            "documents": {
                doc_id: doc.to_dict()
                for doc_id, doc in self.documents.items()
            },
            "embeddings": {
                doc_id: doc.embedding
                for doc_id, doc in self.documents.items()
            },
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved FAISS store to {save_dir}")
    
    def load(self, directory: Optional[str] = None):
        """
        Load index and metadata from disk.
        
        Args:
            directory: Directory to load from (uses persist_directory if not provided)
        """
        load_dir = directory or self.persist_directory
        if not load_dir or not os.path.exists(load_dir):
            logger.warning(f"Load directory not found: {load_dir}")
            return
        
        index_path = os.path.join(load_dir, "faiss_index.bin")
        metadata_path = os.path.join(load_dir, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning("Index or metadata file not found")
            return
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Loaded index to GPU")
            except Exception as e:
                logger.warning(f"Could not move to GPU: {e}")
        
        self.index = index
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Restore config (update with saved config)
        saved_config = metadata.get("config")
        if saved_config:
            self.config = saved_config
        
        # Restore documents
        embeddings = metadata.get("embeddings", {})
        self.documents = {
            doc_id: Document.from_dict(doc_data, embeddings[doc_id])
            for doc_id, doc_data in metadata.get("documents", {}).items()
        }
        
        # Restore mappings
        self.id_to_index = metadata.get("id_to_index", {})
        self.index_to_id = {
            int(k): v for k, v in metadata.get("index_to_id", {}).items()
        }
        
        logger.info(f"Loaded FAISS store from {load_dir}")
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "num_documents": len(self.documents),
            "num_vectors": self.index.ntotal,
            "index_type": self.config.index_type,
            "dimension": self.config.dimension,
            "metric": self.config.metric,
            "is_trained": getattr(self.index, 'is_trained', True),
            "use_gpu": self.config.use_gpu
        }


class FAISSRetriever:
    """
    Retriever interface compatible with RAG system.
    
    Wraps FAISSVectorStore with embedding function.
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_function: Callable[[str], np.ndarray],
        k: int = 5
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: FAISS vector store
            embedding_function: Function to generate embeddings from text
            k: Default number of results to retrieve
        """
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.k = k
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Query text
            k: Number of results (uses default if not provided)
            metadata_filter: Filter by metadata
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        k = k or self.k
        
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Define filter function if metadata_filter provided
        filter_func = None
        if metadata_filter:
            def filter_func(doc: Document) -> bool:
                return all(
                    doc.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )
        
        # Search
        results = self.vector_store.search(query_embedding, k, filter_func)
        
        # Format results
        return [
            {
                "text": doc.text,
                "metadata": doc.metadata,
                "score": score,
                "id": doc.id
            }
            for doc, score in results
        ]
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text content
            metadatas: List of metadata dicts
            
        Returns:
            List of document IDs
        """
        # Generate embeddings
        embeddings = [self.embedding_function(text) for text in texts]
        
        # Add to store
        return self.vector_store.add(texts, embeddings, metadatas)
