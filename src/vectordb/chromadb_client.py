"""
ChromaDB vector database client.

ChromaDB is a lightweight, open-source vector database perfect for RAG.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from src.utils.logging_config import setup_logging
from src.embeddings.providers.embedding_providers import EmbeddingProvider

logger = setup_logging("rag.chromadb")


class ChromaDBClient:
    """
    ChromaDB client for vector storage and retrieval.
    
    Features:
        - Persistent storage
        - HNSW indexing for fast similarity search
        - Metadata filtering
        - Automatic embedding (if embedder provided)
    """
    
    def __init__(
        self,
        collection_name: str = "company_knowledge",
        persist_directory: str = "./chromadb_data",
        embedder: Optional[EmbeddingProvider] = None,
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedder: Optional embedding provider
            distance_metric: "cosine", "l2", or "ip" (inner product)
        """
        self.collection_name = collection_name
        self.embedder = embedder
        
        # Initialize client with persistence
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )
        
        logger.info(
            f"ChromaDB initialized: collection={collection_name}, "
            f"documents={self.collection.count()}"
        )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add documents to collection.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of document IDs
            embeddings: Optional pre-computed embeddings
        """
        if embeddings is None and self.embedder:
            logger.info(f"Embedding {len(documents)} documents...")
            embeddings = self.embedder.embed_batch(documents)
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results
            where: Metadata filter (e.g., {"category": "HR"})
            query_embedding: Optional pre-computed query embedding
            
        Returns:
            Search results with documents, distances, and metadata
        """
        if query_embedding is None and self.embedder:
            query_embedding = self.embedder.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding] if query_embedding else None,
            query_texts=[query] if query_embedding is None else None,
            n_results=top_k,
            where=where
        )
        
        return results
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
