"""
RAG package initialization
"""

__version__ = "1.0.0"
__author__ = "RAG Learning Project"

from src.ingestion.loaders.document_loaders import load_document
from src.ingestion.chunking.chunking_strategies import chunk_document
from src.embeddings.providers.embedding_providers import EmbeddingFactory
from src.vectordb.chromadb_client import ChromaDBClient
from src.rag_patterns.basic_rag import create_basic_rag

__all__ = [
    'load_document',
    'chunk_document',
    'EmbeddingFactory',
    'ChromaDBClient',
    'create_basic_rag'
]
