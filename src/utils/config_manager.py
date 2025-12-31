"""
Utility module for configuration management.
Loads and validates configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models"""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100


class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    @validator('chunk_overlap')
    def overlap_smaller_than_size(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        return v


class VectorDBConfig(BaseModel):
    """Configuration for vector database"""
    provider: str = "chromadb"
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "company_knowledge"
    distance_metric: str = "cosine"


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline"""
    pattern: str = "basic"
    top_k: int = 5
    score_threshold: float = 0.7
    enable_reranking: bool = True


class AppConfig(BaseModel):
    """Main application configuration"""
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vectordb: VectorDBConfig = VectorDBConfig()
    rag: RAGConfig = RAGConfig()


class ConfigManager:
    """
    Manages application configuration from YAML files.
    Provides singleton access to configuration settings.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.reload()
    
    def reload(self):
        """Reload configuration from files"""
        config_dir = Path(__file__).parent.parent.parent / "config"
        settings_path = config_dir / "settings.yaml"
        
        if not settings_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {settings_path}")
        
        with open(settings_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'embeddings.default_provider')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config = ConfigManager()
            >>> config.get('embeddings.dimension')
            1536
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get validated embedding configuration"""
        config = self.get('embeddings', {})
        return EmbeddingConfig(
            provider=config.get('default_provider', 'openai'),
            dimension=config.get('dimension', 1536),
            batch_size=config.get('batch_size', 100)
        )
    
    def get_chunking_config(self) -> ChunkingConfig:
        """Get validated chunking configuration"""
        config = self.get('document_processing.chunking', {})
        strategy = config.get('default_strategy', 'recursive')
        strategy_config = config.get('strategies', {}).get(strategy, {})
        
        return ChunkingConfig(
            strategy=strategy,
            chunk_size=strategy_config.get('chunk_size', 1000),
            chunk_overlap=strategy_config.get('chunk_overlap', 200)
        )
    
    def get_vectordb_config(self) -> VectorDBConfig:
        """Get validated vector database configuration"""
        default = self.get('vectordb.default', 'chromadb')
        config = self.get(f'vectordb.{default}', {})
        
        return VectorDBConfig(
            provider=default,
            host=config.get('host', 'localhost'),
            port=config.get('port', 8000),
            collection_name=config.get('collection_name', 'company_knowledge'),
            distance_metric=config.get('distance_metric', 'cosine')
        )
    
    def get_rag_config(self) -> RAGConfig:
        """Get validated RAG configuration"""
        retrieval_config = self.get('retrieval', {})
        
        return RAGConfig(
            pattern=self.get('rag_patterns.basic.enabled', 'basic'),
            top_k=retrieval_config.get('top_k', 5),
            score_threshold=retrieval_config.get('score_threshold', 0.7),
            enable_reranking=retrieval_config.get('reranker.enabled', True)
        )
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._config or {}


# Singleton instance
config = ConfigManager()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return config.config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_env_var(key: str, default: Any = None) -> Any:
    """
    Get environment variable with fallback to config.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, config.get(key.lower().replace('_', '.'), default))
