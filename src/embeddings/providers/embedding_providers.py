"""
Embedding providers for RAG.

Supports multiple embedding models:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Cohere (embed-multilingual-v3.0)
- Local models (sentence-transformers)
"""

import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

import openai
import cohere
from sentence_transformers import SentenceTransformer

from src.utils.logging_config import setup_logging
from src.utils.config_manager import config

logger = setup_logging("rag.embeddings")


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All providers should implement:
        - embed_text: Embed single text
        - embed_batch: Embed multiple texts efficiently
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch (more efficient).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name"""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """
    OpenAI embedding provider.
    
    Models:
        - text-embedding-3-small: 1536 dimensions, fast, cost-effective
        - text-embedding-3-large: 3072 dimensions, highest quality
        - text-embedding-ada-002: 1536 dimensions, legacy
    
    Features:
        - High quality embeddings
        - Batch processing
        - Automatic retries
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Dimension mapping
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Initialized OpenAI embeddings: {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch.
        
        OpenAI supports up to 2048 texts per request.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {str(e)}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)
    
    @property
    def model_name(self) -> str:
        return f"openai/{self.model}"


class CohereEmbedding(EmbeddingProvider):
    """
    Cohere embedding provider.
    
    Models:
        - embed-multilingual-v3.0: Multilingual, 1024 dimensions
        - embed-english-v3.0: English-only, 1024 dimensions
    
    Features:
        - Multilingual support
        - Input type specification (search vs classification)
        - Compression options
    """
    
    def __init__(
        self,
        model: str = "embed-multilingual-v3.0",
        input_type: str = "search_document",
        api_key: Optional[str] = None
    ):
        """
        Initialize Cohere embedding provider.
        
        Args:
            model: Cohere embedding model
            input_type: "search_document", "search_query", or "classification"
            api_key: Cohere API key
        """
        self.model = model
        self.input_type = input_type
        self.client = cohere.Client(
            api_key=api_key or os.getenv("COHERE_API_KEY")
        )
        
        logger.info(f"Initialized Cohere embeddings: {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type
            )
            return response.embeddings[0]
            
        except Exception as e:
            logger.error(f"Cohere embedding error: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch"""
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=self.input_type
            )
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Cohere batch embedding error: {str(e)}")
            raise
    
    @property
    def dimension(self) -> int:
        return 1024  # All Cohere v3 models use 1024
    
    @property
    def model_name(self) -> str:
        return f"cohere/{self.model}"


class LocalEmbedding(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Popular models:
        - all-mpnet-base-v2: 768 dim, good quality
        - all-MiniLM-L6-v2: 384 dim, fast
        - paraphrase-multilingual-mpnet-base-v2: Multilingual
    
    Pros:
        - No API costs
        - Privacy (runs locally)
        - No rate limits
        - Offline capable
    
    Cons:
        - Requires GPU for speed
        - Larger model files
        - Slightly lower quality than OpenAI
    """
    
    def __init__(
        self,
        model: str = "all-mpnet-base-v2",
        device: str = "cpu"
    ):
        """
        Initialize local embedding provider.
        
        Args:
            model: Sentence transformer model name
            device: "cpu" or "cuda"
        """
        self.model_path = f"sentence-transformers/{model}"
        self.device = device
        
        logger.info(f"Loading local model: {model} on {device}")
        self.model = SentenceTransformer(self.model_path, device=device)
        
        logger.info(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Local embedding error: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch.
        
        Uses batching and GPU (if available) for efficiency.
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Local batch embedding error: {str(e)}")
            raise
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        return self.model_path


class MultimodalEmbedding:
    """
    Multimodal embedding for images and text.
    
    Uses CLIP (Contrastive Language-Image Pre-training) to create
    embeddings in a shared space for images and text.
    
    Use cases:
        - Image search with text queries
        - Document retrieval with diagrams
        - Product search
    """
    
    def __init__(self, model: str = "openai/clip-vit-base-patch32"):
        """
        Initialize multimodal embedding.
        
        Args:
            model: CLIP model name
        """
        from transformers import CLIPModel, CLIPProcessor
        
        logger.info(f"Loading multimodal model: {model}")
        self.model = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)
        
        logger.info("Multimodal model loaded")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        return text_features[0].detach().numpy().tolist()
    
    def embed_image(self, image_path: str) -> List[float]:
        """Embed image"""
        from PIL import Image
        
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features[0].detach().numpy().tolist()
    
    @property
    def dimension(self) -> int:
        return self.model.config.projection_dim


class EmbeddingFactory:
    """
    Factory to create embedding providers.
    
    Usage:
        >>> embedder = EmbeddingFactory.create("openai")
        >>> embedding = embedder.embed_text("Hello world")
    """
    
    @staticmethod
    def create(
        provider: str = "openai",
        **kwargs
    ) -> EmbeddingProvider:
        """
        Create embedding provider.
        
        Args:
            provider: "openai", "cohere", or "local"
            **kwargs: Provider-specific arguments
            
        Returns:
            EmbeddingProvider instance
        """
        providers = {
            "openai": OpenAIEmbedding,
            "cohere": CohereEmbedding,
            "local": LocalEmbedding
        }
        
        if provider not in providers:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(providers.keys())}"
            )
        
        provider_class = providers[provider]
        return provider_class(**kwargs)
    
    @staticmethod
    def create_from_config() -> EmbeddingProvider:
        """Create embedding provider from configuration"""
        provider = config.get('embeddings.default_provider', 'openai')
        provider_config = config.get(f'embeddings.providers.{provider}', {})
        
        return EmbeddingFactory.create(provider, **provider_config)


def compare_embeddings(
    text: str,
    providers: List[str] = ["openai", "cohere", "local"]
) -> Dict[str, Any]:
    """
    Compare embeddings from different providers.
    
    Useful for understanding differences in embedding quality and
    choosing the right provider for your use case.
    
    Args:
        text: Sample text to embed
        providers: List of providers to compare
        
    Returns:
        Comparison results
        
    Example:
        >>> comparison = compare_embeddings("Machine learning is amazing")
        >>> for provider, result in comparison.items():
        ...     print(f"{provider}: {result['dimension']} dimensions")
    """
    results = {}
    
    for provider_name in providers:
        try:
            logger.info(f"Testing {provider_name} embeddings")
            embedder = EmbeddingFactory.create(provider_name)
            
            import time
            start = time.time()
            embedding = embedder.embed_text(text)
            elapsed = time.time() - start
            
            results[provider_name] = {
                'dimension': embedder.dimension,
                'model': embedder.model_name,
                'latency': elapsed,
                'sample': embedding[:5]  # First 5 values
            }
            
        except Exception as e:
            logger.error(f"Error testing {provider_name}: {str(e)}")
            results[provider_name] = {'error': str(e)}
    
    return results


def visualize_similarity(
    texts: List[str],
    provider: str = "openai"
) -> np.ndarray:
    """
    Create similarity matrix for visualization.
    
    Args:
        texts: List of texts to compare
        provider: Embedding provider
        
    Returns:
        NxN similarity matrix
        
    Example:
        >>> texts = ["cat", "dog", "car", "truck"]
        >>> sim_matrix = visualize_similarity(texts)
        >>> # Now plot with matplotlib/seaborn
    """
    embedder = EmbeddingFactory.create(provider)
    embeddings = np.array(embedder.embed_batch(texts))
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix
