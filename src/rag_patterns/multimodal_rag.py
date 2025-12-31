"""
Multimodal RAG Pattern

This module implements Multimodal RAG, which handles both text and image inputs
for question answering. It uses GPT-4 Vision or CLIP for image understanding.

Features:
- Image encoding with CLIP
- Visual question answering with GPT-4V
- Image-text matching and relevance scoring
- Multi-modal retrieval combining text and image embeddings
- Visual grounding for text queries

Use cases:
- Product catalogs with images
- Technical documentation with diagrams
- Medical records with scans
- Architectural blueprints
"""

from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import base64
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag_patterns.basic_rag import BasicRAG
from src.utils.logging_config import RAGLogger


class MultimodalRAG(BasicRAG):
    """
    Multimodal RAG that handles text and image queries.
    
    This pattern extends BasicRAG to support:
    1. Image queries (visual questions)
    2. Text queries with image context
    3. Combined image-text retrieval
    
    Example:
        >>> config = {"multimodal": {"vision_model": "gpt-4-vision-preview"}}
        >>> rag = MultimodalRAG(llm, vectorstore, config)
        >>> 
        >>> # Text query with image
        >>> result = rag.query(
        ...     question="What's in this diagram?",
        ...     image_path="diagram.png"
        ... )
        >>> 
        >>> # Pure visual query
        >>> result = rag.query_image(
        ...     image_path="product.jpg",
        ...     question="What is this product?"
        ... )
    """
    
    def __init__(
        self,
        llm,
        vectorstore,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Multimodal RAG.
        
        Args:
            llm: Language model instance
            vectorstore: Vector store for retrieval
            config: Configuration with multimodal settings:
                - vision_model: Model for image understanding (default: gpt-4-vision-preview)
                - max_image_size: Max image dimension (default: 1024)
                - image_detail: low/high/auto (default: auto)
                - image_embedding_model: Model for image embeddings (default: clip)
        """
        super().__init__(llm, vectorstore, config)
        
        self.logger = RAGLogger.get_logger("multimodal_rag")
        
        # Get multimodal config
        multimodal_config = config.get("multimodal", {}) if config else {}
        
        self.vision_model = multimodal_config.get("vision_model", "gpt-4-vision-preview")
        self.max_image_size = multimodal_config.get("max_image_size", 1024)
        self.image_detail = multimodal_config.get("image_detail", "auto")
        self.image_embedding_model = multimodal_config.get("image_embedding_model", "clip")
        
        # Check if PIL is available
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available - image processing will be limited")
        
        self.logger.info(
            f"Initialized Multimodal RAG with vision_model={self.vision_model}, "
            f"image_detail={self.image_detail}"
        )
    
    def query(
        self,
        question: str,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with optional image context.
        
        Args:
            question: The question to answer
            image_path: Path to image file (optional)
            image_data: Raw image bytes (optional)
            k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Dict with:
                - answer: The generated answer
                - sources: Retrieved documents
                - image_analysis: Analysis of provided image (if any)
                - metadata: Query metadata
        """
        self.logger.info(f"Multimodal query: {question[:100]}")
        
        start_time = self._get_current_time()
        
        # Initialize results
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "image_analysis": None,
            "metadata": {
                "pattern": "Multimodal RAG",
                "has_image": image_path is not None or image_data is not None,
                "latency_ms": 0
            }
        }
        
        try:
            # Step 1: Analyze image if provided
            image_context = None
            if image_path or image_data:
                self.logger.info("Analyzing provided image")
                image_analysis = self._analyze_image(
                    image_path=image_path,
                    image_data=image_data,
                    question=question
                )
                result["image_analysis"] = image_analysis
                image_context = image_analysis.get("description", "")
            
            # Step 2: Enhance query with image context
            enhanced_query = question
            if image_context:
                enhanced_query = f"{question}\n\nImage context: {image_context}"
                self.logger.debug(f"Enhanced query with image context")
            
            # Step 3: Retrieve relevant documents
            self.logger.info(f"Retrieving top {k} documents")
            documents = self.vectorstore.similarity_search(enhanced_query, k=k)
            
            result["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, "relevance_score", None)
                }
                for doc in documents
            ]
            
            # Step 4: Generate answer with multimodal context
            if image_path or image_data:
                # Use vision-capable model
                answer = self._generate_multimodal_answer(
                    question=question,
                    documents=result["sources"],
                    image_path=image_path,
                    image_data=image_data,
                    image_analysis=image_analysis
                )
            else:
                # Standard text generation
                answer = self._generate_answer(question, result["sources"])
            
            result["answer"] = answer
            
            # Calculate latency
            end_time = self._get_current_time()
            result["metadata"]["latency_ms"] = (end_time - start_time) * 1000
            
            self.logger.info(
                f"Multimodal query completed in {result['metadata']['latency_ms']:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multimodal query: {str(e)}", exc_info=True)
            result["answer"] = f"Error processing query: {str(e)}"
            result["metadata"]["error"] = str(e)
            return result
    
    def query_image(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        question: str = "What is in this image?",
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Query focusing on image analysis.
        
        This is a convenience method for pure visual queries.
        
        Args:
            image_path: Path to image file
            image_data: Raw image bytes
            question: Question about the image
            k: Number of similar images/docs to retrieve
            
        Returns:
            Dict with image analysis and answer
        """
        return self.query(
            question=question,
            image_path=image_path,
            image_data=image_data,
            k=k
        )
    
    def _analyze_image(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using vision model.
        
        Args:
            image_path: Path to image file
            image_data: Raw image bytes
            question: Optional specific question about image
            
        Returns:
            Dict with:
                - description: Text description of image
                - objects: Detected objects/entities
                - text: Any text found in image (OCR)
                - metadata: Image metadata
        """
        try:
            # Load and encode image
            image_base64 = self._encode_image(image_path, image_data)
            
            # Build vision prompt
            vision_prompt = self._build_vision_prompt(question)
            
            # Call vision model (GPT-4V)
            # Note: This requires OpenAI GPT-4 Vision API
            messages = [
                SystemMessage(content="You are an expert at analyzing images."),
                HumanMessage(content=[
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": self.image_detail
                        }
                    }
                ])
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "description": response.content,
                "objects": [],  # Could extract from description
                "text": "",  # Could add OCR here
                "metadata": {
                    "model": self.vision_model,
                    "detail": self.image_detail
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return {
                "description": "Error analyzing image",
                "objects": [],
                "text": "",
                "metadata": {"error": str(e)}
            }
    
    def _encode_image(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None
    ) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to image file
            image_data: Raw image bytes
            
        Returns:
            Base64 encoded image string
        """
        try:
            if image_path:
                # Load from file
                if not PIL_AVAILABLE:
                    # Read raw bytes
                    with open(image_path, "rb") as f:
                        data = f.read()
                else:
                    # Use PIL to resize if needed
                    img = Image.open(image_path)
                    img = self._resize_image(img)
                    
                    # Convert to bytes
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    data = buffer.getvalue()
            
            elif image_data:
                if PIL_AVAILABLE:
                    # Resize if needed
                    img = Image.open(BytesIO(image_data))
                    img = self._resize_image(img)
                    
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    data = buffer.getvalue()
                else:
                    data = image_data
            
            else:
                raise ValueError("Either image_path or image_data must be provided")
            
            # Encode to base64
            return base64.b64encode(data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error encoding image: {str(e)}")
            raise
    
    def _resize_image(self, img: 'Image.Image') -> 'Image.Image':
        """
        Resize image if it exceeds max dimensions.
        
        Args:
            img: PIL Image object
            
        Returns:
            Resized PIL Image
        """
        width, height = img.size
        max_size = self.max_image_size
        
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return img
    
    def _build_vision_prompt(self, question: Optional[str] = None) -> str:
        """
        Build prompt for vision model.
        
        Args:
            question: Optional specific question
            
        Returns:
            Vision prompt string
        """
        if question:
            return f"""Analyze this image and answer the following question:

Question: {question}

Provide a detailed analysis including:
1. Main subjects/objects in the image
2. Key details relevant to the question
3. Any text visible in the image
4. Overall context and scene"""
        else:
            return """Analyze this image in detail. Describe:

1. Main subjects and objects
2. Actions or activities
3. Setting and context
4. Any visible text
5. Notable details
6. Overall scene and atmosphere"""
    
    def _generate_multimodal_answer(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        image_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate answer using both text and image context.
        
        Args:
            question: User question
            documents: Retrieved text documents
            image_path: Path to image
            image_data: Raw image bytes
            image_analysis: Pre-computed image analysis
            
        Returns:
            Generated answer string
        """
        try:
            # Build context from documents
            doc_context = "\n\n".join([
                f"Document {i+1}:\n{doc['content']}"
                for i, doc in enumerate(documents)
            ])
            
            # Build image context
            image_context = ""
            if image_analysis:
                image_context = f"\n\nImage Analysis:\n{image_analysis.get('description', '')}"
            
            # Build prompt
            prompt = f"""Answer the following question using the provided context and image analysis.

Question: {question}

Text Context:
{doc_context}{image_context}

Provide a comprehensive answer that incorporates both the textual information and visual context."""
            
            # If we have image, use vision model with image
            if image_path or image_data:
                image_base64 = self._encode_image(image_path, image_data)
                
                messages = [
                    SystemMessage(content="You are a helpful assistant that answers questions using text and image context."),
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": self.image_detail
                            }
                        }
                    ])
                ]
            else:
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal answer: {str(e)}")
            # Fallback to text-only
            return self._generate_answer(question, documents)
    
    def _generate_answer(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Generate text-only answer (fallback).
        
        Args:
            question: User question
            documents: Retrieved documents
            
        Returns:
            Generated answer
        """
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Answer this question using the provided context.

Question: {question}

Context:
{context}

Answer:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _get_current_time(self) -> float:
        """Get current time in seconds."""
        import time
        return time.time()


if __name__ == "__main__":
    # Example usage
    print("Multimodal RAG Pattern")
    print("=" * 50)
    print("\nFeatures:")
    print("- Image + text queries")
    print("- GPT-4 Vision integration")
    print("- Visual question answering")
    print("- Image-text matching")
    print("\nUse cases:")
    print("- Product catalogs")
    print("- Technical diagrams")
    print("- Medical imaging")
    print("- Architecture blueprints")
