"""
Tests for Multimodal RAG pattern

Tests image processing, vision model integration, and multimodal retrieval.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from typing import List, Dict, Any
import base64

from src.rag_patterns.multimodal_rag import MultimodalRAG


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    llm.model = "gpt-4-vision-preview"
    return llm


@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store."""
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = [
        Mock(page_content="Product documentation for camera X100",
             metadata={"source": "camera_doc.pdf"})
    ]
    return vectorstore


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "multimodal": {
            "vision_model": "gpt-4-vision-preview",
            "max_image_size": 1024,
            "image_detail": "auto",
            "image_embedding_model": "clip"
        }
    }


@pytest.fixture
def multimodal_rag(mock_llm, mock_vectorstore, mock_config):
    """Create MultimodalRAG instance."""
    return MultimodalRAG(
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        config=mock_config
    )


class TestMultimodalRAGInitialization:
    """Test MultimodalRAG initialization."""

    def test_initialization_with_config(self, mock_llm, mock_vectorstore, mock_config):
        """Test successful initialization with config."""
        rag = MultimodalRAG(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            config=mock_config
        )
        
        assert rag.vision_model == "gpt-4-vision-preview"
        assert rag.max_image_size == 1024
        assert rag.image_detail == "auto"
        assert rag.image_embedding_model == "clip"

    def test_initialization_with_defaults(self, mock_llm, mock_vectorstore):
        """Test initialization with default settings."""
        rag = MultimodalRAG(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            config={}
        )
        
        assert rag.vision_model == "gpt-4-vision-preview"
        assert rag.max_image_size == 1024
        assert rag.image_detail == "auto"


class TestImageEncoding:
    """Test image encoding functionality."""

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    def test_encode_image_from_path_without_pil(self, mock_file, multimodal_rag):
        """Test encoding image from file path without PIL."""
        with patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', False):
            result = multimodal_rag._encode_image(image_path="test.jpg")
            
            assert isinstance(result, str)
            # Verify it's base64 encoded
            decoded = base64.b64decode(result)
            assert decoded == b'fake_image_data'

    def test_encode_image_from_bytes_without_pil(self, multimodal_rag):
        """Test encoding image from bytes without PIL."""
        with patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', False):
            image_data = b'test_image_bytes'
            result = multimodal_rag._encode_image(image_data=image_data)
            
            assert isinstance(result, str)
            decoded = base64.b64decode(result)
            assert decoded == image_data

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', True)
    @patch('src.rag_patterns.multimodal_rag.Image')
    def test_encode_image_with_pil_resize(self, mock_image_class, multimodal_rag):
        """Test encoding with PIL and resizing."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (2048, 1536)  # Larger than max_image_size
        mock_img.save = Mock()
        mock_img.resize = Mock(return_value=mock_img)
        
        mock_image_class.open.return_value = mock_img
        mock_image_class.Resampling.LANCZOS = 1
        
        # Mock BytesIO
        with patch('src.rag_patterns.multimodal_rag.BytesIO') as mock_bytesio:
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b'resized_image'
            mock_bytesio.return_value = mock_buffer
            
            result = multimodal_rag._encode_image(image_path="large.jpg")
            
            # Should have called resize
            mock_img.resize.assert_called_once()

    def test_encode_image_raises_without_input(self, multimodal_rag):
        """Test that encoding raises error without image path or data."""
        with pytest.raises(ValueError, match="Either image_path or image_data"):
            multimodal_rag._encode_image()


class TestImageResize:
    """Test image resizing functionality."""

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', True)
    def test_resize_large_width(self, multimodal_rag):
        """Test resizing image with large width."""
        from src.rag_patterns.multimodal_rag import Image as PILImage
        
        mock_img = Mock()
        mock_img.size = (2048, 1000)
        mock_img.resize = Mock(return_value=mock_img)
        
        with patch.object(PILImage, 'Resampling') as mock_resampling:
            mock_resampling.LANCZOS = 1
            
            result = multimodal_rag._resize_image(mock_img)
            
            # Should resize to max_size maintaining aspect ratio
            mock_img.resize.assert_called_once()
            call_args = mock_img.resize.call_args[0][0]
            assert call_args[0] == 1024  # Width should be max_size

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', True)
    def test_resize_large_height(self, multimodal_rag):
        """Test resizing image with large height."""
        mock_img = Mock()
        mock_img.size = (1000, 2048)
        mock_img.resize = Mock(return_value=mock_img)
        
        from src.rag_patterns.multimodal_rag import Image as PILImage
        with patch.object(PILImage, 'Resampling') as mock_resampling:
            mock_resampling.LANCZOS = 1
            
            result = multimodal_rag._resize_image(mock_img)
            
            mock_img.resize.assert_called_once()
            call_args = mock_img.resize.call_args[0][0]
            assert call_args[1] == 1024  # Height should be max_size

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', True)
    def test_no_resize_when_small(self, multimodal_rag):
        """Test that small images are not resized."""
        mock_img = Mock()
        mock_img.size = (512, 384)
        
        result = multimodal_rag._resize_image(mock_img)
        
        # Should not call resize
        mock_img.resize.assert_not_called()
        assert result == mock_img


class TestVisionPrompts:
    """Test vision prompt building."""

    def test_build_vision_prompt_with_question(self, multimodal_rag):
        """Test building prompt with specific question."""
        question = "What type of camera is this?"
        prompt = multimodal_rag._build_vision_prompt(question)
        
        assert question in prompt
        assert "Analyze this image" in prompt
        assert "Main subjects/objects" in prompt

    def test_build_vision_prompt_without_question(self, multimodal_rag):
        """Test building general analysis prompt."""
        prompt = multimodal_rag._build_vision_prompt(None)
        
        assert "Analyze this image in detail" in prompt
        assert "Main subjects and objects" in prompt
        assert "visible text" in prompt


class TestImageAnalysis:
    """Test image analysis with vision model."""

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', False)
    @patch('builtins.open', new_callable=mock_open, read_data=b'image_data')
    def test_analyze_image_success(self, mock_file, multimodal_rag):
        """Test successful image analysis."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a professional DSLR camera with multiple lenses."
        multimodal_rag.llm.invoke.return_value = mock_response
        
        result = multimodal_rag._analyze_image(
            image_path="camera.jpg",
            question="What is this device?"
        )
        
        assert result["description"] == mock_response.content
        assert "model" in result["metadata"]
        assert result["metadata"]["model"] == "gpt-4-vision-preview"

    def test_analyze_image_error_handling(self, multimodal_rag):
        """Test error handling in image analysis."""
        # Make encoding fail
        with patch.object(multimodal_rag, '_encode_image', side_effect=Exception("Encoding failed")):
            result = multimodal_rag._analyze_image(image_path="bad.jpg")
            
            assert "Error" in result["description"]
            assert "error" in result["metadata"]


class TestQueryExecution:
    """Test query execution with images."""

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', False)
    @patch('builtins.open', new_callable=mock_open, read_data=b'image_data')
    def test_query_with_image_path(self, mock_file, multimodal_rag):
        """Test querying with image path."""
        # Mock image analysis
        mock_analysis_response = Mock()
        mock_analysis_response.content = "Camera image description"
        
        # Mock final answer
        mock_answer_response = Mock()
        mock_answer_response.content = "This is a professional camera suitable for photography."
        
        multimodal_rag.llm.invoke.side_effect = [
            mock_analysis_response,
            mock_answer_response
        ]
        
        result = multimodal_rag.query(
            question="What type of camera is this?",
            image_path="camera.jpg"
        )
        
        assert "answer" in result
        assert "image_analysis" in result
        assert result["metadata"]["has_image"] is True
        assert "professional camera" in result["answer"]

    def test_query_without_image(self, multimodal_rag):
        """Test standard text-only query."""
        mock_response = Mock()
        mock_response.content = "Standard text answer"
        multimodal_rag.llm.invoke.return_value = mock_response
        
        result = multimodal_rag.query(question="What is RAG?")
        
        assert "answer" in result
        assert result["metadata"]["has_image"] is False
        assert result["image_analysis"] is None

    def test_query_image_convenience_method(self, multimodal_rag):
        """Test query_image convenience method."""
        with patch.object(multimodal_rag, 'query') as mock_query:
            mock_query.return_value = {"answer": "Image answer"}
            
            result = multimodal_rag.query_image(
                image_path="test.jpg",
                question="What is this?"
            )
            
            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["image_path"] == "test.jpg"
            assert call_kwargs["question"] == "What is this?"


class TestMultimodalAnswerGeneration:
    """Test multimodal answer generation."""

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', False)
    @patch('builtins.open', new_callable=mock_open, read_data=b'img')
    def test_generate_multimodal_answer_with_image(self, mock_file, multimodal_rag):
        """Test generating answer with image context."""
        documents = [
            {"content": "Camera X100 specs: 24MP sensor, 4K video"},
            {"content": "Camera X100 price: $1299"}
        ]
        
        image_analysis = {
            "description": "Professional DSLR camera with detachable lens",
            "objects": ["camera", "lens"],
            "text": "X100"
        }
        
        mock_response = Mock()
        mock_response.content = "This is the Camera X100, a professional DSLR."
        multimodal_rag.llm.invoke.return_value = mock_response
        
        answer = multimodal_rag._generate_multimodal_answer(
            question="What camera is this?",
            documents=documents,
            image_path="camera.jpg",
            image_analysis=image_analysis
        )
        
        assert "Camera X100" in answer
        assert "professional" in answer

    def test_generate_answer_fallback(self, multimodal_rag):
        """Test fallback to text-only generation."""
        documents = [
            {"content": "RAG stands for Retrieval Augmented Generation"}
        ]
        
        mock_response = Mock()
        mock_response.content = "RAG is a technique that combines retrieval with generation."
        multimodal_rag.llm.invoke.return_value = mock_response
        
        answer = multimodal_rag._generate_answer(
            question="What is RAG?",
            documents=documents
        )
        
        assert "RAG" in answer
        assert "retrieval" in answer.lower()


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_query_handles_analysis_error(self, multimodal_rag):
        """Test query continues even if image analysis fails."""
        with patch.object(multimodal_rag, '_analyze_image', side_effect=Exception("Analysis failed")):
            with patch.object(multimodal_rag, '_generate_answer', return_value="Fallback answer"):
                result = multimodal_rag.query(
                    question="Test?",
                    image_path="bad.jpg"
                )
                
                assert "error" in result["metadata"]

    def test_multimodal_answer_falls_back_on_error(self, multimodal_rag):
        """Test multimodal answer generation falls back to text-only on error."""
        with patch.object(multimodal_rag, '_encode_image', side_effect=Exception("Encoding error")):
            with patch.object(multimodal_rag, '_generate_answer', return_value="Text answer") as mock_gen:
                
                answer = multimodal_rag._generate_multimodal_answer(
                    question="Test?",
                    documents=[{"content": "Doc"}],
                    image_path="bad.jpg"
                )
                
                # Should fall back to text generation
                mock_gen.assert_called_once()


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('src.rag_patterns.multimodal_rag.PIL_AVAILABLE', False)
    @patch('builtins.open', new_callable=mock_open, read_data=b'camera_image')
    def test_complete_multimodal_workflow(self, mock_file, multimodal_rag):
        """Test complete workflow: image + text query."""
        # Mock all LLM calls
        analysis_response = Mock()
        analysis_response.content = "Professional camera with lens"
        
        final_response = Mock()
        final_response.content = "This is a Camera X100, professional DSLR camera."
        
        multimodal_rag.llm.invoke.side_effect = [
            analysis_response,
            final_response
        ]
        
        result = multimodal_rag.query(
            question="What camera model is this?",
            image_path="camera.jpg",
            k=5
        )
        
        # Verify complete result structure
        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert "image_analysis" in result
        assert "metadata" in result
        
        assert result["metadata"]["pattern"] == "Multimodal RAG"
        assert result["metadata"]["has_image"] is True
        assert "latency_ms" in result["metadata"]
        
        assert "Camera X100" in result["answer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
