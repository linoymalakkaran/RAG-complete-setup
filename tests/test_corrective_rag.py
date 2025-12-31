"""
Test Corrective RAG (CRAG) Pattern

Tests the CRAG implementation including:
- Quality evaluation
- Web search fallback
- Source combination
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.rag_patterns.corrective_rag import CorrectiveRAG, RetrievalQuality


class TestCorrectiveRAG:
    """Test suite for Corrective RAG pattern."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.search = Mock(return_value=[
            {
                'content': 'Python 3.11 was released in October 2022',
                'score': 0.85,
                'metadata': {'source': 'internal_docs'}
            },
            {
                'content': 'Python is a programming language',
                'score': 0.75,
                'metadata': {'source': 'internal_docs'}
            }
        ])
        return store
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        
        # Mock evaluation response
        eval_response = Mock()
        eval_response.choices = [Mock()]
        eval_response.choices[0].message.content = """QUALITY: HIGH
SCORE: 0.85
REASONING: Documents contain direct information about Python releases"""
        
        # Mock generation response
        gen_response = Mock()
        gen_response.choices = [Mock()]
        gen_response.choices[0].message.content = "Python 3.11 was released in October 2022."
        
        client.chat.completions.create = Mock(side_effect=[eval_response, gen_response])
        return client
    
    def test_crag_initialization(self, mock_vector_store, mock_llm_client):
        """Test CRAG initialization."""
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            enable_web_search=True
        )
        
        assert crag.enable_web_search == True
        assert crag.quality_threshold_high == 0.7
        assert crag.quality_threshold_low == 0.3
        assert crag.max_web_results == 3
    
    def test_high_quality_retrieval(self, mock_vector_store, mock_llm_client):
        """Test CRAG with high quality retrieval (no web search needed)."""
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            enable_web_search=True
        )
        
        result = crag.query("When was Python 3.11 released?")
        
        assert result['answer'] is not None
        assert result['metadata']['retrieval_quality'] == 'high'
        assert result['metadata']['used_web_search'] == False
        assert len(result['sources']) > 0
    
    @patch('src.rag_patterns.corrective_rag.DuckDuckGoSearchAPIWrapper')
    def test_low_quality_triggers_web_search(
        self,
        mock_ddg,
        mock_vector_store,
        mock_llm_client
    ):
        """Test that low quality retrieval triggers web search."""
        # Setup mock web search
        mock_search = Mock()
        mock_search.results = Mock(return_value=[
            {
                'title': 'Python Releases',
                'snippet': 'Latest Python release information',
                'link': 'https://python.org'
            }
        ])
        mock_ddg.return_value = mock_search
        
        # Mock low quality evaluation
        eval_response = Mock()
        eval_response.choices = [Mock()]
        eval_response.choices[0].message.content = """QUALITY: LOW
SCORE: 0.2
REASONING: Documents don't contain relevant information"""
        
        gen_response = Mock()
        gen_response.choices = [Mock()]
        gen_response.choices[0].message.content = "Latest Python release info from web."
        
        mock_llm_client.chat.completions.create = Mock(
            side_effect=[eval_response, gen_response]
        )
        
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            enable_web_search=True
        )
        
        result = crag.query("What is the latest AI breakthrough?")
        
        assert result['metadata']['retrieval_quality'] == 'low'
        assert result['metadata']['used_web_search'] == True
        assert result['metadata']['web_results_count'] > 0
    
    def test_evaluate_retrieval_quality(self, mock_vector_store, mock_llm_client):
        """Test retrieval quality evaluation."""
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        docs = [
            {'content': 'Relevant document', 'score': 0.9},
            {'content': 'Also relevant', 'score': 0.8}
        ]
        
        assessment = crag._evaluate_retrieval_quality(
            question="Test question",
            documents=docs
        )
        
        assert 'quality' in assessment
        assert 'score' in assessment
        assert 'reasoning' in assessment
        assert isinstance(assessment['quality'], RetrievalQuality)
    
    def test_empty_documents_quality(self, mock_vector_store, mock_llm_client):
        """Test quality evaluation with no documents."""
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        assessment = crag._evaluate_retrieval_quality(
            question="Test question",
            documents=[]
        )
        
        assert assessment['quality'] == RetrievalQuality.LOW
        assert assessment['score'] == 0.0
    
    @patch('src.rag_patterns.corrective_rag.DuckDuckGoSearchAPIWrapper')
    def test_web_search_fallback(self, mock_ddg, mock_vector_store, mock_llm_client):
        """Test web search functionality."""
        mock_search = Mock()
        mock_search.results = Mock(return_value=[
            {
                'title': 'Result 1',
                'snippet': 'Content 1',
                'link': 'https://example.com/1'
            },
            {
                'title': 'Result 2',
                'snippet': 'Content 2',
                'link': 'https://example.com/2'
            }
        ])
        mock_ddg.return_value = mock_search
        
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            enable_web_search=True
        )
        
        results = crag._perform_web_search("test query")
        
        assert len(results) == 2
        assert results[0]['source_type'] == 'web'
        assert 'url' in results[0]['metadata']
    
    def test_combine_sources(self, mock_vector_store, mock_llm_client):
        """Test combining internal and web sources."""
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        internal_docs = [
            {'content': 'Internal 1', 'score': 0.9, 'source_type': 'internal'},
            {'content': 'Internal 2', 'score': 0.6, 'source_type': 'internal'}
        ]
        
        web_docs = [
            {'content': 'Web 1', 'score': 0.95, 'source_type': 'web'},
            {'content': 'Web 2', 'score': 0.90, 'source_type': 'web'}
        ]
        
        combined = crag._combine_sources(internal_docs, web_docs)
        
        assert len(combined) > 0
        # Should include high-scoring internal doc
        assert any(d['content'] == 'Internal 1' for d in combined)
        # Should include web docs
        assert any(d['source_type'] == 'web' for d in combined)
    
    def test_quality_threshold_mapping(self, mock_vector_store, mock_llm_client):
        """Test that quality thresholds correctly map to quality levels."""
        crag = CorrectiveRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            quality_threshold_high=0.7,
            quality_threshold_low=0.3
        )
        
        # Mock responses for different quality levels
        high_eval = Mock()
        high_eval.choices = [Mock()]
        high_eval.choices[0].message.content = "QUALITY: HIGH\nSCORE: 0.9\nREASONING: Excellent"
        
        low_eval = Mock()
        low_eval.choices = [Mock()]
        low_eval.choices[0].message.content = "QUALITY: LOW\nSCORE: 0.2\nREASONING: Poor"
        
        amb_eval = Mock()
        amb_eval.choices = [Mock()]
        amb_eval.choices[0].message.content = "QUALITY: AMBIGUOUS\nSCORE: 0.5\nREASONING: Moderate"
        
        mock_llm_client.chat.completions.create = Mock(
            side_effect=[high_eval, low_eval, amb_eval]
        )
        
        docs = [{'content': 'Test', 'score': 0.5}]
        
        # Test high quality
        high_result = crag._evaluate_retrieval_quality("test", docs)
        assert high_result['quality'] == RetrievalQuality.HIGH
        
        # Test low quality
        low_result = crag._evaluate_retrieval_quality("test", docs)
        assert low_result['quality'] == RetrievalQuality.LOW
        
        # Test ambiguous quality
        amb_result = crag._evaluate_retrieval_quality("test", docs)
        assert amb_result['quality'] == RetrievalQuality.AMBIGUOUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
