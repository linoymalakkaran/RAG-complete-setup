"""
Unit Tests for Query Enhancement

Tests for:
- Multi-Query Generation
- HyDE
- Reranking
- Query Expansion
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.query_enhancement import (
    LLMQueryGenerator,
    TemplateQueryGenerator,
    HyDE,
    CrossEncoderReranker,
    QueryExpander
)


class TestMultiQuery:
    """Tests for multi-query generation."""
    
    def test_template_generator(self):
        """Test template-based query generation."""
        generator = TemplateQueryGenerator()
        
        queries = generator.generate("What is RAG?", num_queries=3)
        
        assert len(queries) == 4  # Original + 3 variations
        assert "What is RAG?" in queries
    
    def test_template_generator_without_original(self):
        """Test template generation without original."""
        generator = TemplateQueryGenerator()
        
        queries = generator.generate(
            "What is RAG?",
            num_queries=3,
            include_original=False
        )
        
        assert len(queries) == 3
        assert "What is RAG?" not in queries


class TestHyDE:
    """Tests for HyDE."""
    
    def test_hyde_initialization(self):
        """Test HyDE initialization."""
        hyde = HyDE(num_documents=2)
        
        assert hyde.num_documents == 2
        assert hyde.model == "gpt-4o-mini"


class TestReranking:
    """Tests for reranking."""
    
    def test_cross_encoder_initialization(self):
        """Test cross-encoder initialization."""
        reranker = CrossEncoderReranker()
        
        # Should initialize without error
        assert reranker.model_name is not None


class TestQueryExpansion:
    """Tests for query expansion."""
    
    def test_rule_based_expansion(self):
        """Test rule-based query expansion."""
        expander = QueryExpander(use_llm=False)
        
        # Test abbreviation expansion
        query = "What is ML and AI?"
        expanded = expander.expand(query, expand_abbreviations=True)
        
        assert "machine learning" in expanded.lower()
        assert "artificial intelligence" in expanded.lower()
    
    def test_get_expansion_terms(self):
        """Test getting expansion terms."""
        expander = QueryExpander(use_llm=False)
        
        terms = expander.get_expansion_terms("ML model training")
        
        # Should find ML expansion
        assert any("machine learning" in term.lower() for term in terms)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
