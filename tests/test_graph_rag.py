"""
Tests for Graph RAG pattern

Tests graph construction, entity extraction, relationship detection,
and graph-based retrieval.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

from src.rag_patterns.graph_rag import GraphRAG, Entity, Relationship


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    llm.model = "gpt-4"
    return llm


@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store."""
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = [
        Mock(page_content="Document about AI and machine learning",
             metadata={"source": "ai_doc.txt"})
    ]
    return vectorstore


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "graph_rag": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "entity_types": ["PERSON", "ORGANIZATION", "CONCEPT"],
            "relationship_types": ["WORKS_FOR", "RELATES_TO", "IS_A"],
            "max_depth": 2,
            "min_confidence": 0.7
        }
    }


@pytest.fixture
def graph_rag(mock_llm, mock_vectorstore, mock_config):
    """Create GraphRAG instance."""
    with patch('src.rag_patterns.graph_rag.NEO4J_AVAILABLE', True):
        with patch('src.rag_patterns.graph_rag.GraphDatabase') as mock_gdb:
            # Mock Neo4j driver
            mock_driver = Mock()
            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_gdb.driver.return_value = mock_driver
            
            rag = GraphRAG(
                llm=mock_llm,
                vectorstore=mock_vectorstore,
                config=mock_config
            )
            rag.driver = mock_driver
            return rag


class TestGraphRAGInitialization:
    """Test GraphRAG initialization."""

    def test_initialization_with_neo4j(self, mock_llm, mock_vectorstore, mock_config):
        """Test successful initialization with Neo4j."""
        with patch('src.rag_patterns.graph_rag.NEO4J_AVAILABLE', True):
            with patch('src.rag_patterns.graph_rag.GraphDatabase') as mock_gdb:
                mock_driver = Mock()
                mock_gdb.driver.return_value = mock_driver
                
                rag = GraphRAG(
                    llm=mock_llm,
                    vectorstore=mock_vectorstore,
                    config=mock_config
                )
                
                assert rag.driver is not None
                mock_gdb.driver.assert_called_once_with(
                    "bolt://localhost:7687",
                    auth=("neo4j", "password")
                )

    def test_initialization_without_neo4j(self, mock_llm, mock_vectorstore, mock_config):
        """Test initialization without Neo4j library."""
        with patch('src.rag_patterns.graph_rag.NEO4J_AVAILABLE', False):
            rag = GraphRAG(
                llm=mock_llm,
                vectorstore=mock_vectorstore,
                config=mock_config
            )
            
            assert rag.driver is None


class TestEntityExtraction:
    """Test entity extraction from text."""

    def test_extract_entities_from_text(self, graph_rag):
        """Test entity extraction using LLM."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """[
            {"name": "John Doe", "type": "PERSON", "properties": {}},
            {"name": "Acme Corp", "type": "ORGANIZATION", "properties": {}}
        ]"""
        graph_rag.llm.invoke.return_value = mock_response
        
        text = "John Doe works at Acme Corp"
        entities = graph_rag._extract_entities_from_text(text)
        
        assert len(entities) == 2
        assert entities[0].name == "John Doe"
        assert entities[0].type == "PERSON"
        assert entities[1].name == "Acme Corp"
        assert entities[1].type == "ORGANIZATION"

    def test_extract_entities_handles_invalid_json(self, graph_rag):
        """Test entity extraction with invalid JSON response."""
        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.content = "Not a valid JSON"
        graph_rag.llm.invoke.return_value = mock_response
        
        text = "Some text"
        entities = graph_rag._extract_entities_from_text(text)
        
        assert entities == []


class TestRelationshipExtraction:
    """Test relationship extraction."""

    def test_extract_relationships(self, graph_rag):
        """Test relationship extraction from text and entities."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """[
            {"source": "John Doe", "target": "Acme Corp", "type": "WORKS_FOR", "properties": {}}
        ]"""
        graph_rag.llm.invoke.return_value = mock_response
        
        text = "John Doe works at Acme Corp"
        entities = [
            Entity("John Doe", "PERSON", {}),
            Entity("Acme Corp", "ORGANIZATION", {})
        ]
        
        relationships = graph_rag._extract_relationships(text, entities)
        
        assert len(relationships) == 1
        assert relationships[0].source == "John Doe"
        assert relationships[0].target == "Acme Corp"
        assert relationships[0].type == "WORKS_FOR"

    def test_extract_relationships_handles_invalid_json(self, graph_rag):
        """Test relationship extraction with invalid JSON."""
        mock_response = Mock()
        mock_response.content = "Not valid JSON"
        graph_rag.llm.invoke.return_value = mock_response
        
        text = "Some text"
        entities = [Entity("Entity1", "CONCEPT", {})]
        
        relationships = graph_rag._extract_relationships(text, entities)
        
        assert relationships == []


class TestGraphOperations:
    """Test graph database operations."""

    def test_add_entities_to_graph(self, graph_rag):
        """Test adding entities to Neo4j."""
        entities = [
            Entity("John Doe", "PERSON", {"age": 30}),
            Entity("Acme Corp", "ORGANIZATION", {"industry": "Tech"})
        ]
        
        mock_session = Mock()
        graph_rag.driver.session.return_value.__enter__.return_value = mock_session
        
        graph_rag._add_entities_to_graph(entities)
        
        # Should call session.run for each entity
        assert mock_session.run.call_count == 2

    def test_add_relationships_to_graph(self, graph_rag):
        """Test adding relationships to Neo4j."""
        relationships = [
            Relationship("John Doe", "Acme Corp", "WORKS_FOR", {}),
            Relationship("Acme Corp", "Tech Industry", "OPERATES_IN", {})
        ]
        
        mock_session = Mock()
        graph_rag.driver.session.return_value.__enter__.return_value = mock_session
        
        graph_rag._add_relationships_to_graph(relationships)
        
        # Should call session.run for each relationship
        assert mock_session.run.call_count == 2

    def test_graph_operations_without_driver(self, graph_rag):
        """Test graph operations when Neo4j is not available."""
        graph_rag.driver = None
        
        entities = [Entity("Test", "CONCEPT", {})]
        relationships = [Relationship("A", "B", "RELATES_TO", {})]
        
        # Should not raise errors
        graph_rag._add_entities_to_graph(entities)
        graph_rag._add_relationships_to_graph(relationships)


class TestGraphTraversal:
    """Test graph traversal for retrieval."""

    def test_graph_traversal(self, graph_rag):
        """Test graph traversal from entity."""
        # Mock graph query results
        mock_record = Mock()
        mock_record.data.return_value = {
            'source': 'John Doe',
            'relationship': 'WORKS_FOR',
            'target': 'Acme Corp',
            'properties': {}
        }
        
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        
        mock_session = Mock()
        mock_session.run.return_value = mock_result
        graph_rag.driver.session.return_value.__enter__.return_value = mock_session
        
        relationships = graph_rag._graph_traversal("John Doe", max_depth=1)
        
        assert len(relationships) == 1
        assert relationships[0]['source'] == 'John Doe'
        assert relationships[0]['relationship'] == 'WORKS_FOR'
        assert relationships[0]['target'] == 'Acme Corp'

    def test_graph_traversal_without_driver(self, graph_rag):
        """Test graph traversal when Neo4j unavailable."""
        graph_rag.driver = None
        
        relationships = graph_rag._graph_traversal("Entity", max_depth=2)
        
        assert relationships == []


class TestBuildGraph:
    """Test graph construction from documents."""

    def test_build_graph_from_documents(self, graph_rag):
        """Test building graph from document collection."""
        documents = [
            {"content": "John Doe works at Acme Corp"},
            {"content": "Acme Corp is a tech company"}
        ]
        
        # Mock entity extraction
        mock_entities_response = Mock()
        mock_entities_response.content = """[
            {"name": "John Doe", "type": "PERSON", "properties": {}},
            {"name": "Acme Corp", "type": "ORGANIZATION", "properties": {}}
        ]"""
        
        # Mock relationship extraction
        mock_relations_response = Mock()
        mock_relations_response.content = """[
            {"source": "John Doe", "target": "Acme Corp", "type": "WORKS_FOR", "properties": {}}
        ]"""
        
        graph_rag.llm.invoke.side_effect = [
            mock_entities_response, mock_relations_response,
            mock_entities_response, mock_relations_response
        ]
        
        mock_session = Mock()
        graph_rag.driver.session.return_value.__enter__.return_value = mock_session
        
        result = graph_rag.build_graph(documents)
        
        assert result["status"] == "success"
        assert result["entities_added"] > 0
        assert result["relationships_added"] > 0


class TestQueryExecution:
    """Test query execution with graph context."""

    def test_query_with_graph_context(self, graph_rag):
        """Test querying with graph and vector context."""
        question = "Where does John Doe work?"
        
        # Mock vector search
        graph_rag.vectorstore.similarity_search.return_value = [
            Mock(page_content="John Doe works at Acme Corp",
                 metadata={"source": "doc1.txt"})
        ]
        
        # Mock entity extraction from query
        mock_entity_response = Mock()
        mock_entity_response.content = """[
            {"name": "John Doe", "type": "PERSON", "properties": {}}
        ]"""
        
        # Mock final answer generation
        mock_answer_response = Mock()
        mock_answer_response.content = "John Doe works at Acme Corp, a tech company."
        
        graph_rag.llm.invoke.side_effect = [
            mock_entity_response,  # Entity extraction
            mock_answer_response   # Answer generation
        ]
        
        # Mock graph traversal
        mock_record = Mock()
        mock_record.data.return_value = {
            'source': 'John Doe',
            'relationship': 'WORKS_FOR',
            'target': 'Acme Corp',
            'properties': {}
        }
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        
        mock_session = Mock()
        mock_session.run.return_value = mock_result
        graph_rag.driver.session.return_value.__enter__.return_value = mock_session
        
        result = graph_rag.query(question)
        
        assert "answer" in result
        assert "John Doe works at Acme Corp" in result["answer"]
        assert "metadata" in result
        assert result["metadata"]["pattern"] == "Graph RAG"

    def test_query_without_graph_driver(self, graph_rag):
        """Test querying falls back when no graph available."""
        graph_rag.driver = None
        question = "Test question?"
        
        # Mock vector search
        graph_rag.vectorstore.similarity_search.return_value = [
            Mock(page_content="Test content",
                 metadata={"source": "test.txt"})
        ]
        
        # Mock answer generation
        mock_response = Mock()
        mock_response.content = "Test answer"
        graph_rag.llm.invoke.return_value = mock_response
        
        result = graph_rag.query(question)
        
        assert "answer" in result
        assert result["answer"] == "Test answer"


class TestCombineGraphAndVector:
    """Test combining graph and vector results."""

    def test_combine_results(self, graph_rag):
        """Test combining graph relationships with vector docs."""
        documents = [
            {"content": "Doc 1", "metadata": {"source": "doc1.txt"}},
            {"content": "Doc 2", "metadata": {"source": "doc2.txt"}}
        ]
        
        relationships = [
            {'source': 'A', 'relationship': 'RELATES_TO', 'target': 'B', 'properties': {}},
            {'source': 'B', 'relationship': 'IS_A', 'target': 'C', 'properties': {}}
        ]
        
        combined = graph_rag._combine_graph_and_vector(documents, relationships)
        
        assert len(combined) == len(documents) + len(relationships)
        # First items should be documents
        assert combined[0]["content"] == "Doc 1"
        assert combined[1]["content"] == "Doc 2"
        # Then relationships
        assert "source" in combined[2]
        assert "relationship" in combined[3]


class TestPromptBuilding:
    """Test prompt construction with graph context."""

    def test_build_graph_prompt(self, graph_rag):
        """Test building prompt with graph relationships."""
        question = "Test question?"
        documents = [
            {"content": "Document content", "metadata": {"source": "doc.txt"}}
        ]
        relationships = [
            {'source': 'A', 'relationship': 'RELATES_TO', 'target': 'B', 'properties': {}}
        ]
        
        prompt = graph_rag._build_graph_prompt(question, documents, relationships)
        
        assert question in prompt
        assert "Document content" in prompt
        assert "A RELATES_TO B" in prompt

    def test_build_graph_prompt_without_relationships(self, graph_rag):
        """Test prompt building with no relationships."""
        question = "Test question?"
        documents = [{"content": "Content", "metadata": {}}]
        
        prompt = graph_rag._build_graph_prompt(question, documents, [])
        
        assert question in prompt
        assert "Content" in prompt
        assert "Relationships" not in prompt or prompt.count("Relationships") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
