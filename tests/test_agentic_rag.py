"""
Test Agentic RAG Pattern

Tests the ReAct-based agentic RAG implementation including:
- Question decomposition
- Multi-step reasoning
- Tool selection
- Reflection
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.rag_patterns.agentic_rag import AgenticRAG, AgentAction, AgentStep


class TestAgenticRAG:
    """Test suite for Agentic RAG pattern."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.search = Mock(return_value=[
            {
                'content': 'Vacation policy: 15 days per year',
                'score': 0.9,
                'metadata': {'source': 'hr_policies'}
            }
        ])
        return store
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        
        # Default response
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Test response"
        
        client.chat.completions.create = Mock(return_value=response)
        return client
    
    def test_agentic_rag_initialization(self, mock_vector_store, mock_llm_client):
        """Test AgenticRAG initialization."""
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            max_iterations=5
        )
        
        assert agent.max_iterations == 5
        assert agent.enable_tools == ["search", "clarify", "summarize", "combine"]
        assert len(agent.agent_steps) == 0
    
    def test_question_decomposition_simple(self, mock_vector_store, mock_llm_client):
        """Test decomposition of simple question."""
        # Mock response for simple question
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "SUB-QUESTION 1: What is the vacation policy?"
        mock_llm_client.chat.completions.create = Mock(return_value=response)
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        sub_questions = agent._decompose_question("What is the vacation policy?")
        
        assert len(sub_questions) >= 1
        assert isinstance(sub_questions[0], str)
    
    def test_question_decomposition_complex(self, mock_vector_store, mock_llm_client):
        """Test decomposition of complex question."""
        # Mock response for complex question
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = """SUB-QUESTION 1: What is the engineering vacation policy?
SUB-QUESTION 2: What is the sales vacation policy?
SUB-QUESTION 3: How do these policies compare?"""
        mock_llm_client.chat.completions.create = Mock(return_value=response)
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        sub_questions = agent._decompose_question(
            "Compare vacation policies between engineering and sales"
        )
        
        assert len(sub_questions) > 1
    
    def test_reasoning_step(self, mock_vector_store, mock_llm_client):
        """Test agent reasoning step."""
        # Mock reasoning response
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = """THOUGHT: Need to search for vacation policy information
ACTION: SEARCH
INPUT: vacation policy details"""
        mock_llm_client.chat.completions.create = Mock(return_value=response)
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        thought, action, action_input = agent._reason(
            question="What is the vacation policy?",
            sub_questions=["What is the vacation policy?"],
            accumulated_knowledge=[],
            previous_steps=[]
        )
        
        assert thought is not None
        assert isinstance(action, AgentAction)
        assert action_input is not None
    
    def test_search_action(self, mock_vector_store, mock_llm_client):
        """Test search action execution."""
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        observation = agent._act(
            action=AgentAction.SEARCH,
            action_input="vacation policy",
            top_k=5,
            filters=None
        )
        
        assert observation is not None
        if isinstance(observation, dict):
            assert 'summary' in observation or 'sources' in observation
    
    def test_summarize_action(self, mock_vector_store, mock_llm_client):
        """Test summarize action."""
        # Mock summary response
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Summary of information"
        mock_llm_client.chat.completions.create = Mock(return_value=response)
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        agent.accumulated_knowledge = ["Info 1", "Info 2", "Info 3"]
        
        observation = agent._act(
            action=AgentAction.SUMMARIZE,
            action_input="",
            top_k=5,
            filters=None
        )
        
        assert isinstance(observation, str)
        assert len(observation) > 0
    
    def test_reflection_step(self, mock_vector_store, mock_llm_client):
        """Test agent reflection."""
        # Mock reflection response
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Good progress, found relevant info."
        mock_llm_client.chat.completions.create = Mock(return_value=response)
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        reflection = agent._reflect(
            question="What is the vacation policy?",
            accumulated_knowledge=["Found vacation policy document"],
            current_observation="15 days per year"
        )
        
        assert isinstance(reflection, str)
        assert len(reflection) > 0
    
    def test_max_iterations_limit(self, mock_vector_store, mock_llm_client):
        """Test that agent respects max iterations."""
        # Setup mocks for continuous search
        reason_response = Mock()
        reason_response.choices = [Mock()]
        reason_response.choices[0].message.content = """THOUGHT: Keep searching
ACTION: SEARCH
INPUT: more information"""
        
        decompose_response = Mock()
        decompose_response.choices = [Mock()]
        decompose_response.choices[0].message.content = "SUB-QUESTION 1: test"
        
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message.content = "Final answer"
        
        # Return different responses in sequence
        mock_llm_client.chat.completions.create = Mock(
            side_effect=[decompose_response] + [reason_response] * 10 + [final_response]
        )
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            max_iterations=3
        )
        
        result = agent.query("Test question")
        
        # Should stop at max_iterations
        assert result['metadata']['iterations'] <= 3
        assert result['answer'] is not None
    
    def test_agent_step_tracking(self, mock_vector_store, mock_llm_client):
        """Test that agent steps are properly tracked."""
        decompose_response = Mock()
        decompose_response.choices = [Mock()]
        decompose_response.choices[0].message.content = "SUB-QUESTION 1: test"
        
        reason_response = Mock()
        reason_response.choices = [Mock()]
        reason_response.choices[0].message.content = """THOUGHT: Search for info
ACTION: ANSWER
INPUT: Final answer here"""
        
        mock_llm_client.chat.completions.create = Mock(
            side_effect=[decompose_response, reason_response]
        )
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        result = agent.query("Test question")
        
        assert 'agent_steps' in result['metadata']
        assert len(result['metadata']['agent_steps']) > 0
        assert 'thought' in result['metadata']['agent_steps'][0]
        assert 'action' in result['metadata']['agent_steps'][0]
    
    def test_answer_action_stops_iteration(self, mock_vector_store, mock_llm_client):
        """Test that ANSWER action stops iteration."""
        decompose_response = Mock()
        decompose_response.choices = [Mock()]
        decompose_response.choices[0].message.content = "SUB-QUESTION 1: test"
        
        reason_response = Mock()
        reason_response.choices = [Mock()]
        reason_response.choices[0].message.content = """THOUGHT: Have enough info
ACTION: ANSWER
INPUT: test question"""
        
        answer_response = Mock()
        answer_response.choices = [Mock()]
        answer_response.choices[0].message.content = "This is the final answer"
        
        mock_llm_client.chat.completions.create = Mock(
            side_effect=[decompose_response, reason_response, answer_response]
        )
        
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            max_iterations=10
        )
        
        result = agent.query("Test question")
        
        # Should stop after first iteration when ANSWER is chosen
        assert result['metadata']['iterations'] == 1
        assert "final answer" in result['answer'].lower()
    
    def test_accumulated_knowledge(self, mock_vector_store, mock_llm_client):
        """Test that knowledge accumulates across steps."""
        agent = AgenticRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        # Simulate multiple steps
        agent.accumulated_knowledge.append("Step 1: Found policy document")
        agent.accumulated_knowledge.append("Step 2: Found vacation details")
        
        assert len(agent.accumulated_knowledge) == 2
        assert all(isinstance(k, str) for k in agent.accumulated_knowledge)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
