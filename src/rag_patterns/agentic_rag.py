"""
Agentic RAG Pattern with ReAct Framework

This module implements Agentic RAG using the ReAct (Reason + Act) pattern,
where an agent autonomously decides which tools to use and how to combine
information to answer complex questions.

The agent can:
- Decompose complex queries into sub-questions
- Decide which tools to use (search, clarify, summarize, combine)
- Execute multi-step reasoning
- Self-reflect and adjust strategy
- Handle multi-hop questions

Reference: ReAct Paper - https://arxiv.org/abs/2210.03629
"""

from typing import List, Dict, Any, Optional, Literal
import logging
from enum import Enum
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.rag_patterns.basic_rag import BasicRAG
from src.utils.logging_config import RAGLogger


class AgentAction(Enum):
    """Available agent actions."""
    SEARCH = "search"          # Search vector database
    CLARIFY = "clarify"        # Ask clarifying questions
    SUMMARIZE = "summarize"    # Summarize information
    COMBINE = "combine"        # Combine multiple pieces of info
    ANSWER = "answer"          # Provide final answer
    STOP = "stop"              # Stop iteration


@dataclass
class AgentStep:
    """Represents one step in the agent's reasoning."""
    step_number: int
    thought: str
    action: AgentAction
    action_input: str
    observation: str
    reflection: Optional[str] = None


class AgenticRAG(BasicRAG):
    """
    Agentic RAG with autonomous multi-step reasoning.
    
    Uses the ReAct framework:
    - Reason: Think about what to do next
    - Act: Execute an action (search, clarify, etc.)
    - Observe: See the result
    - Reflect: Assess progress and adjust
    
    Example:
        >>> agent = AgenticRAG(
        ...     vector_store=chroma_client,
        ...     llm_client=openai_client,
        ...     max_iterations=5
        ... )
        >>> result = agent.query("Compare vacation policies across departments")
        >>> print(result['answer'])
        >>> print(f"Steps taken: {len(result['metadata']['agent_steps'])}")
    """
    
    def __init__(
        self,
        vector_store,
        llm_client,
        embedding_model=None,
        enable_hybrid: bool = False,
        max_iterations: int = 5,
        enable_tools: Optional[List[str]] = None,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Agentic RAG.
        
        Args:
            vector_store: Vector database client
            llm_client: Language model client
            embedding_model: Embedding model for retrieval
            enable_hybrid: Whether to use hybrid search
            max_iterations: Maximum agent iterations
            enable_tools: List of enabled tools (default: all)
            logger: Optional logger instance
        """
        super().__init__(
            vector_store=vector_store,
            llm_client=llm_client,
            embedding_model=embedding_model,
            enable_hybrid=enable_hybrid,
            logger=logger
        )
        
        self.max_iterations = max_iterations
        self.enable_tools = enable_tools or ["search", "clarify", "summarize", "combine"]
        
        self.agent_steps: List[AgentStep] = []
        self.accumulated_knowledge: List[str] = []
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agentic RAG query with multi-step reasoning.
        
        Args:
            question: User's question
            top_k: Number of documents per search
            filters: Optional metadata filters
            temperature: LLM temperature
            **kwargs: Additional arguments
            
        Returns:
            Dict containing:
                - answer: Final generated answer
                - sources: All retrieved documents from all steps
                - metadata: Including agent_steps, iterations, and reasoning trace
        """
        self.logger.info(f"Agentic RAG query: {question[:100]}...")
        
        # Reset state
        self.agent_steps = []
        self.accumulated_knowledge = []
        all_sources = []
        
        # Step 1: Decompose question (optional for complex queries)
        sub_questions = self._decompose_question(question)
        if len(sub_questions) > 1:
            self.logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
        
        # Step 2: Agent loop (ReAct)
        iteration = 0
        should_continue = True
        final_answer = None
        
        while should_continue and iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"Agent iteration {iteration}/{self.max_iterations}")
            
            # Reason: Decide what to do next
            thought, action, action_input = self._reason(
                question=question,
                sub_questions=sub_questions,
                accumulated_knowledge=self.accumulated_knowledge,
                previous_steps=self.agent_steps
            )
            
            # Act: Execute the action
            observation = self._act(
                action=action,
                action_input=action_input,
                top_k=top_k,
                filters=filters
            )
            
            # Track sources if search was performed
            if action == AgentAction.SEARCH and isinstance(observation, dict):
                if 'sources' in observation:
                    all_sources.extend(observation['sources'])
                observation = observation.get('summary', str(observation))
            
            # Reflect: Assess progress
            reflection = self._reflect(
                question=question,
                accumulated_knowledge=self.accumulated_knowledge,
                current_observation=observation
            )
            
            # Record step
            step = AgentStep(
                step_number=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                reflection=reflection
            )
            self.agent_steps.append(step)
            
            # Update accumulated knowledge
            if action != AgentAction.STOP:
                self.accumulated_knowledge.append(f"Step {iteration}: {observation}")
            
            # Check if we should stop
            if action == AgentAction.ANSWER:
                final_answer = observation
                should_continue = False
            elif action == AgentAction.STOP:
                should_continue = False
                final_answer = self._generate_final_answer(
                    question=question,
                    accumulated_knowledge=self.accumulated_knowledge,
                    temperature=temperature
                )
        
        # Fallback if max iterations reached
        if final_answer is None:
            self.logger.warning("Max iterations reached, generating final answer")
            final_answer = self._generate_final_answer(
                question=question,
                accumulated_knowledge=self.accumulated_knowledge,
                temperature=temperature
            )
        
        # Prepare result
        result = {
            'answer': final_answer,
            'sources': all_sources,
            'metadata': {
                'pattern': 'agentic_rag',
                'iterations': iteration,
                'agent_steps': [
                    {
                        'step': s.step_number,
                        'thought': s.thought,
                        'action': s.action.value,
                        'observation': s.observation[:200] + '...' if len(s.observation) > 200 else s.observation
                    }
                    for s in self.agent_steps
                ],
                'sub_questions_count': len(sub_questions),
                'total_searches': sum(1 for s in self.agent_steps if s.action == AgentAction.SEARCH),
                'temperature': temperature
            }
        }
        
        self.logger.info(
            f"Agentic RAG completed in {iteration} iterations, "
            f"{len(all_sources)} sources, {len(sub_questions)} sub-questions"
        )
        
        return result
    
    def _decompose_question(self, question: str) -> List[str]:
        """
        Decompose complex question into sub-questions.
        
        Args:
            question: Original question
            
        Returns:
            List of sub-questions (may include original if not complex)
        """
        decompose_prompt = f"""Analyze this question and determine if it requires multiple steps to answer.

Question: {question}

If the question is complex (requires multiple pieces of information, comparisons, or multi-hop reasoning), break it down into 2-4 simpler sub-questions.
If the question is simple, return just the original question.

Format your response as:
SUB-QUESTION 1: [question]
SUB-QUESTION 2: [question]
...

If only one question, format as:
SUB-QUESTION 1: {question}"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a question decomposition expert."},
                    {"role": "user", "content": decompose_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            # Parse sub-questions
            sub_questions = []
            for line in content.split('\n'):
                if line.strip().startswith('SUB-QUESTION'):
                    q = line.split(':', 1)[1].strip() if ':' in line else question
                    sub_questions.append(q)
            
            return sub_questions if sub_questions else [question]
        
        except Exception as e:
            self.logger.error(f"Question decomposition failed: {e}")
            return [question]
    
    def _reason(
        self,
        question: str,
        sub_questions: List[str],
        accumulated_knowledge: List[str],
        previous_steps: List[AgentStep]
    ) -> tuple[str, AgentAction, str]:
        """
        Reasoning step: Decide what action to take next.
        
        Returns:
            Tuple of (thought, action, action_input)
        """
        # Build reasoning context
        context = f"""Original Question: {question}

Sub-questions to address:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))}

Accumulated Knowledge:
{chr(10).join(accumulated_knowledge) if accumulated_knowledge else "None yet"}

Previous Steps:
{chr(10).join(f"Step {s.step_number}: {s.action.value} - {s.observation[:100]}" for s in previous_steps[-3:]) if previous_steps else "None yet"}

Available Actions:
- SEARCH: Search the knowledge base for information
- CLARIFY: Identify what information is still needed
- SUMMARIZE: Summarize accumulated knowledge
- COMBINE: Combine multiple pieces of information
- ANSWER: Provide the final answer
- STOP: Stop if unable to answer

Decide the NEXT action to take. Think step-by-step:
1. What have we learned so far?
2. What information is still missing?
3. What action will help most?

Format your response as:
THOUGHT: [your reasoning]
ACTION: [action name]
INPUT: [what to search for / what to do]"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a reasoning agent that plans actions to answer questions."},
                    {"role": "user", "content": context}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            thought = ""
            action = AgentAction.SEARCH
            action_input = question
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('THOUGHT:'):
                    thought = line.split(':', 1)[1].strip()
                elif line.startswith('ACTION:'):
                    action_str = line.split(':', 1)[1].strip().upper()
                    try:
                        action = AgentAction[action_str]
                    except KeyError:
                        action = AgentAction.SEARCH
                elif line.startswith('INPUT:'):
                    action_input = line.split(':', 1)[1].strip()
            
            return thought, action, action_input
        
        except Exception as e:
            self.logger.error(f"Reasoning step failed: {e}")
            return "Searching for relevant information", AgentAction.SEARCH, question
    
    def _act(
        self,
        action: AgentAction,
        action_input: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Any:
        """
        Action step: Execute the chosen action.
        
        Returns:
            Observation from the action
        """
        if action == AgentAction.SEARCH:
            # Perform search
            docs = self._retrieve(action_input, top_k, filters)
            if docs:
                summary = "\n\n".join([
                    f"Doc {i+1} (score: {d.get('score', 0):.2f}): {d['content'][:300]}"
                    for i, d in enumerate(docs[:3])
                ])
                return {
                    'summary': summary,
                    'sources': docs
                }
            return "No relevant documents found"
        
        elif action == AgentAction.CLARIFY:
            # Identify missing information
            return f"Need to search for: {action_input}"
        
        elif action == AgentAction.SUMMARIZE:
            # Summarize accumulated knowledge
            knowledge_text = "\n\n".join(self.accumulated_knowledge)
            summary_prompt = f"Summarize this information concisely:\n\n{knowledge_text}"
            
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Summary: {knowledge_text[:500]}"
        
        elif action == AgentAction.COMBINE:
            # Combine information
            return f"Combined information from previous steps"
        
        elif action == AgentAction.ANSWER:
            # Generate final answer
            return self._generate_final_answer(
                question=action_input,
                accumulated_knowledge=self.accumulated_knowledge,
                temperature=0.7
            )
        
        else:  # STOP
            return "Stopping agent execution"
    
    def _reflect(
        self,
        question: str,
        accumulated_knowledge: List[str],
        current_observation: str
    ) -> str:
        """
        Reflection step: Assess progress and quality.
        
        Returns:
            Reflection on current progress
        """
        if not accumulated_knowledge:
            return "Starting to gather information"
        
        reflection_prompt = f"""Question: {question}

Knowledge so far:
{chr(10).join(accumulated_knowledge[-3:])}

Latest observation:
{current_observation if isinstance(current_observation, str) else str(current_observation)[:200]}

Briefly assess:
1. Are we making progress?
2. Is this information relevant?
3. What's still missing?

Keep response under 50 words."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content
        except:
            return "Progress check: continuing"
    
    def _generate_final_answer(
        self,
        question: str,
        accumulated_knowledge: List[str],
        temperature: float
    ) -> str:
        """Generate final answer from accumulated knowledge."""
        context = "\n\n".join(accumulated_knowledge)
        
        prompt = f"""Based on the following information, answer this question comprehensively:

Question: {question}

Information gathered:
{context}

Provide a clear, complete answer. If the information is insufficient, say so."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides comprehensive answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Final answer generation failed: {e}")
            return "Unable to generate answer from available information."
