"""
Query Expansion - Expand queries with synonyms and related terms

Enhances queries by:
- Adding synonyms
- Including related terms
- Expanding abbreviations
- Adding context-specific terms
"""

from typing import List, Dict, Any, Optional, Set
import logging

from src.llm.openai_client import OpenAIClient
from src.utils.logging_config import RAGLogger


class QueryExpander:
    """
    Query expansion with synonyms and related terms.
    
    Expands queries to improve retrieval coverage by:
    1. Adding synonyms for key terms
    2. Including related concepts
    3. Expanding abbreviations
    4. Adding domain-specific variations
    
    Example:
        >>> expander = QueryExpander()
        >>> expanded = expander.expand("ML model training")
        >>> # Returns: "ML model training machine learning algorithm optimization..."
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        model: str = "gpt-4o-mini",
        use_llm: bool = True
    ):
        """
        Initialize query expander.
        
        Args:
            llm_client: OpenAI client
            model: Model for expansion
            use_llm: Use LLM for expansion (vs. rule-based)
        """
        self.llm_client = llm_client or OpenAIClient()
        self.model = model
        self.use_llm = use_llm
        
        self.logger = RAGLogger.get_logger("query_expansion")
        
        # Common abbreviation mappings
        self.abbreviations = {
            "ML": "machine learning",
            "AI": "artificial intelligence",
            "NLP": "natural language processing",
            "RAG": "retrieval augmented generation",
            "LLM": "large language model",
            "GPU": "graphics processing unit",
            "API": "application programming interface",
            "UI": "user interface",
            "UX": "user experience",
            "DB": "database",
            "SQL": "structured query language"
        }
    
    def expand(
        self,
        query: str,
        max_expansions: int = 5,
        include_synonyms: bool = True,
        include_related: bool = True,
        expand_abbreviations: bool = True
    ) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansion terms
            include_synonyms: Include synonym expansions
            include_related: Include related concepts
            expand_abbreviations: Expand abbreviations
            
        Returns:
            Expanded query string
        """
        if self.use_llm:
            return self._llm_expansion(
                query,
                max_expansions,
                include_synonyms,
                include_related,
                expand_abbreviations
            )
        else:
            return self._rule_based_expansion(
                query,
                expand_abbreviations
            )
    
    def get_expansion_terms(
        self,
        query: str,
        max_terms: int = 10
    ) -> List[str]:
        """
        Get list of expansion terms (without combining into query).
        
        Args:
            query: Original query
            max_terms: Maximum expansion terms
            
        Returns:
            List of expansion terms
        """
        if not self.use_llm:
            return self._rule_based_terms(query)
        
        system_prompt = """You are a query expansion assistant.
Generate expansion terms for the query - synonyms, related concepts, abbreviations.

Return ONLY a comma-separated list of expansion terms, nothing else."""
        
        user_prompt = f"""Query: {query}

Expansion terms (max {max_terms}):"""
        
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.7,
                max_tokens=150
            )
            
            # Parse terms
            terms = [
                term.strip()
                for term in response.split(',')
                if term.strip() and term.strip().lower() not in query.lower()
            ]
            
            return terms[:max_terms]
            
        except Exception as e:
            self.logger.error(f"Term expansion failed: {e}")
            return []
    
    def _llm_expansion(
        self,
        query: str,
        max_expansions: int,
        include_synonyms: bool,
        include_related: bool,
        expand_abbreviations: bool
    ) -> str:
        """LLM-based query expansion."""
        
        expansion_types = []
        if include_synonyms:
            expansion_types.append("synonyms")
        if include_related:
            expansion_types.append("related concepts")
        if expand_abbreviations:
            expansion_types.append("expanded abbreviations")
        
        system_prompt = f"""You are a query expansion assistant.
Expand the query by adding {', '.join(expansion_types)}.

Rules:
1. Keep the original query
2. Add expansion terms that would help find relevant documents
3. Don't change the query's meaning
4. Add up to {max_expansions} expansion terms

Return the expanded query as a single line."""
        
        try:
            response = self.llm_client.generate(
                prompt=query,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.5,
                max_tokens=200
            )
            
            expanded = response.strip()
            
            self.logger.info(f"Expanded query: {len(query)} -> {len(expanded)} chars")
            return expanded
            
        except Exception as e:
            self.logger.error(f"LLM expansion failed: {e}")
            return query
    
    def _rule_based_expansion(
        self,
        query: str,
        expand_abbreviations: bool
    ) -> str:
        """Rule-based query expansion."""
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            
            # Expand abbreviations
            if expand_abbreviations and word.upper() in self.abbreviations:
                expansion = self.abbreviations[word.upper()]
                expanded_words.append(expansion)
                self.logger.debug(f"Expanded {word} -> {expansion}")
        
        return ' '.join(expanded_words)
    
    def _rule_based_terms(self, query: str) -> List[str]:
        """Get expansion terms using rules."""
        
        terms = []
        words = query.split()
        
        for word in words:
            # Check for abbreviations
            if word.upper() in self.abbreviations:
                terms.append(self.abbreviations[word.upper()])
        
        return terms
    
    def expand_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        max_expansions: int = 5
    ) -> str:
        """
        Expand query with additional context.
        
        Args:
            query: Original query
            context: Additional context (domain, previous queries, etc.)
            max_expansions: Maximum expansion terms
            
        Returns:
            Context-aware expanded query
        """
        if not self.use_llm:
            return self.expand(query)
        
        # Build context description
        context_parts = []
        
        if "domain" in context:
            context_parts.append(f"Domain: {context['domain']}")
        
        if "previous_queries" in context:
            prev = context['previous_queries']
            if prev:
                context_parts.append(f"Previous queries: {', '.join(prev[-3:])}")
        
        if "user_preferences" in context:
            prefs = context['user_preferences']
            context_parts.append(f"User focus: {prefs}")
        
        context_str = "\n".join(context_parts)
        
        system_prompt = f"""You are a context-aware query expansion assistant.

Context:
{context_str}

Expand the query considering this context. Add up to {max_expansions} relevant terms.
Keep the original query and add expansion terms."""
        
        try:
            response = self.llm_client.generate(
                prompt=query,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.5,
                max_tokens=200
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Context expansion failed: {e}")
            return query


class PRFExpander(QueryExpander):
    """
    Pseudo-Relevance Feedback (PRF) query expansion.
    
    Uses top retrieved documents to extract expansion terms.
    Assumes top-k documents are relevant.
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        model: str = "gpt-4o-mini"
    ):
        """Initialize PRF expander."""
        super().__init__(llm_client, model)
        self.logger = RAGLogger.get_logger("prf_expansion")
    
    def expand_from_documents(
        self,
        query: str,
        top_documents: List[Any],
        max_terms: int = 5
    ) -> str:
        """
        Expand query using pseudo-relevance feedback.
        
        Args:
            query: Original query
            top_documents: Top-k retrieved documents (assumed relevant)
            max_terms: Maximum expansion terms
            
        Returns:
            Expanded query
        """
        # Extract content from top documents
        doc_contents = []
        for doc in top_documents[:3]:  # Use top 3 docs
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('page_content', ''))
            else:
                content = str(doc)
            
            # Truncate
            doc_contents.append(content[:500])
        
        combined_docs = "\n\n".join(doc_contents)
        
        system_prompt = f"""You are a query expansion assistant using pseudo-relevance feedback.

Given a query and the top retrieved documents, extract {max_terms} important terms
that would improve the query.

Return ONLY comma-separated expansion terms, nothing else."""
        
        user_prompt = f"""Query: {query}

Top retrieved documents:
{combined_docs}

Expansion terms:"""
        
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.3,
                max_tokens=100
            )
            
            # Parse terms
            terms = [
                term.strip()
                for term in response.split(',')
                if term.strip()
            ]
            
            # Combine with original query
            expanded = f"{query} {' '.join(terms[:max_terms])}"
            
            self.logger.info(f"PRF expansion added {len(terms)} terms")
            return expanded
            
        except Exception as e:
            self.logger.error(f"PRF expansion failed: {e}")
            return query


if __name__ == "__main__":
    # Example usage
    expander = QueryExpander()
    
    # Simple expansion
    query = "ML model training"
    expanded = expander.expand(query)
    
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")
    
    # Get expansion terms
    terms = expander.get_expansion_terms(query)
    print(f"\nExpansion terms: {', '.join(terms)}")
    
    # Context-aware expansion
    context = {
        "domain": "healthcare",
        "previous_queries": ["patient records", "medical imaging"]
    }
    
    context_expanded = expander.expand_with_context(query, context)
    print(f"\nContext-aware: {context_expanded}")
