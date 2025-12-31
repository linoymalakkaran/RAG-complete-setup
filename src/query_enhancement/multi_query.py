"""
Multi-Query Generation - Generate multiple query variations

Expands a single query into multiple variations for broader retrieval:
- Semantic variations
- Different perspectives
- Synonym expansion
- Temporal/spatial variations
"""

from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from src.llm.openai_client import OpenAIClient
from src.utils.logging_config import RAGLogger


class QueryGenerator(ABC):
    """Base class for query generators."""
    
    @abstractmethod
    def generate(self, query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple query variations."""
        pass


class LLMQueryGenerator(QueryGenerator):
    """
    LLM-based multi-query generation.
    
    Uses an LLM to generate semantically similar queries
    that approach the question from different angles.
    
    Example:
        >>> generator = LLMQueryGenerator()
        >>> queries = generator.generate("What is RAG?", num_queries=3)
        >>> # Returns:
        >>> # [
        >>> #   "What is RAG?",
        >>> #   "Can you explain Retrieval Augmented Generation?",
        >>> #   "How does RAG work in AI systems?"
        >>> # ]
    """
    
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize query generator.
        
        Args:
            llm_client: OpenAI client (creates new if None)
            model: Model to use for generation
        """
        self.llm_client = llm_client or OpenAIClient()
        self.model = model
        self.logger = RAGLogger.get_logger("multi_query")
    
    def generate(
        self,
        query: str,
        num_queries: int = 3,
        include_original: bool = True
    ) -> List[str]:
        """
        Generate multiple query variations.
        
        Args:
            query: Original query
            num_queries: Number of variations to generate
            include_original: Include original query in results
            
        Returns:
            List of query variations
        """
        system_prompt = """You are a query expansion assistant. Generate alternative phrasings of the user's question that:
1. Ask the same thing from different angles
2. Use different terminology/synonyms
3. Are more specific or more general
4. Cover temporal or contextual variations

Generate ONLY the alternative questions, one per line. Do not number them or add explanations."""
        
        user_prompt = f"""Original question: {query}

Generate {num_queries} alternative phrasings of this question."""
        
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.8,  # Higher for diversity
                max_tokens=300
            )
            
            # Parse generated queries
            generated = [
                line.strip()
                for line in response.strip().split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*'))
            ]
            
            # Clean up numbered lists (1., 2., etc.)
            generated = [
                q.split('.', 1)[1].strip() if q.split('.', 1)[0].strip().isdigit() else q
                for q in generated
            ]
            
            # Limit to requested number
            generated = generated[:num_queries]
            
            # Include original if requested
            if include_original:
                queries = [query] + generated
            else:
                queries = generated
            
            self.logger.info(f"Generated {len(queries)} query variations")
            return queries
            
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
            # Fallback to original query
            return [query]


class TemplateQueryGenerator(QueryGenerator):
    """
    Template-based query generation.
    
    Uses predefined templates to generate variations.
    Faster but less flexible than LLM-based generation.
    """
    
    TEMPLATES = [
        "What is {topic}?",
        "Can you explain {topic}?",
        "Tell me about {topic}",
        "How does {topic} work?",
        "What are the key aspects of {topic}?",
        "Describe {topic} in detail",
        "What should I know about {topic}?",
        "{topic} - overview and explanation"
    ]
    
    def __init__(self):
        """Initialize template generator."""
        self.logger = RAGLogger.get_logger("template_query")
    
    def generate(
        self,
        query: str,
        num_queries: int = 3,
        include_original: bool = True
    ) -> List[str]:
        """
        Generate queries using templates.
        
        Args:
            query: Original query
            num_queries: Number of variations
            include_original: Include original query
            
        Returns:
            List of query variations
        """
        # Extract topic from query (simple approach)
        topic = self._extract_topic(query)
        
        # Generate from templates
        generated = []
        for template in self.TEMPLATES[:num_queries]:
            generated.append(template.format(topic=topic))
        
        if include_original:
            queries = [query] + generated
        else:
            queries = generated
        
        self.logger.debug(f"Generated {len(queries)} template-based queries")
        return queries[:num_queries + (1 if include_original else 0)]
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query."""
        # Remove common question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'does', 'do']
        
        words = query.lower().split()
        topic_words = [w for w in words if w not in question_words and len(w) > 2]
        
        return ' '.join(topic_words) if topic_words else query


class MultiQueryRetriever:
    """
    Multi-query retrieval orchestrator.
    
    Generates multiple queries and aggregates results.
    
    Example:
        >>> retriever = MultiQueryRetriever(vector_store, generator)
        >>> results = retriever.retrieve(
        >>>     "What is RAG?",
        >>>     num_queries=3,
        >>>     top_k=5
        >>> )
    """
    
    def __init__(
        self,
        vector_store: Any,
        query_generator: Optional[QueryGenerator] = None
    ):
        """
        Initialize multi-query retriever.
        
        Args:
            vector_store: Vector store for retrieval
            query_generator: Query generator (creates LLM-based if None)
        """
        self.vector_store = vector_store
        self.query_generator = query_generator or LLMQueryGenerator()
        self.logger = RAGLogger.get_logger("multi_query_retriever")
    
    def retrieve(
        self,
        query: str,
        num_queries: int = 3,
        top_k: int = 5,
        fusion_method: str = "rrf"  # 'rrf', 'unique', 'concat'
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using multiple query variations.
        
        Args:
            query: Original query
            num_queries: Number of query variations
            top_k: Documents per query
            fusion_method: How to combine results
                - 'rrf': Reciprocal Rank Fusion
                - 'unique': Remove duplicates, keep highest scores
                - 'concat': Concatenate all results
            
        Returns:
            List of retrieved documents with scores
        """
        # Generate query variations
        queries = self.query_generator.generate(query, num_queries)
        
        self.logger.info(f"Retrieving with {len(queries)} query variations")
        
        # Retrieve for each query
        all_results = []
        for i, q in enumerate(queries):
            try:
                results = self.vector_store.similarity_search_with_score(q, k=top_k)
                all_results.append(results)
                self.logger.debug(f"Query {i+1} retrieved {len(results)} docs")
            except Exception as e:
                self.logger.error(f"Retrieval failed for query {i+1}: {e}")
        
        # Fuse results
        if fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(all_results, top_k)
        elif fusion_method == "unique":
            fused = self._unique_fusion(all_results, top_k)
        else:  # concat
            fused = self._concat_fusion(all_results, top_k)
        
        self.logger.info(f"Fused to {len(fused)} documents")
        return fused
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[tuple]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF).
        
        Score(d) = sum(1 / (k + rank(d)))
        """
        doc_scores = {}
        
        for results in result_lists:
            for rank, (doc, score) in enumerate(results, start=1):
                # Use page_content as doc ID
                doc_id = doc.page_content[:100]  # First 100 chars as ID
                
                rrf_score = 1.0 / (k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0,
                        "original_scores": []
                    }
                
                doc_scores[doc_id]["score"] += rrf_score
                doc_scores[doc_id]["original_scores"].append(score)
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [
            {
                "document": item["doc"],
                "score": item["score"],
                "original_scores": item["original_scores"]
            }
            for item in sorted_docs
        ]
    
    def _unique_fusion(
        self,
        result_lists: List[List[tuple]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Keep unique documents with highest scores."""
        doc_scores = {}
        
        for results in result_lists:
            for doc, score in results:
                doc_id = doc.page_content[:100]
                
                if doc_id not in doc_scores or score > doc_scores[doc_id]["score"]:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "score": score
                    }
        
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return [
            {
                "document": item["doc"],
                "score": item["score"]
            }
            for item in sorted_docs
        ]
    
    def _concat_fusion(
        self,
        result_lists: List[List[tuple]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Concatenate all results."""
        all_docs = []
        
        for results in result_lists:
            for doc, score in results:
                all_docs.append({
                    "document": doc,
                    "score": score
                })
        
        # Sort by score and limit
        sorted_docs = sorted(
            all_docs,
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return sorted_docs


if __name__ == "__main__":
    # Example usage
    generator = LLMQueryGenerator()
    
    queries = generator.generate(
        "What is Retrieval Augmented Generation?",
        num_queries=3
    )
    
    print("Generated queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
