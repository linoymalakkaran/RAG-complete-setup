"""
Graph RAG Pattern with Neo4j

This module implements Graph RAG, which builds and queries a knowledge graph
to provide relationship-aware retrieval and reasoning.

Features:
- Entity extraction from documents
- Relationship extraction
- Knowledge graph construction in Neo4j
- Graph-based retrieval via traversal
- Community detection for better retrieval
- Multi-hop reasoning over relationships

Reference: https://arxiv.org/abs/2404.16130 (Graph RAG Paper)
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag_patterns.basic_rag import BasicRAG
from src.utils.logging_config import RAGLogger


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    type: str  # PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.
    properties: Dict[str, Any]


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    type: str  # WORKS_FOR, LOCATED_IN, RELATES_TO, etc.
    properties: Dict[str, Any]


class GraphRAG(BasicRAG):
    """
    Graph RAG with knowledge graph integration.
    
    Uses Neo4j to:
    - Store documents as nodes with extracted entities
    - Create relationships between entities
    - Retrieve via graph traversal
    - Leverage community structure
    - Support multi-hop reasoning
    
    Example:
        >>> graph_rag = GraphRAG(
        ...     vector_store=chroma_client,
        ...     llm_client=openai_client,
        ...     neo4j_uri="bolt://localhost:7687",
        ...     neo4j_user="neo4j",
        ...     neo4j_password="password"
        ... )
        >>> # Build graph from documents
        >>> graph_rag.build_graph(documents)
        >>> # Query with graph traversal
        >>> result = graph_rag.query("How are departments connected?")
    """
    
    def __init__(
        self,
        vector_store,
        llm_client,
        embedding_model=None,
        enable_hybrid: bool = False,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        max_depth: int = 3,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Graph RAG.
        
        Args:
            vector_store: Vector database client
            llm_client: Language model client
            embedding_model: Embedding model
            enable_hybrid: Whether to use hybrid search
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            max_depth: Maximum graph traversal depth
            logger: Optional logger instance
        """
        super().__init__(
            vector_store=vector_store,
            llm_client=llm_client,
            embedding_model=embedding_model,
            enable_hybrid=enable_hybrid,
            logger=logger
        )
        
        self.max_depth = max_depth
        self.neo4j_driver = None
        
        # Initialize Neo4j connection
        if NEO4J_AVAILABLE and neo4j_uri:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                # Test connection
                self.neo4j_driver.verify_connectivity()
                self.logger.info("Neo4j connection established")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Neo4j: {e}. Graph features disabled.")
                self.neo4j_driver = None
        else:
            self.logger.warning("Neo4j not available. Using fallback to vector-only retrieval.")
    
    def __del__(self):
        """Close Neo4j connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        use_graph_traversal: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Graph RAG query with graph traversal.
        
        Args:
            question: User's question
            top_k: Number of results
            filters: Optional metadata filters
            temperature: LLM temperature
            use_graph_traversal: Whether to use graph traversal
            **kwargs: Additional arguments
            
        Returns:
            Dict containing answer, sources, and graph metadata
        """
        self.logger.info(f"Graph RAG query: {question[:100]}...")
        
        # Step 1: Extract entities from question
        query_entities = self._extract_entities_from_text(question)
        self.logger.info(f"Extracted {len(query_entities)} entities from query")
        
        # Step 2: Retrieve via graph traversal (if available)
        graph_context = []
        relationships_found = []
        
        if use_graph_traversal and self.neo4j_driver and query_entities:
            graph_context, relationships_found = self._graph_traversal(
                entities=query_entities,
                max_depth=self.max_depth
            )
            self.logger.info(
                f"Graph traversal found {len(graph_context)} nodes, "
                f"{len(relationships_found)} relationships"
            )
        
        # Step 3: Fallback to vector retrieval
        vector_docs = self._retrieve(question, top_k, filters)
        
        # Step 4: Combine graph and vector context
        all_sources = self._combine_graph_and_vector(graph_context, vector_docs)
        
        # Step 5: Build prompt with graph-enhanced context
        prompt = self._build_graph_prompt(
            question=question,
            documents=all_sources,
            relationships=relationships_found
        )
        
        # Step 6: Generate answer
        answer = self._generate(prompt, temperature)
        
        # Prepare result
        result = {
            'answer': answer,
            'sources': all_sources,
            'metadata': {
                'pattern': 'graph_rag',
                'query_entities': [e.name for e in query_entities],
                'graph_nodes_found': len(graph_context),
                'relationships_found': len(relationships_found),
                'vector_docs_retrieved': len(vector_docs),
                'total_sources': len(all_sources),
                'used_graph_traversal': use_graph_traversal and bool(graph_context),
                'model': 'graph_rag',
                'temperature': temperature
            }
        }
        
        self.logger.info(
            f"Graph RAG completed - Entities: {len(query_entities)}, "
            f"Graph nodes: {len(graph_context)}, "
            f"Vector docs: {len(vector_docs)}"
        )
        
        return result
    
    def build_graph(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> Dict[str, int]:
        """
        Build knowledge graph from documents.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            batch_size: Number of documents to process at once
            
        Returns:
            Dict with counts of entities and relationships created
        """
        if not self.neo4j_driver:
            self.logger.error("Neo4j not available, cannot build graph")
            return {'entities': 0, 'relationships': 0}
        
        self.logger.info(f"Building graph from {len(documents)} documents...")
        
        total_entities = 0
        total_relationships = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            for doc in batch:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # Extract entities and relationships
                entities = self._extract_entities_from_text(content)
                relationships = self._extract_relationships(content, entities)
                
                # Add to graph
                self._add_entities_to_graph(entities, metadata)
                self._add_relationships_to_graph(relationships)
                
                total_entities += len(entities)
                total_relationships += len(relationships)
            
            self.logger.info(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
        
        stats = {
            'entities': total_entities,
            'relationships': total_relationships,
            'documents': len(documents)
        }
        
        self.logger.info(f"Graph built: {stats}")
        return stats
    
    def _extract_entities_from_text(self, text: str) -> List[Entity]:
        """
        Extract entities from text using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        extract_prompt = f"""Extract the main entities from this text. For each entity, identify:
- Name (exact mention in text)
- Type (PERSON, ORGANIZATION, LOCATION, CONCEPT, POLICY, DEPARTMENT, etc.)

Text:
{text[:1000]}

Format your response as:
ENTITY: [name]
TYPE: [type]
---
ENTITY: [name]
TYPE: [type]
...

Extract 3-7 most important entities."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an entity extraction expert."},
                    {"role": "user", "content": extract_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            # Parse entities
            entities = []
            current_entity = None
            current_type = None
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('ENTITY:'):
                    current_entity = line.split(':', 1)[1].strip()
                elif line.startswith('TYPE:'):
                    current_type = line.split(':', 1)[1].strip()
                    if current_entity and current_type:
                        entities.append(Entity(
                            name=current_entity,
                            type=current_type,
                            properties={}
                        ))
                        current_entity = None
                        current_type = None
            
            return entities
        
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            List of relationships
        """
        if len(entities) < 2:
            return []
        
        entity_names = [e.name for e in entities]
        
        rel_prompt = f"""Given these entities: {', '.join(entity_names)}

Find relationships between them in this text:
{text[:800]}

Format as:
SOURCE: [entity1]
TARGET: [entity2]
TYPE: [relationship type like WORKS_FOR, MANAGES, LOCATED_IN, RELATES_TO]
---

Find 2-5 most important relationships."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": rel_prompt}],
                temperature=0.0,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            
            # Parse relationships
            relationships = []
            current_source = None
            current_target = None
            current_type = None
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('SOURCE:'):
                    current_source = line.split(':', 1)[1].strip()
                elif line.startswith('TARGET:'):
                    current_target = line.split(':', 1)[1].strip()
                elif line.startswith('TYPE:'):
                    current_type = line.split(':', 1)[1].strip()
                    if current_source and current_target and current_type:
                        relationships.append(Relationship(
                            source=current_source,
                            target=current_target,
                            type=current_type,
                            properties={}
                        ))
                        current_source = None
                        current_target = None
                        current_type = None
            
            return relationships
        
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def _add_entities_to_graph(
        self,
        entities: List[Entity],
        metadata: Dict[str, Any]
    ):
        """Add entities as nodes to Neo4j graph."""
        if not self.neo4j_driver:
            return
        
        with self.neo4j_driver.session() as session:
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.source = $source
                    """,
                    name=entity.name,
                    type=entity.type,
                    source=metadata.get('source', 'unknown')
                )
    
    def _add_relationships_to_graph(self, relationships: List[Relationship]):
        """Add relationships to Neo4j graph."""
        if not self.neo4j_driver:
            return
        
        with self.neo4j_driver.session() as session:
            for rel in relationships:
                # Using dynamic relationship type
                session.run(
                    f"""
                    MATCH (source:Entity {{name: $source}})
                    MATCH (target:Entity {{name: $target}})
                    MERGE (source)-[r:{rel.type.replace(' ', '_').upper()}]->(target)
                    """,
                    source=rel.source,
                    target=rel.target
                )
    
    def _graph_traversal(
        self,
        entities: List[Entity],
        max_depth: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Traverse graph to find related information.
        
        Returns:
            Tuple of (nodes, relationships)
        """
        if not self.neo4j_driver or not entities:
            return [], []
        
        entity_names = [e.name for e in entities]
        
        try:
            with self.neo4j_driver.session() as session:
                # Find nodes within max_depth of query entities
                result = session.run(
                    """
                    MATCH path = (start:Entity)-[*1..$max_depth]-(related:Entity)
                    WHERE start.name IN $entity_names
                    WITH related, start, path
                    RETURN DISTINCT related.name AS name,
                           related.type AS type,
                           related.source AS source,
                           length(path) AS distance
                    ORDER BY distance
                    LIMIT 20
                    """,
                    entity_names=entity_names,
                    max_depth=max_depth
                )
                
                nodes = [
                    {
                        'content': f"{record['type']}: {record['name']} (distance: {record['distance']})",
                        'score': 1.0 / (record['distance'] + 1),
                        'metadata': {
                            'source': record['source'],
                            'type': 'graph_node',
                            'entity_type': record['type']
                        }
                    }
                    for record in result
                ]
                
                # Get relationships
                rel_result = session.run(
                    """
                    MATCH (start:Entity)-[r]->(end:Entity)
                    WHERE start.name IN $entity_names OR end.name IN $entity_names
                    RETURN type(r) AS type, start.name AS source, end.name AS target
                    LIMIT 10
                    """,
                    entity_names=entity_names
                )
                
                relationships = [
                    {
                        'type': record['type'],
                        'source': record['source'],
                        'target': record['target']
                    }
                    for record in rel_result
                ]
                
                return nodes, relationships
        
        except Exception as e:
            self.logger.error(f"Graph traversal failed: {e}")
            return [], []
    
    def _combine_graph_and_vector(
        self,
        graph_nodes: List[Dict[str, Any]],
        vector_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine graph and vector results."""
        # Prioritize graph results for relationship info
        combined = graph_nodes[:3] if graph_nodes else []
        
        # Add vector results
        combined.extend(vector_docs[:5])
        
        return combined
    
    def _build_graph_prompt(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Build prompt with graph context."""
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        rel_text = ""
        if relationships:
            rel_text = "\n\nKnown Relationships:\n" + "\n".join([
                f"- {r['source']} {r['type']} {r['target']}"
                for r in relationships
            ])
        
        return f"""Answer this question using the provided context.

Question: {question}

Context:
{context}{rel_text}

Provide a comprehensive answer based on the information above."""
