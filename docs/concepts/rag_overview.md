# RAG Concepts Overview

## What is RAG?

**Retrieval Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge during generation.

### The Problem RAG Solves

LLMs have limitations:
- ❌ Knowledge cutoff date
- ❌ No access to private/proprietary data
- ❌ Hallucination on factual questions
- ❌ Can't be updated without retraining

### How RAG Works

```
User Query
    ↓
Retrieve relevant documents from knowledge base
    ↓
Augment LLM prompt with retrieved context
    ↓
Generate answer grounded in retrieved documents
```

## Core Components

### 1. Document Processing

**Chunking**: Break documents into manageable pieces
- **Fixed-size**: Simple, consistent size
- **Recursive**: Respects structure (paragraphs → sentences)
- **Semantic**: Groups by topic similarity
- **Parent-document**: Small chunks + large context

### 2. Embeddings

Convert text to vectors in semantic space:
- Similar meanings → Similar vectors
- Enables semantic search (not just keywords)

**Popular Models:**
- OpenAI `text-embedding-3-small` (1536d)
- Cohere `embed-multilingual-v3.0` (1024d)
- Local `all-mpnet-base-v2` (768d)

### 3. Vector Database

Store and search embeddings efficiently:
- **ChromaDB**: Simple, embedded
- **FAISS**: Fast, scalable
- **Pinecone**: Managed, cloud
- **Weaviate**: Feature-rich

### 4. Retrieval

**Dense Retrieval** (Vector Search):
- Semantic similarity via embeddings
- Good for: Understanding intent

**Sparse Retrieval** (BM25):
- Keyword matching with TF-IDF
- Good for: Exact matches, rare terms

**Hybrid**: Combine both for best results

### 5. Generation

Use LLM to generate answer with retrieved context:
```python
prompt = f"""
Context: {retrieved_documents}
Question: {user_question}
Answer:
"""
```

## RAG Patterns

### 1. Basic RAG
```
Query → Retrieve → Generate
```
**Use case**: Simple Q&A, getting started

### 2. Self-RAG
```
Query → Check if retrieval needed → Retrieve (if needed) → Generate → Evaluate quality → Retry if poor
```
**Use case**: Reducing costs, improving quality

### 3. Corrective RAG (CRAG)
```
Query → Retrieve → Evaluate relevance → Use if good, else web search → Generate
```
**Use case**: Handling incomplete knowledge

### 4. Agentic RAG
```
Query → Agent decides: [Search | Clarify | Combine sources] → Generate
```
**Use case**: Complex multi-step queries

### 5. Graph RAG
```
Query → Build knowledge graph → Retrieve via graph traversal → Generate
```
**Use case**: Relationship-heavy data (org charts, dependencies)

### 6. Multimodal RAG
```
Query → Retrieve text + images → Process with vision model → Generate
```
**Use case**: Documents with diagrams, product catalogs

## Key Techniques

### Query Enhancement

**Multi-Query**: Generate variations
```
"What is vacation policy?" →
- "How many vacation days do employees get?"
- "What is the company's PTO policy?"
- "Time off entitlement for staff"
```

**HyDE** (Hypothetical Document Embeddings):
```
Query → Generate hypothetical answer → Embed that → Search
```

**Query Expansion**: Add synonyms and related terms

### Reranking

After initial retrieval, use cross-encoder to rerank:
```
Initial: [Doc1: 0.85, Doc2: 0.82, Doc3: 0.80, ...]
After rerank: [Doc2: 0.91, Doc1: 0.88, Doc3: 0.75, ...]
```

### Context Management

**Lost in the Middle**: LLMs pay less attention to middle of context
- Solution: Place most relevant chunks at start/end

**Context Window**: Limited by model (4K, 8K, 32K tokens)
- Solution: Chunk selection, summarization

### Evaluation

**Retrieval Metrics:**
- Precision@K: % of retrieved docs that are relevant
- Recall@K: % of relevant docs that are retrieved
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

**Generation Metrics (RAGAS):**
- Faithfulness: Grounded in sources?
- Answer Relevance: Addresses question?
- Context Relevance: Sources are relevant?

## Best Practices

### ✅ Do:
- Use hybrid search (dense + sparse)
- Enable reranking for better precision
- Implement caching for repeated queries
- Track metrics and iterate
- Add source citations
- Handle "I don't know" gracefully

### ❌ Don't:
- Make chunks too small (< 100 chars) or too large (> 2000 chars)
- Ignore metadata (very useful for filtering)
- Skip evaluation (you can't improve what you don't measure)
- Hallucinate when sources don't have the answer
- Forget to update embeddings when documents change

## Common Challenges

### Challenge 1: Poor Retrieval Quality
**Symptoms**: Wrong documents retrieved
**Solutions**:
- Try different chunking strategies
- Adjust chunk size
- Enable hybrid search
- Add reranking
- Improve metadata

### Challenge 2: Hallucination
**Symptoms**: Answer not grounded in sources
**Solutions**:
- Lower temperature (0.0-0.2)
- Explicit prompting: "Only use provided context"
- Add source grounding check
- Implement confidence scoring

### Challenge 3: Slow Performance
**Symptoms**: High latency
**Solutions**:
- Implement caching
- Use faster embedding models
- Optimize vector DB (HNSW index)
- Reduce top_k
- Batch processing

### Challenge 4: Inconsistent Quality
**Symptoms**: Some queries work great, others fail
**Solutions**:
- Use Self-RAG for quality checks
- Implement A/B testing
- Track metrics per query type
- Build query classifier

## Resources

### Papers
- [RAG Original Paper](https://arxiv.org/abs/2005.11401)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [CRAG](https://arxiv.org/abs/2401.15884)

### Tools
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [ChromaDB](https://www.trychroma.com/)
- [RAGAS](https://github.com/explodinggradients/ragas)

### Tutorials
- This project's notebooks!
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
