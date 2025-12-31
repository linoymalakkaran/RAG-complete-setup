# Company Policy & Knowledge Assistant - Comprehensive RAG Learning Project

A production-ready RAG (Retrieval Augmented Generation) system demonstrating all major concepts and patterns for enterprise knowledge management.

> **📊 Implementation Status:** 55-58% Complete | **✅ Core Features:** Working | **🎓 Learning Value:** Excellent  
> **See:** [VERIFICATION_SUMMARY.md](VERIFICATION_SUMMARY.md) for detailed verification report

## 🚀 Quick Start

```bash
# Windows
start.bat

# Mac/Linux
chmod +x start.sh && ./start.sh
```

Or manually:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Verify setup
python setup_verify.py

# 4. Run Streamlit UI
streamlit run ui/app.py
```

**✅ What Works Right Now:**
- Document processing (PDF, Word, images, text)
- 4 chunking strategies with auto-optimizer  
- 3 embedding providers (OpenAI, Cohere, Local)
- Hybrid retrieval (BM25 + vector search)
- 2 RAG patterns (Basic + Self-RAG)
- Interactive Streamlit UI
- Complete tutorial notebook

## 🎯 Project Overview

This project implements a complete RAG pipeline that allows employees to query company policies, HR documents, technical documentation, and training materials. It's designed as a learning platform showcasing:

- 6 different RAG patterns (Basic, Self-RAG, CRAG, Agentic, Graph RAG, Multimodal)
- Multiple chunking strategies with optimization
- Hybrid search (dense + sparse retrieval)
- Multiple embedding models comparison
- Production-ready features (caching, monitoring, evaluation)
- Security guardrails and PII protection

## ✨ Feature Matrix

| Feature | Status | Description |
|---------|--------|-------------|
| **Document Processing** |
| PDF Loading | ✅ | Extract text from PDFs with OCR fallback |
| Word Loading | ✅ | Extract from .docx with tables |
| Image OCR | ✅ | Tesseract integration for scanned docs |
| Fixed Chunking | ✅ | Simple character-based splitting |
| Recursive Chunking | ✅ | Hierarchical paragraph/sentence splitting |
| Semantic Chunking | ✅ | Group by semantic similarity |
| Parent-Document Chunking | ✅ | Small chunks + large context |
| Chunk Optimizer | ✅ | Auto-recommend best settings |
| **Embeddings** |
| OpenAI Embeddings | ✅ | text-embedding-3-small/large |
| Cohere Embeddings | ✅ | embed-multilingual-v3.0 |
| Local Embeddings | ✅ | Sentence-transformers (free) |
| CLIP Multimodal | ✅ | Image + text embeddings |
| Embedding Comparison | ✅ | Side-by-side benchmarking |
| **Retrieval** |
| Vector Search | ✅ | Dense semantic search |
| BM25 Sparse Search | ✅ | Keyword-based retrieval |
| Hybrid Search | ✅ | Weighted combination |
| Reciprocal Rank Fusion | ✅ | Advanced fusion method |
| Reranking | 🔨 | Cross-encoder reranking |
| **Vector Databases** |
| ChromaDB | ✅ | Primary vector store |
| FAISS | 🔨 | Fast similarity search |
| Neo4j (Graph RAG) | 🔨 | Knowledge graph storage |
| **RAG Patterns** |
| Basic RAG | ✅ | Simple retrieve-generate |
| Self-RAG | ✅ | With quality reflection |
| Corrective RAG | 🔨 | Web search fallback |
| Agentic RAG | 🔨 | Multi-step reasoning |
| Graph RAG | 🔨 | Relationship-aware |
| Multimodal RAG | 🔨 | Image + text queries |
| **Query Enhancement** |
| Multi-Query | 🔨 | Generate query variations |
| HyDE | 🔨 | Hypothetical document embeddings |
| Query Expansion | 🔨 | Add synonyms/related terms |
| **Context Management** |
| Buffer Memory | 🔨 | Recent conversation history |
| Summarization | 🔨 | Long-term memory |
| Context Window Manager | 🔨 | Smart context selection |
| **Evaluation** |
| Precision@K | 🔨 | Retrieval precision |
| Recall@K | 🔨 | Retrieval recall |
| MRR/NDCG | 🔨 | Ranking metrics |
| RAGAS Integration | 🔨 | Answer quality metrics |
| **User Interface** |
| Main Dashboard | ✅ | Overview and quick start |
| Document Upload | ✅ | With chunking preview |
| Query Playground | ✅ | Interactive testing |
| Pattern Comparison | 🔨 | Side-by-side comparison |
| Vector Explorer | 🔨 | UMAP visualization |
| Knowledge Graph Viewer | 🔨 | Graph visualization |
| Evaluation Dashboard | 🔨 | Metrics and trends |
| Settings Page | 🔨 | Configuration UI |
| **Production Features** |
| Semantic Caching | 🔨 | Cache similar queries |
| Exact Caching | 🔨 | Cache repeated queries |
| FastAPI Server | 🔨 | REST API endpoints |
| Authentication | 🔨 | JWT-based auth |
| Rate Limiting | 🔨 | API throttling |
| Monitoring | 🔨 | Prometheus/Grafana |
| MLflow Tracking | 🔨 | Experiment tracking |
| **Security** |
| PII Redaction | 🔨 | Remove sensitive data |
| Prompt Injection Detection | 🔨 | Security guardrails |
| Hallucination Detection | 🔨 | Source grounding check |
| Confidence Scoring | 🔨 | Answer confidence |

**Legend**: ✅ Implemented | 🔨 Scaffold Ready | ⏳ Planned

## 🏗️ Architecture

```
┌─────────────────┐
│  Documents      │
│ (PDF/Word/Imgs) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Document Processing Pipeline       │
│  - OCR (images/PDFs)               │
│  - Multiple chunking strategies     │
│  - Metadata extraction             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Embedding Layer                    │
│  - OpenAI / Cohere / Local         │
│  - Dense + Sparse (BM25)           │
│  - Multimodal embeddings           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Vector Databases                   │
│  - ChromaDB (primary)              │
│  - FAISS (comparison)              │
│  - Neo4j (Graph RAG)               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Query Processing                   │
│  - Multi-query generation          │
│  - HyDE                            │
│  - Query expansion                 │
│  - Reranking                       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  RAG Patterns (6 variations)        │
│  - Basic → Self → CRAG → Agentic   │
│  - Graph RAG → Multimodal          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Response Generation                │
│  - Context management              │
│  - Conversation memory             │
│  - Source citation                 │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Guardrails & Evaluation            │
│  - Security checks                 │
│  - Hallucination detection         │
│  - Quality metrics (RAGAS)         │
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
rag/
├── config/                      # Configuration files
│   ├── settings.yaml           # Main configuration
│   ├── embedding_models.yaml   # Embedding model configs
│   └── rag_patterns.yaml       # RAG pattern configurations
│
├── src/
│   ├── ingestion/              # Document processing
│   │   ├── loaders/           # PDF, Word, image, video loaders
│   │   ├── chunking/          # Multiple chunking strategies
│   │   ├── ocr/               # OCR processing
│   │   └── optimizer.py       # Chunk size optimizer
│   │
│   ├── embeddings/             # Embedding layer
│   │   ├── providers/         # OpenAI, Cohere, local models
│   │   ├── hybrid.py          # Dense + sparse retrieval
│   │   └── multimodal.py      # Image/video embeddings
│   │
│   ├── vectordb/               # Vector database layer
│   │   ├── chromadb_client.py
│   │   ├── faiss_client.py
│   │   ├── neo4j_client.py    # For Graph RAG
│   │   └── benchmark.py        # DB comparison tools
│   │
│   ├── rag_patterns/           # RAG implementations
│   │   ├── basic_rag.py
│   │   ├── self_rag.py
│   │   ├── corrective_rag.py
│   │   ├── agentic_rag.py
│   │   ├── graph_rag.py
│   │   └── multimodal_rag.py
│   │
│   ├── query/                  # Query enhancement
│   │   ├── multi_query.py
│   │   ├── hyde.py
│   │   ├── expansion.py
│   │   └── reranker.py
│   │
│   ├── context/                # Context management
│   │   ├── memory.py
│   │   ├── summarization.py
│   │   └── window_manager.py
│   │
│   ├── cache/                  # Caching layer
│   │   ├── semantic_cache.py
│   │   └── exact_cache.py
│   │
│   ├── evaluation/             # Metrics & evaluation
│   │   ├── retrieval_metrics.py
│   │   ├── response_metrics.py
│   │   ├── ragas_integration.py
│   │   └── debugger.py
│   │
│   ├── guardrails/             # Security & safety
│   │   ├── input_guards.py
│   │   ├── output_guards.py
│   │   └── pii_redaction.py
│   │
│   ├── frameworks/             # Framework comparisons
│   │   ├── langchain_impl.py
│   │   ├── llamaindex_impl.py
│   │   └── crewai_impl.py
│   │
│   ├── api/                    # FastAPI server
│   │   ├── main.py
│   │   ├── routes/
│   │   └── middleware/
│   │
│   └── utils/                  # Shared utilities
│       ├── logging_config.py
│       └── helpers.py
│
├── ui/                         # Streamlit UI
│   ├── app.py                 # Main app
│   ├── pages/
│   │   ├── 1_document_upload.py
│   │   ├── 2_query_playground.py
│   │   ├── 3_pattern_comparison.py
│   │   ├── 4_vector_explorer.py
│   │   ├── 5_knowledge_graph.py
│   │   ├── 6_evaluation_dashboard.py
│   │   └── 7_settings.py
│   └── components/            # Reusable UI components
│
├── notebooks/                  # Learning notebooks
│   ├── 01_chunking_strategies.ipynb
│   ├── 02_embedding_comparison.ipynb
│   ├── 03_rag_patterns.ipynb
│   ├── 04_evaluation_metrics.ipynb
│   └── 05_production_optimization.ipynb
│
├── data/                       # Sample data
│   ├── sample_documents/
│   │   ├── hr_policies/
│   │   ├── technical_docs/
│   │   ├── training_materials/
│   │   └── images/
│   └── processed/             # Processed chunks & embeddings
│
├── mlruns/                     # MLflow tracking
├── tests/                      # Unit & integration tests
├── docs/                       # Documentation
│   ├── concepts/              # RAG concept explanations
│   └── api/                   # API documentation
│
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
# Start the Streamlit UI
streamlit run ui/app.py

# Or start the FastAPI server
uvicorn src.api.main:app --reload

# Or launch Jupyter notebooks for learning
jupyter notebook notebooks/
```

## 📚 RAG Patterns Explained

### 1. Basic RAG
Simple retrieve-and-generate pattern. Query → Retrieve top-k chunks → Generate response.

### 2. Self-RAG
Adds self-reflection: checks if retrieval is needed, validates answer quality against sources.

### 3. Corrective RAG (CRAG)
Evaluates retrieval quality. If internal docs insufficient, triggers web search fallback.

### 4. Agentic RAG
Autonomous decision-making. Agent decides whether to search, clarify, or combine multiple retrievals.

### 5. Graph RAG
Builds knowledge graph from documents. Retrieves via graph traversal for relationship-aware answers.

### 6. Multimodal RAG
Handles queries about images, diagrams, and video content in documents.

## 🎓 Learning Path

1. **Start with notebooks**: Work through numbered notebooks to understand each concept
2. **Experiment in UI**: Use Query Playground to see patterns in action
3. **Compare approaches**: Use Pattern Comparison page for side-by-side evaluation
4. **Tune parameters**: Use Settings page to experiment with configurations
5. **Monitor quality**: Use Evaluation Dashboard to track metrics

## 🔧 Configuration

Edit `config/settings.yaml` to customize:
- Embedding models
- Chunk sizes and strategies
- Vector database settings
- RAG pattern parameters
- Cache settings
- Guardrail thresholds

## 📊 Evaluation Metrics

### Retrieval Metrics
- **Precision@K**: Proportion of retrieved docs that are relevant
- **Recall@K**: Proportion of relevant docs that are retrieved
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
- **Hit Rate**: Whether any relevant doc was retrieved

### Response Metrics (RAGAS)
- **Faithfulness**: Answer grounded in sources
- **Answer Relevance**: Answer addresses the question
- **Context Relevance**: Retrieved context is relevant
- **Answer Similarity**: Compared to reference answers

## 🛡️ Security Features

- **Input Guardrails**: Prompt injection detection, off-topic filtering
- **Output Guardrails**: PII redaction, hallucination detection
- **Topic Guardrails**: Ensures responses stay within knowledge domain
- **Confidence Scoring**: "I don't know" responses when quality is low

## 🏭 Production Features

- **Caching**: Semantic + exact match caching for performance
- **Monitoring**: Query latency, retrieval quality, error tracking
- **A/B Testing**: Compare RAG configurations
- **MLflow Integration**: Experiment tracking and model versioning
- **Rate Limiting**: API protection
- **Authentication**: Secure access control

## 🤝 Contributing

This is a learning project. Each module includes:
- Detailed code comments explaining RAG concepts
- Docstrings with examples
- Unit tests
- Performance benchmarks

## 📖 Additional Resources

- [RAG Concepts Guide](docs/concepts/rag_overview.md)
- [API Documentation](docs/api/README.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

## 📝 License

MIT License - Free for learning and commercial use

## 🙏 Acknowledgments

Built with: LangChain, LlamaIndex, ChromaDB, FAISS, Neo4j, OpenAI, Cohere, Streamlit, FastAPI, MLflow
# RAG-complete-setup
