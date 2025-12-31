# 🚀 Quick Start Guide

Welcome to the **Company Policy & Knowledge Assistant** - your comprehensive RAG learning platform!

## Installation

### 1. Create Virtual Environment

```bash
cd rag
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
copy .env.example .env  # Windows
# or
cp .env.example .env    # Mac/Linux

# Edit .env and add your API keys:
# - OPENAI_API_KEY (required for embeddings and LLM)
# - COHERE_API_KEY (optional, for Cohere embeddings)
# - ANTHROPIC_API_KEY (optional, for Claude)
```

## Running the Application

### Option 1: Streamlit UI (Recommended for Beginners)

```bash
streamlit run ui/app.py
```

Then open your browser to http://localhost:8501

### Option 2: Jupyter Notebooks (Recommended for Learning)

```bash
jupyter notebook notebooks/
```

Start with `01_getting_started.ipynb`

### Option 3: FastAPI Server (For Production)

```bash
uvicorn src.api.main:app --reload
```

API will be available at http://localhost:8000

## Quick Tutorial

### 1. Process Your First Document

```python
from src.ingestion.loaders.document_loaders import load_document
from src.ingestion.chunking.chunking_strategies import chunk_document

# Load document
doc = load_document("data/sample_documents/hr_policies/vacation_policy.pdf")

# Chunk it
chunks = chunk_document(
    doc['content'],
    strategy="recursive",
    chunk_size=1000,
    metadata={'source': 'hr_policy'}
)

print(f"Created {len(chunks)} chunks")
```

### 2. Create Embeddings

```python
from src.embeddings.providers.embedding_providers import EmbeddingFactory

# Initialize embedder
embedder = EmbeddingFactory.create("openai")

# Embed chunks
embeddings = embedder.embed_batch([c.content for c in chunks])
```

### 3. Store in Vector Database

```python
from src.vectordb.chromadb_client import ChromaDBClient

# Initialize ChromaDB
vectordb = ChromaDBClient(
    collection_name="my_knowledge_base",
    embedder=embedder
)

# Add chunks
vectordb.add_documents(
    documents=[c.content for c in chunks],
    metadatas=[c.metadata for c in chunks],
    ids=[c.chunk_id for c in chunks]
)
```

### 4. Query with RAG

```python
from src.rag_patterns.basic_rag import create_basic_rag

# Create RAG system
rag = create_basic_rag(vectordb)

# Ask a question
result = rag.query("What is the vacation policy?")

print("Answer:", result['answer'])
print("Sources:", len(result['sources']))
```

## Learning Path

### Week 1: Foundations
1. ✅ Run `01_getting_started.ipynb`
2. ✅ Explore Document Upload page in UI
3. ✅ Read RAG Concepts Guide

### Week 2: Advanced Techniques
1. ✅ Try different chunking strategies (`02_chunking_strategies.ipynb`)
2. ✅ Compare embedding models (`03_embedding_comparison.ipynb`)
3. ✅ Experiment with hybrid search

### Week 3: RAG Patterns
1. ✅ Study all 6 RAG patterns (`04_rag_patterns.ipynb`)
2. ✅ Use Pattern Comparison page
3. ✅ Build your own pattern

### Week 4: Production
1. ✅ Learn evaluation metrics (`05_evaluation_metrics.ipynb`)
2. ✅ Set up monitoring
3. ✅ Deploy with Docker

## Project Structure

```
rag/
├── src/                    # Source code
│   ├── ingestion/         # Document loading & chunking
│   ├── embeddings/        # Embedding providers
│   ├── vectordb/          # Vector databases
│   ├── rag_patterns/      # RAG implementations
│   └── api/               # FastAPI server
├── ui/                    # Streamlit interface
├── notebooks/             # Learning notebooks
├── data/                  # Sample documents
├── config/                # Configuration files
└── tests/                 # Unit tests
```

## Common Issues & Solutions

### Issue: "OPENAI_API_KEY not found"
**Solution**: Make sure you've created `.env` file and added your API key

### Issue: "Module not found"
**Solution**: Ensure you're in the virtual environment and ran `pip install -r requirements.txt`

### Issue: ChromaDB connection error
**Solution**: Check that the `chromadb_data` directory exists and has write permissions

### Issue: Out of memory when embedding
**Solution**: Process documents in smaller batches or use a local embedding model

## Docker Setup (Optional)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services will be available at:
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8080
- ChromaDB: http://localhost:8000
- Neo4j: http://localhost:7474
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000

## Next Steps

1. **📓 Work through notebooks**: Start with `01_getting_started.ipynb`
2. **🖥️ Explore UI**: Use Query Playground to test queries
3. **📚 Read docs**: Check out `docs/concepts/rag_overview.md`
4. **🔬 Experiment**: Try different configurations in Settings page
5. **📊 Evaluate**: Use Evaluation Dashboard to measure quality

## Getting Help

- 📖 Documentation: `docs/`
- 💬 Issues: GitHub Issues
- 📧 Email: support@example.com

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

Happy learning! 🚀
