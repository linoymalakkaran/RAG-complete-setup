"""
Company Policy & Knowledge Assistant - Main Streamlit App

A comprehensive RAG learning platform demonstrating all major patterns.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging

logger = setup_logging("rag.ui")


# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=RAG+Assistant", use_column_width=True)
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Navigation")
        st.markdown("""
        - 📄 **Document Upload**: Add knowledge sources
        - 🔍 **Query Playground**: Test RAG patterns
        - 📊 **Pattern Comparison**: Compare approaches
        - 🗺️ **Vector Explorer**: Visualize embeddings
        - 🕸️ **Knowledge Graph**: Graph RAG viewer
        - 📈 **Evaluation**: Quality metrics
        - ⚙️ **Settings**: Configure system
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Learning Resources")
        st.markdown("""
        - [RAG Concepts Guide](docs/concepts/rag_overview.md)
        - [API Documentation](docs/api/README.md)
        - [Jupyter Notebooks](notebooks/)
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Current Status")
        
        # Mock metrics (in production, fetch from system)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", "0", "Add some!")
        with col2:
            st.metric("Queries Today", "0")
    
    # Main content
    st.markdown('<p class="main-header">🤖 Company Policy & Knowledge Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A Comprehensive RAG Learning Platform</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the **RAG Knowledge Assistant** - a production-ready system that demonstrates 
    all major Retrieval Augmented Generation (RAG) patterns and techniques.
    
    This platform helps you:
    - 📖 Understand different RAG architectures through hands-on experimentation
    - 🔬 Compare chunking strategies, embedding models, and retrieval methods
    - 📊 Evaluate system quality with comprehensive metrics
    - 🚀 Build production-ready RAG applications
    """)
    
    # Feature showcase
    st.markdown("## 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🔄 6 RAG Patterns</h3>
            <ul>
                <li>Basic RAG</li>
                <li>Self-RAG</li>
                <li>Corrective RAG</li>
                <li>Agentic RAG</li>
                <li>Graph RAG</li>
                <li>Multimodal RAG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>📝 Document Processing</h3>
            <ul>
                <li>PDF, Word, Images (OCR)</li>
                <li>4 Chunking Strategies</li>
                <li>Chunk Size Optimizer</li>
                <li>Metadata Extraction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🔍 Advanced Retrieval</h3>
            <ul>
                <li>Hybrid Search (Dense + Sparse)</li>
                <li>Multiple Embedding Models</li>
                <li>Query Enhancement</li>
                <li>Reranking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown("## 🚀 Quick Start")
    
    st.markdown("""
    ### Getting Started in 3 Steps:
    
    1. **📄 Upload Documents** - Go to the Document Upload page and add your knowledge sources (PDFs, Word docs, images)
    
    2. **🔍 Test Queries** - Visit the Query Playground to ask questions and see different RAG patterns in action
    
    3. **📊 Evaluate** - Use the Evaluation Dashboard to measure retrieval quality and answer accuracy
    
    ### Learning Path:
    
    For a structured learning experience:
    - Start with the **Jupyter notebooks** (recommended for deep understanding)
    - Experiment in the **Query Playground** (interactive testing)
    - Compare approaches in **Pattern Comparison** (side-by-side evaluation)
    - Dive into **Settings** to understand configuration options
    """)
    
    # Architecture overview
    with st.expander("📐 System Architecture", expanded=False):
        st.markdown("""
        ```
        User Query
            ↓
        Query Enhancement (multi-query, HyDE, expansion)
            ↓
        Hybrid Retrieval (Vector Search + BM25)
            ↓
        Reranking (Cross-encoder)
            ↓
        RAG Pattern (Basic/Self/CRAG/Agentic/Graph/Multimodal)
            ↓
        Context Management (Memory, Summarization)
            ↓
        LLM Generation (GPT-4/Local)
            ↓
        Guardrails (PII Redaction, Hallucination Detection)
            ↓
        Response + Source Citations
        ```
        
        **Tech Stack:**
        - **Vector DBs**: ChromaDB, FAISS
        - **Graph DB**: Neo4j
        - **Embeddings**: OpenAI, Cohere, Sentence Transformers
        - **LLMs**: GPT-4, Claude, Local (Ollama)
        - **Frameworks**: LangChain, LlamaIndex, CrewAI
        - **Evaluation**: RAGAS, Custom Metrics
        """)
    
    # RAG patterns explained
    with st.expander("🧠 RAG Patterns Explained", expanded=False):
        st.markdown("""
        ### 1. Basic RAG
        Simple retrieve-and-generate. Best for: Getting started, simple Q&A
        
        ### 2. Self-RAG
        Adds self-reflection: "Do I need to retrieve?", "Is my answer good?"
        Best for: Reducing unnecessary retrievals, improving answer quality
        
        ### 3. Corrective RAG (CRAG)
        Evaluates retrieval quality, falls back to web search if needed.
        Best for: Handling out-of-domain queries, ensuring completeness
        
        ### 4. Agentic RAG
        Agent decides autonomously: search, clarify, or combine sources.
        Best for: Complex queries requiring multiple steps
        
        ### 5. Graph RAG
        Builds knowledge graph, retrieves via relationship traversal.
        Best for: Relationship-heavy data (org charts, dependencies)
        
        ### 6. Multimodal RAG
        Handles images, diagrams, and videos alongside text.
        Best for: Technical docs with diagrams, product catalogs
        """)
    
    # Tips
    st.markdown("## 💡 Pro Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 For Best Results:**
        - Use semantic chunking for topic-heavy documents
        - Enable hybrid search for keyword-sensitive queries
        - Try Graph RAG for organizational/relationship data
        - Use parent-document chunking for long documents
        """)
    
    with col2:
        st.warning("""
        **⚠️ Common Pitfalls:**
        - Chunks too small → Lost context
        - Chunks too large → Imprecise retrieval
        - No reranking → Lower quality results
        - Ignoring metadata → Missed filtering opportunities
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using LangChain, ChromaDB, and Streamlit</p>
        <p>📖 <a href='README.md'>Documentation</a> | 💻 <a href='https://github.com'>GitHub</a> | 📧 <a href='mailto:support@example.com'>Support</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
