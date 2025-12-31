"""
Document Upload & Processing Page

Allows users to:
- Upload documents (PDF, Word, images, etc.)
- Preview chunking strategies
- Process and index documents
- View processing status
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.loaders.document_loaders import LoaderFactory, load_document
from src.ingestion.chunking.chunking_strategies import ChunkerFactory, chunk_document
from src.ingestion.chunking.optimizer import ChunkOptimizer

st.set_page_config(page_title="Document Upload", page_icon="📄", layout="wide")

st.title("📄 Document Upload & Processing")
st.markdown("Upload and process documents to build your knowledge base.")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Documents",
    "✂️ Chunking Preview",
    "⚙️ Optimize Settings",
    "📊 Processing Status"
])

with tab1:
    st.markdown("### Upload Your Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'png', 'jpg', 'jpeg'],
            help="Supported: PDF, Word, Text, Markdown, Images"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            for file in uploaded_files:
                st.markdown(f"- **{file.name}** ({file.size / 1024:.1f} KB)")
    
    with col2:
        st.markdown("#### Processing Options")
        
        use_ocr = st.checkbox("Enable OCR for images/scanned PDFs", value=True)
        extract_metadata = st.checkbox("Extract metadata", value=True)
        
        st.markdown("#### Chunking Strategy")
        chunking_strategy = st.selectbox(
            "Select strategy",
            ["recursive", "fixed", "semantic", "parent_document"],
            help="Recursive is recommended for most use cases"
        )
        
        if chunking_strategy == "recursive":
            chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk overlap", 50, 500, 200, 50)
        elif chunking_strategy == "fixed":
            chunk_size = st.slider("Chunk size", 256, 1024, 512, 64)
            chunk_overlap = st.slider("Chunk overlap", 25, 200, 50, 25)
        
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                            tmp.write(file.getbuffer())
                            tmp_path = tmp.name
                        
                        try:
                            # Load document
                            doc_data = load_document(tmp_path)
                            
                            # Chunk document
                            chunks = chunk_document(
                                doc_data['content'],
                                strategy=chunking_strategy,
                                chunk_size=chunk_size if chunking_strategy in ['recursive', 'fixed'] else None,
                                chunk_overlap=chunk_overlap if chunking_strategy in ['recursive', 'fixed'] else None,
                                metadata={'filename': file.name, **doc_data['metadata']}
                            )
                            
                            # TODO: Add to vector database
                            
                            st.success(f"✅ Processed {file.name}: {len(chunks)} chunks created")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
                        
                        finally:
                            os.unlink(tmp_path)
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.text("✅ All documents processed!")
            else:
                st.warning("Please select files to upload")

with tab2:
    st.markdown("### Chunking Strategy Preview")
    st.markdown("See how different strategies chunk your text.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        sample_text = st.text_area(
            "Enter sample text",
            value="""Company Policy: Vacation Time

All full-time employees are entitled to paid vacation time. The amount of vacation time varies based on length of service:

- 0-2 years: 10 days per year
- 2-5 years: 15 days per year  
- 5+ years: 20 days per year

Vacation requests must be submitted at least 2 weeks in advance through the HR portal. Unused vacation days can be carried over to the next year, up to a maximum of 5 days.

For questions about vacation policy, please contact HR at hr@company.com.""",
            height=300
        )
        
        preview_strategy = st.selectbox(
            "Strategy",
            ["recursive", "fixed", "semantic", "parent_document"],
            key="preview_strategy"
        )
        
        if preview_strategy in ["recursive", "fixed"]:
            preview_chunk_size = st.slider("Chunk size", 100, 500, 200, 50, key="preview_size")
            preview_overlap = st.slider("Overlap", 0, 100, 20, 10, key="preview_overlap")
        
        if st.button("Preview Chunks"):
            with st.spinner("Chunking..."):
                kwargs = {}
                if preview_strategy in ["recursive", "fixed"]:
                    kwargs = {
                        'chunk_size': preview_chunk_size,
                        'chunk_overlap': preview_overlap
                    }
                
                chunks = chunk_document(sample_text, strategy=preview_strategy, **kwargs)
                
                st.session_state['preview_chunks'] = chunks
    
    with col2:
        if 'preview_chunks' in st.session_state:
            chunks = st.session_state['preview_chunks']
            
            st.markdown(f"**Result: {len(chunks)} chunks**")
            
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk.content)} chars)"):
                    st.text(chunk.content)
                    if chunk.metadata:
                        st.json(chunk.metadata, expanded=False)

with tab3:
    st.markdown("### 🔧 Chunk Size Optimizer")
    st.markdown("Find optimal chunking settings for your documents.")
    
    st.info("""
    The optimizer tests different chunking configurations and recommends the best settings based on:
    - Chunk size consistency
    - Content coverage
    - Processing speed
    """)
    
    optimize_use_case = st.selectbox(
        "Use case",
        ["general", "precise", "context", "speed"],
        help="general=balanced, precise=small chunks, context=large chunks, speed=fast processing"
    )
    
    optimize_sample = st.text_area(
        "Sample documents (one per line)",
        value="""Sample document 1 about vacation policy...""",
        height=200
    )
    
    if st.button("🔍 Run Optimization"):
        with st.spinner("Testing configurations..."):
            documents = [doc.strip() for doc in optimize_sample.split('\n') if doc.strip()]
            
            if documents:
                optimizer = ChunkOptimizer()
                recommendation = optimizer.recommend(documents, use_case=optimize_use_case)
                
                st.success("✅ Optimization complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recommended Strategy", recommendation['strategy'])
                with col2:
                    st.metric("Chunk Size", recommendation['chunk_size'])
                with col3:
                    st.metric("Overlap", recommendation['chunk_overlap'])
                
                st.markdown("#### Explanation")
                st.info(recommendation['reasoning'])
                
                st.markdown("#### Expected Results")
                st.json(recommendation, expanded=True)
            else:
                st.warning("Please provide sample documents")

with tab4:
    st.markdown("### 📊 Processing Status")
    
    # Mock status (in production, fetch from database)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", "0", "Add some!")
    with col2:
        st.metric("Total Chunks", "0")
    with col3:
        st.metric("Indexed", "0")
    with col4:
        st.metric("Storage Used", "0 MB")
    
    st.markdown("#### Recent Uploads")
    st.dataframe({
        'Filename': [],
        'Upload Date': [],
        'Chunks': [],
        'Status': []
    }, use_container_width=True)
    
    st.markdown("#### Document Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By File Type**")
        st.bar_chart({'PDF': 0, 'Word': 0, 'Text': 0, 'Image': 0})
    
    with col2:
        st.markdown("**By Category**")
        st.bar_chart({'HR Policies': 0, 'Technical Docs': 0, 'Training': 0})
