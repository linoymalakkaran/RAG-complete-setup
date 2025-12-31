"""
Query Playground - Test RAG patterns interactively

Features:
- Select and test different RAG patterns
- See retrieved chunks with scores
- Compare retrieval methods
- View token usage and latency
"""

import streamlit as st
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Query Playground", page_icon="🔍", layout="wide")

st.title("🔍 Query Playground")
st.markdown("Test different RAG patterns and see how they perform.")

# Sidebar settings
with st.sidebar:
    st.markdown("### ⚙️ Query Settings")
    
    selected_pattern = st.selectbox(
        "RAG Pattern",
        ["Basic RAG", "Self-RAG", "Corrective RAG (CRAG)", "Agentic RAG", "Graph RAG", "Multimodal RAG"],
        help="Select which RAG pattern to use"
    )
    
    # Show pattern description
    pattern_descriptions = {
        "Basic RAG": "Simple retrieve and generate - good baseline",
        "Self-RAG": "Checks if retrieval is needed and validates answers",
        "Corrective RAG (CRAG)": "✨ Evaluates retrieval quality, uses web search if needed",
        "Agentic RAG": "Multi-step reasoning with autonomous decisions",
        "Graph RAG": "Uses knowledge graph for relationship-aware retrieval",
        "Multimodal RAG": "Handles text + image queries"
    }
    st.caption(pattern_descriptions.get(selected_pattern, ""))
    
    st.markdown("---")
    st.markdown("### 🔧 Retrieval Settings")
    
    top_k = st.slider("Top K Results", 1, 20, 5)
    use_hybrid = st.checkbox("Enable Hybrid Search", value=True)
    use_reranking = st.checkbox("Enable Reranking", value=True)
    
    if use_hybrid:
        dense_weight = st.slider("Dense Weight", 0.0, 1.0, 0.7, 0.1)
        sparse_weight = 1.0 - dense_weight
        st.caption(f"Sparse Weight: {sparse_weight:.1f}")
    
    st.markdown("---")
    st.markdown("### 🎯 Generation Settings")
    
    model = st.selectbox("LLM Model", ["gpt-4-turbo-preview", "gpt-3.5-turbo", "local-llama2"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Ask a Question")
    
    # Sample questions
    st.markdown("**Try these sample questions:**")
    sample_questions = [
        "What is the company vacation policy?",
        "How do I submit an expense report?",
        "What are the remote work guidelines?",
        "Who should I contact for IT support?",
        "What is the performance review process?"
    ]
    
    selected_sample = st.selectbox("Sample questions", [""] + sample_questions, label_visibility="collapsed")
    
    query = st.text_area(
        "Your question",
        value=selected_sample if selected_sample else "",
        height=100,
        placeholder="Type your question here..."
    )
    
    # Metadata filters
    with st.expander("🔍 Advanced Filters", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_category = st.multiselect("Category", ["HR", "IT", "Finance", "Legal"])
        with col_f2:
            filter_date = st.date_input("Documents after", value=None)
    
    if st.button("🚀 Ask Question", type="primary", use_container_width=True):
        if query:
            with st.spinner(f"Processing with {selected_pattern}..."):
                # Simulate processing (in production, call actual RAG system)
                time.sleep(1.5)
                
                # Mock response
                st.success("✅ Answer generated!")
                
                st.markdown("### 💡 Answer")
                st.markdown("""
                Based on the company policy document, employees are entitled to paid vacation time 
                based on their length of service:
                - 0-2 years: 10 days per year
                - 2-5 years: 15 days per year
                - 5+ years: 20 days per year
                
                Requests must be submitted at least 2 weeks in advance through the HR portal.
                """)
                
                # Show retrieved sources
                with st.expander("📚 Retrieved Sources", expanded=True):
                    for i in range(3):
                        st.markdown(f"**Source {i+1}** (Score: {0.85 - i*0.05:.2f})")
                        st.markdown(f"> Sample content from document {i+1}...")
                        st.caption(f"Metadata: HR Policy Doc, Page {i+1}")
                        st.markdown("---")
                
                # Show metadata
                with st.expander("📊 Query Metadata", expanded=False):
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Latency", "1.5s")
                    with col_m2:
                        st.metric("Tokens Used", "450")
                    with col_m3:
                        st.metric("Confidence", "0.87")
        else:
            st.warning("Please enter a question")

with col2:
    st.markdown("### 📊 Pattern Info")
    
    # Pattern descriptions
    pattern_info = {
        "Basic RAG": {
            "description": "Simple retrieve and generate",
            "best_for": "General Q&A, getting started",
            "pros": ["Simple", "Fast", "Easy to debug"],
            "cons": ["No quality checks", "May retrieve unnecessarily"]
        },
        "Self-RAG": {
            "description": "Self-reflection on retrieval need and quality",
            "best_for": "When retrieval isn't always needed",
            "pros": ["Reduces costs", "Better quality control", "Self-correcting"],
            "cons": ["Slightly slower", "More complex"]
        },
        "Corrective RAG": {
            "description": "Evaluates retrieval, falls back to web search",
            "best_for": "Handling incomplete knowledge",
            "pros": ["Handles missing info", "Web fallback", "Quality evaluation"],
            "cons": ["Web search costs", "Slower"]
        },
        "Agentic RAG": {
            "description": "Autonomous agent decides actions",
            "best_for": "Complex multi-step queries",
            "pros": ["Flexible", "Handles complexity", "Autonomous"],
            "cons": ["Expensive", "Less predictable"]
        },
        "Graph RAG": {
            "description": "Uses knowledge graph for retrieval",
            "best_for": "Relationship-heavy data",
            "pros": ["Relationship-aware", "Better for structured data"],
            "cons": ["Setup overhead", "Graph maintenance"]
        },
        "Multimodal RAG": {
            "description": "Handles images and videos",
            "best_for": "Visual content queries",
            "pros": ["Image search", "Diagram understanding"],
            "cons": ["Expensive", "Complex processing"]
        }
    }
    
    info = pattern_info[selected_pattern]
    
    st.markdown(f"**Description:** {info['description']}")
    st.markdown(f"**Best for:** {info['best_for']}")
    
    st.markdown("**Pros:**")
    for pro in info['pros']:
        st.markdown(f"✅ {pro}")
    
    st.markdown("**Cons:**")
    for con in info['cons']:
        st.markdown(f"⚠️ {con}")
    
    st.markdown("---")
    st.markdown("### 🎓 Learn More")
    st.markdown(f"[Read about {selected_pattern}](docs/concepts/{selected_pattern.lower().replace(' ', '_')}.md)")

# Query history
st.markdown("---")
st.markdown("### 📜 Recent Queries")

# Mock history
history_data = {
    'Query': ["What is vacation policy?", "How to submit expenses?"],
    'Pattern': ["Basic RAG", "Self-RAG"],
    'Latency': ["1.2s", "1.8s"],
    'Tokens': [420, 550],
    'Confidence': [0.89, 0.91]
}

st.dataframe(history_data, use_container_width=True)

# Tips
st.info("""
💡 **Tips for better results:**
- Be specific in your questions
- Use natural language
- Try different RAG patterns for comparison
- Enable hybrid search for keyword-heavy queries
- Use reranking for better precision
""")
