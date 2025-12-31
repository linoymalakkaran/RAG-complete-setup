"""
Evaluation Dashboard - Comprehensive RAG Metrics Visualization

Displays all 19 evaluation metrics across 3 categories:
- RAGAS Metrics (5): Faithfulness, Relevancy, Precision, Recall, Context Relevancy
- Retrieval Metrics (8): Precision@K, Recall@K, F1, MRR, MAP, NDCG, Hit Rate, Coverage
- Response Metrics (6): BLEU, ROUGE-1/2/L, BERT Score, Exact Match
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Evaluation Dashboard")
st.markdown("Comprehensive RAG system evaluation across 19 metrics")

# Sidebar filters
with st.sidebar:
    st.markdown("### 🔍 Filters")
    
    selected_pattern = st.multiselect(
        "RAG Patterns",
        ["Basic RAG", "Self-RAG", "Corrective RAG", "Agentic RAG", "Graph RAG", "Multimodal RAG"],
        default=["Basic RAG", "Self-RAG"]
    )
    
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Display Settings")
    
    show_details = st.checkbox("Show Detailed Breakdown", value=True)
    show_trends = st.checkbox("Show Trends", value=True)
    normalize_scores = st.checkbox("Normalize Scores (0-1)", value=True)

# Generate sample data (in production, load from database/logs)
@st.cache_data
def generate_sample_data():
    """Generate sample evaluation data for demonstration."""
    patterns = ["Basic RAG", "Self-RAG", "Corrective RAG", "Agentic RAG", "Graph RAG", "Multimodal RAG"]
    
    data = {
        "pattern": patterns,
        # RAGAS Metrics
        "faithfulness": [0.72, 0.85, 0.88, 0.82, 0.86, 0.79],
        "answer_relevancy": [0.68, 0.81, 0.85, 0.88, 0.83, 0.82],
        "context_precision": [0.65, 0.78, 0.83, 0.76, 0.89, 0.74],
        "context_recall": [0.70, 0.82, 0.86, 0.80, 0.91, 0.77],
        "context_relevancy": [0.67, 0.80, 0.84, 0.78, 0.87, 0.75],
        # Retrieval Metrics
        "precision_k": [0.64, 0.76, 0.81, 0.75, 0.88, 0.72],
        "recall_k": [0.68, 0.79, 0.84, 0.78, 0.90, 0.74],
        "f1_score": [0.66, 0.77, 0.82, 0.76, 0.89, 0.73],
        "mrr": [0.71, 0.83, 0.87, 0.81, 0.91, 0.78],
        "map": [0.69, 0.81, 0.86, 0.79, 0.90, 0.76],
        "ndcg": [0.70, 0.82, 0.87, 0.80, 0.91, 0.77],
        "hit_rate": [0.85, 0.92, 0.95, 0.90, 0.96, 0.88],
        "coverage": [0.62, 0.74, 0.79, 0.73, 0.85, 0.70],
        # Response Metrics
        "bleu": [0.42, 0.58, 0.64, 0.61, 0.59, 0.55],
        "rouge1": [0.56, 0.68, 0.73, 0.70, 0.69, 0.65],
        "rouge2": [0.38, 0.52, 0.58, 0.55, 0.54, 0.50],
        "rougeL": [0.52, 0.64, 0.70, 0.67, 0.66, 0.62],
        "bert_score": [0.74, 0.84, 0.88, 0.86, 0.85, 0.83],
        "exact_match": [0.18, 0.32, 0.38, 0.35, 0.33, 0.28],
    }
    
    return pd.DataFrame(data)

df = generate_sample_data()

# Filter by selected patterns
if selected_pattern:
    df_filtered = df[df['pattern'].isin(selected_pattern)]
else:
    df_filtered = df

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🎯 RAGAS Metrics",
    "🔍 Retrieval Metrics",
    "📝 Response Metrics"
])

with tab1:
    st.markdown("## Overall Performance Summary")
    
    # Calculate aggregate scores
    ragas_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'context_relevancy']
    retrieval_cols = ['precision_k', 'recall_k', 'f1_score', 'mrr', 'map', 'ndcg', 'hit_rate', 'coverage']
    response_cols = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'exact_match']
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = df_filtered[ragas_cols + retrieval_cols + response_cols].mean().mean()
        st.metric("Overall Score", f"{overall_score:.3f}", delta="+0.05")
    
    with col2:
        ragas_score = df_filtered[ragas_cols].mean().mean()
        st.metric("RAGAS Score", f"{ragas_score:.3f}", delta="+0.08")
    
    with col3:
        retrieval_score = df_filtered[retrieval_cols].mean().mean()
        st.metric("Retrieval Score", f"{retrieval_score:.3f}", delta="+0.06")
    
    with col4:
        response_score = df_filtered[response_cols].mean().mean()
        st.metric("Response Score", f"{response_score:.3f}", delta="+0.04")
    
    st.markdown("---")
    
    # Radar chart for pattern comparison
    st.markdown("### 📈 Pattern Comparison - Overall Performance")
    
    if len(df_filtered) > 0:
        # Calculate average scores per category
        pattern_scores = []
        for _, row in df_filtered.iterrows():
            pattern_scores.append({
                'Pattern': row['pattern'],
                'RAGAS': row[ragas_cols].mean(),
                'Retrieval': row[retrieval_cols].mean(),
                'Response': row[response_cols].mean()
            })
        
        # Create radar chart
        fig = go.Figure()
        
        for score_data in pattern_scores:
            fig.add_trace(go.Scatterpolar(
                r=[score_data['RAGAS'], score_data['Retrieval'], score_data['Response']],
                theta=['RAGAS Metrics', 'Retrieval Metrics', 'Response Metrics'],
                fill='toself',
                name=score_data['Pattern']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance by pattern - bar chart
    st.markdown("### 📊 Detailed Scores by Pattern")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RAGAS metrics
        fig_ragas = px.bar(
            df_filtered,
            x='pattern',
            y=ragas_cols,
            title="RAGAS Metrics by Pattern",
            barmode='group',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        fig_ragas.update_layout(height=400)
        st.plotly_chart(fig_ragas, use_container_width=True)
    
    with col2:
        # Retrieval metrics
        fig_retrieval = px.bar(
            df_filtered,
            x='pattern',
            y=['precision_k', 'recall_k', 'f1_score', 'mrr'],
            title="Key Retrieval Metrics by Pattern",
            barmode='group',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        fig_retrieval.update_layout(height=400)
        st.plotly_chart(fig_retrieval, use_container_width=True)

with tab2:
    st.markdown("## 🎯 RAGAS Metrics (5 Metrics)")
    st.markdown("""
    **RAGAS** (Retrieval Augmented Generation Assessment) evaluates RAG systems across 5 dimensions:
    - **Faithfulness**: Answer grounded in retrieved context
    - **Answer Relevancy**: Answer addresses the question
    - **Context Precision**: Relevant chunks ranked highly
    - **Context Recall**: All necessary information retrieved
    - **Context Relevancy**: Retrieved chunks are relevant
    """)
    
    # RAGAS metrics table
    st.markdown("### 📋 RAGAS Scores")
    ragas_df = df_filtered[['pattern'] + ragas_cols].set_index('pattern')
    st.dataframe(
        ragas_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
        use_container_width=True
    )
    
    # Individual metric analysis
    st.markdown("### 📊 Metric Breakdown")
    
    selected_metric = st.selectbox(
        "Select RAGAS Metric",
        ragas_cols,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart for selected metric
        fig = px.bar(
            df_filtered,
            x='pattern',
            y=selected_metric,
            title=f"{selected_metric.replace('_', ' ').title()} by Pattern",
            color=selected_metric,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stats for selected metric
        st.markdown(f"**{selected_metric.replace('_', ' ').title()} Statistics**")
        metric_values = df_filtered[selected_metric]
        st.metric("Mean", f"{metric_values.mean():.3f}")
        st.metric("Std Dev", f"{metric_values.std():.3f}")
        st.metric("Min", f"{metric_values.min():.3f}")
        st.metric("Max", f"{metric_values.max():.3f}")
        
        # Best performing pattern
        best_pattern = df_filtered.loc[df_filtered[selected_metric].idxmax(), 'pattern']
        st.success(f"🏆 Best: {best_pattern}")
    
    if show_trends:
        st.markdown("### 📈 Trend Analysis")
        # Generate time series data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        trend_data = []
        for date in dates:
            for pattern in selected_pattern:
                base_score = df_filtered[df_filtered['pattern'] == pattern][selected_metric].values[0]
                noise = np.random.normal(0, 0.02)
                trend = 0.001 * (date - dates[0]).days  # Slight upward trend
                trend_data.append({
                    'Date': date,
                    'Pattern': pattern,
                    'Score': np.clip(base_score + noise + trend, 0, 1)
                })
        
        trend_df = pd.DataFrame(trend_data)
        fig_trend = px.line(
            trend_df,
            x='Date',
            y='Score',
            color='Pattern',
            title=f"{selected_metric.replace('_', ' ').title()} Trend (Last 30 Days)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    st.markdown("## 🔍 Retrieval Metrics (8 Metrics)")
    st.markdown("""
    **Retrieval metrics** evaluate how well the system finds relevant documents:
    - **Precision@K**: Fraction of retrieved docs that are relevant
    - **Recall@K**: Fraction of relevant docs that are retrieved
    - **F1 Score**: Harmonic mean of precision and recall
    - **MRR**: Mean Reciprocal Rank of first relevant doc
    - **MAP**: Mean Average Precision across queries
    - **NDCG**: Normalized Discounted Cumulative Gain
    - **Hit Rate**: Queries with at least one relevant doc
    - **Coverage**: Overall relevant document retrieval
    """)
    
    # Retrieval metrics table
    st.markdown("### 📋 Retrieval Scores")
    retrieval_df = df_filtered[['pattern'] + retrieval_cols].set_index('pattern')
    st.dataframe(
        retrieval_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
        use_container_width=True
    )
    
    # Heatmap
    st.markdown("### 🔥 Heatmap View")
    fig_heatmap = px.imshow(
        retrieval_df.T,
        labels=dict(x="Pattern", y="Metric", color="Score"),
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Top 3 patterns by retrieval quality
    st.markdown("### 🏆 Top Performers")
    avg_retrieval = df_filtered[retrieval_cols].mean(axis=1)
    top3 = df_filtered.iloc[avg_retrieval.nlargest(3).index]
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.metric(
                f"#{i+1} {row['pattern']}",
                f"{avg_retrieval.iloc[avg_retrieval.nlargest(3).index[i]]:.3f}"
            )

with tab4:
    st.markdown("## 📝 Response Metrics (6 Metrics)")
    st.markdown("""
    **Response metrics** evaluate the quality of generated answers:
    - **BLEU**: N-gram overlap with reference
    - **ROUGE-1/2/L**: Unigram/bigram/longest common subsequence overlap
    - **BERT Score**: Semantic similarity using embeddings
    - **Exact Match**: String equality rate
    """)
    
    # Response metrics table
    st.markdown("### 📋 Response Quality Scores")
    response_df = df_filtered[['pattern'] + response_cols].set_index('pattern')
    st.dataframe(
        response_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
        use_container_width=True
    )
    
    # Grouped bar chart
    st.markdown("### 📊 Metric Comparison")
    fig_response = px.bar(
        df_filtered,
        x='pattern',
        y=response_cols,
        title="All Response Metrics by Pattern",
        barmode='group',
        labels={'value': 'Score', 'variable': 'Metric'}
    )
    fig_response.update_layout(height=500)
    st.plotly_chart(fig_response, use_container_width=True)
    
    # Correlation analysis
    if show_details:
        st.markdown("### 🔗 Metric Correlations")
        corr_matrix = df_filtered[response_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

# Export functionality
st.markdown("---")
st.markdown("### 📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export to CSV"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "evaluation_metrics.csv",
            "text/csv"
        )

with col2:
    if st.button("📈 Generate Report"):
        st.info("PDF report generation coming soon...")

with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
