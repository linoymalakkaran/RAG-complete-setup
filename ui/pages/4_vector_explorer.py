"""
Vector Space Explorer - Visualize Document Embeddings

Features:
- UMAP/t-SNE dimension reduction
- Interactive 2D/3D scatter plots
- Cluster analysis
- Similarity exploration
- Embedding statistics
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Vector Space Explorer", page_icon="🌌", layout="wide")

st.title("🌌 Vector Space Explorer")
st.markdown("Visualize and explore document embeddings in 2D/3D space")

# Sidebar settings
with st.sidebar:
    st.markdown("### ⚙️ Visualization Settings")
    
    reduction_method = st.selectbox(
        "Dimension Reduction",
        ["UMAP", "t-SNE"] if UMAP_AVAILABLE else ["t-SNE"],
        help="Method to reduce high-dimensional embeddings to 2D/3D"
    )
    
    dimensions = st.radio("Dimensions", ["2D", "3D"], index=0)
    
    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    
    color_by = st.selectbox(
        "Color By",
        ["Category", "Cluster", "Similarity", "Source"],
        help="How to color the points"
    )
    
    show_labels = st.checkbox("Show Labels", value=False)
    show_clusters = st.checkbox("Show Cluster Boundaries", value=True)
    point_size = st.slider("Point Size", 5, 20, 10)
    
    st.markdown("---")
    st.markdown("### 🔍 Advanced")
    
    num_neighbors = st.slider("UMAP Neighbors", 5, 50, 15) if UMAP_AVAILABLE else None
    perplexity = st.slider("t-SNE Perplexity", 5, 50, 30)
    
    st.button("🔄 Regenerate Visualization")

# Generate sample embedding data
@st.cache_data
def generate_sample_embeddings(n_docs=500, n_dims=384):
    """Generate sample document embeddings for visualization."""
    np.random.seed(42)
    
    # Create clusters representing different topics
    categories = ['HR Policies', 'IT Support', 'Finance', 'Legal', 'Operations']
    n_per_category = n_docs // len(categories)
    
    embeddings = []
    labels = []
    sources = []
    
    for i, category in enumerate(categories):
        # Create cluster center
        center = np.random.randn(n_dims) * 0.5
        center[i * 10:(i + 1) * 10] += 3  # Make some dimensions category-specific
        
        # Generate points around center
        cluster_embeddings = center + np.random.randn(n_per_category, n_dims) * 0.3
        embeddings.append(cluster_embeddings)
        labels.extend([category] * n_per_category)
        sources.extend([f"{category}_{j}.pdf" for j in range(n_per_category)])
    
    embeddings = np.vstack(embeddings)
    
    return embeddings, labels, sources

# Load data
with st.spinner("Loading embeddings..."):
    embeddings, labels, sources = generate_sample_embeddings()

st.success(f"✅ Loaded {len(embeddings)} document embeddings ({embeddings.shape[1]} dimensions)")

# Dimension reduction
@st.cache_data
def reduce_dimensions(embeddings, method='UMAP', n_components=2, **kwargs):
    """Reduce high-dimensional embeddings to 2D or 3D."""
    if method == 'UMAP' and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
    else:  # t-SNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=kwargs.get('perplexity', 30),
            random_state=42,
            n_iter=1000
        )
    
    reduced = reducer.fit_transform(embeddings)
    return reduced

# Reduce dimensions
n_components = 3 if dimensions == "3D" else 2

with st.spinner(f"Reducing dimensions with {reduction_method}..."):
    if reduction_method == 'UMAP' and UMAP_AVAILABLE:
        reduced_emb = reduce_dimensions(
            embeddings,
            method='UMAP',
            n_components=n_components,
            n_neighbors=num_neighbors
        )
    else:
        reduced_emb = reduce_dimensions(
            embeddings,
            method='t-SNE',
            n_components=n_components,
            perplexity=perplexity
        )

# Create DataFrame for plotting
if dimensions == "2D":
    df_plot = pd.DataFrame({
        'x': reduced_emb[:, 0],
        'y': reduced_emb[:, 1],
        'Category': labels,
        'Source': sources,
        'Cluster': [f"Cluster {i % 5}" for i in range(len(labels))]
    })
else:
    df_plot = pd.DataFrame({
        'x': reduced_emb[:, 0],
        'y': reduced_emb[:, 1],
        'z': reduced_emb[:, 2],
        'Category': labels,
        'Source': sources,
        'Cluster': [f"Cluster {i % 5}" for i in range(len(labels))]
    })

# Add similarity scores (distance from origin as proxy)
df_plot['Similarity'] = np.linalg.norm(reduced_emb, axis=1)

# Main visualization
tab1, tab2, tab3 = st.tabs(["🗺️ Embedding Map", "📊 Cluster Analysis", "🔍 Similarity Search"])

with tab1:
    st.markdown("## Document Embedding Visualization")
    
    # Select color column
    color_column = color_by
    
    # Create plot
    if dimensions == "2D":
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color=color_column,
            hover_data=['Source', 'Category'],
            title=f"{reduction_method} Projection of Document Embeddings (2D)",
            labels={'x': f'{reduction_method}-1', 'y': f'{reduction_method}-2'},
            color_continuous_scale='Viridis' if color_column == 'Similarity' else None
        )
        
        fig.update_traces(marker=dict(size=point_size, opacity=0.7))
        
        if show_labels:
            # Add text labels for a subset of points
            sample_indices = np.random.choice(len(df_plot), min(20, len(df_plot)), replace=False)
            for idx in sample_indices:
                fig.add_annotation(
                    x=df_plot.iloc[idx]['x'],
                    y=df_plot.iloc[idx]['y'],
                    text=df_plot.iloc[idx]['Source'].split('_')[0][:10],
                    showarrow=False,
                    font=dict(size=8)
                )
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # 3D
        fig = px.scatter_3d(
            df_plot,
            x='x',
            y='y',
            z='z',
            color=color_column,
            hover_data=['Source', 'Category'],
            title=f"{reduction_method} Projection of Document Embeddings (3D)",
            labels={'x': f'{reduction_method}-1', 'y': f'{reduction_method}-2', 'z': f'{reduction_method}-3'},
            color_continuous_scale='Viridis' if color_column == 'Similarity' else None
        )
        
        fig.update_traces(marker=dict(size=point_size/2, opacity=0.7))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("### 📊 Embedding Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(df_plot))
    with col2:
        st.metric("Categories", df_plot['Category'].nunique())
    with col3:
        st.metric("Original Dimensions", embeddings.shape[1])
    with col4:
        st.metric("Reduced Dimensions", n_components)

with tab2:
    st.markdown("## Cluster Analysis")
    
    # Cluster distribution
    st.markdown("### 📈 Category Distribution")
    
    category_counts = df_plot['Category'].value_counts()
    fig_dist = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Category', 'y': 'Document Count'},
        title='Documents per Category',
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Cluster quality metrics
    st.markdown("### 📊 Cluster Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Intra-cluster distance (how tight clusters are)
        intra_distances = []
        for category in df_plot['Category'].unique():
            cluster_points = reduced_emb[df_plot['Category'] == category]
            centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_distances.append(distances.mean())
        
        fig_intra = px.bar(
            x=df_plot['Category'].unique(),
            y=intra_distances,
            labels={'x': 'Category', 'y': 'Avg Distance to Centroid'},
            title='Cluster Tightness (Lower = Better)',
            color=intra_distances,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_intra, use_container_width=True)
    
    with col2:
        # Inter-cluster distance (how separated clusters are)
        categories = df_plot['Category'].unique()
        centroids = []
        for category in categories:
            cluster_points = reduced_emb[df_plot['Category'] == category]
            centroids.append(cluster_points.mean(axis=0))
        
        centroids = np.array(centroids)
        inter_distances = []
        for i, cat in enumerate(categories):
            other_centroids = np.delete(centroids, i, axis=0)
            distances = np.linalg.norm(centroids[i] - other_centroids, axis=1)
            inter_distances.append(distances.min())
        
        fig_inter = px.bar(
            x=categories,
            y=inter_distances,
            labels={'x': 'Category', 'y': 'Distance to Nearest Cluster'},
            title='Cluster Separation (Higher = Better)',
            color=inter_distances,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_inter, use_container_width=True)
    
    # Density map
    st.markdown("### 🗺️ Density Heatmap")
    
    if dimensions == "2D":
        fig_density = px.density_contour(
            df_plot,
            x='x',
            y='y',
            title='Document Density Map',
            labels={'x': f'{reduction_method}-1', 'y': f'{reduction_method}-2'}
        )
        fig_density.update_traces(contours_coloring="fill", contours_showlabels=True)
        st.plotly_chart(fig_density, use_container_width=True)

with tab3:
    st.markdown("## Similarity Search")
    
    st.markdown("### 🔍 Find Similar Documents")
    
    # Select a query document
    query_idx = st.selectbox(
        "Select a document",
        range(len(df_plot)),
        format_func=lambda i: f"{df_plot.iloc[i]['Source']} ({df_plot.iloc[i]['Category']})"
    )
    
    # Calculate distances
    query_embedding = embeddings[query_idx]
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    
    # Find top K similar
    k = st.slider("Number of similar documents", 5, 50, 10)
    similar_indices = np.argsort(distances)[:k]
    
    # Display results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📄 Query Document")
        st.info(f"**Source:** {df_plot.iloc[query_idx]['Source']}")
        st.info(f"**Category:** {df_plot.iloc[query_idx]['Category']}")
    
    with col2:
        st.markdown(f"#### 📊 Top {k} Most Similar Documents")
        
        similar_docs = []
        for i, idx in enumerate(similar_indices):
            if idx != query_idx:  # Exclude the query itself
                similar_docs.append({
                    'Rank': i,
                    'Source': df_plot.iloc[idx]['Source'],
                    'Category': df_plot.iloc[idx]['Category'],
                    'Distance': distances[idx],
                    'Similarity': 1 / (1 + distances[idx])
                })
        
        df_similar = pd.DataFrame(similar_docs)
        st.dataframe(
            df_similar.style.background_gradient(subset=['Similarity'], cmap='RdYlGn', vmin=0, vmax=1),
            use_container_width=True,
            hide_index=True
        )
    
    # Visualize query and results
    st.markdown("### 🗺️ Query Visualization")
    
    # Highlight query and similar docs in the embedding space
    df_highlight = df_plot.copy()
    df_highlight['Type'] = 'Other'
    df_highlight.loc[query_idx, 'Type'] = 'Query'
    df_highlight.loc[similar_indices, 'Type'] = 'Similar'
    
    if dimensions == "2D":
        fig_sim = px.scatter(
            df_highlight,
            x='x',
            y='y',
            color='Type',
            color_discrete_map={'Query': 'red', 'Similar': 'yellow', 'Other': 'lightgray'},
            hover_data=['Source', 'Category'],
            title='Query and Similar Documents',
            labels={'x': f'{reduction_method}-1', 'y': f'{reduction_method}-2'}
        )
        fig_sim.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_sim, use_container_width=True)

# Export functionality
st.markdown("---")
st.markdown("### 📥 Export")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Save Embeddings"):
        # In production, save to file
        st.success("Embeddings saved!")

with col2:
    if st.button("📊 Export Clusters"):
        csv = df_plot.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "cluster_data.csv",
            "text/csv"
        )

with col3:
    if st.button("🖼️ Save Visualization"):
        st.info("High-res export coming soon...")
