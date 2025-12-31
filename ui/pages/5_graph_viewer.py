"""
Knowledge Graph Viewer - Interactive Graph Visualization

Features:
- Interactive graph visualization
- Entity and relationship browsing
- Graph statistics
- Subgraph exploration
- Community detection
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Knowledge Graph Viewer", page_icon="🕸️", layout="wide")

st.title("🕸️ Knowledge Graph Viewer")
st.markdown("Explore the knowledge graph built from your documents")

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Graph Settings")
    
    layout_algorithm = st.selectbox(
        "Layout Algorithm",
        ["Spring", "Circular", "Kamada-Kawai", "Spectral"],
        help="How to arrange nodes in the visualization"
    )
    
    node_size_by = st.selectbox(
        "Node Size By",
        ["Degree", "Betweenness", "Pagerank", "Uniform"],
        help="How to size nodes"
    )
    
    st.markdown("---")
    st.markdown("### 🎨 Display")
    
    show_labels = st.checkbox("Show Labels", value=True)
    show_edge_labels = st.checkbox("Show Edge Labels", value=False)
    max_nodes = st.slider("Max Nodes to Display", 10, 200, 50)
    
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    
    entity_types = st.multiselect(
        "Entity Types",
        ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PRODUCT"],
        default=["PERSON", "ORGANIZATION", "CONCEPT"]
    )
    
    min_degree = st.slider("Minimum Connections", 0, 10, 1)

# Generate sample knowledge graph
@st.cache_data
def generate_sample_graph():
    """Generate a sample knowledge graph for visualization."""
    G = nx.Graph()
    
    # Define entities by type
    entities = {
        "PERSON": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Williams"],
        "ORGANIZATION": ["Acme Corp", "Tech Solutions", "Global Industries", "Innovation Labs"],
        "LOCATION": ["New York", "San Francisco", "London", "Tokyo"],
        "CONCEPT": ["AI", "Machine Learning", "Cloud Computing", "Blockchain", "IoT"],
        "PRODUCT": ["Product X", "Service Y", "Platform Z"]
    }
    
    # Add nodes with attributes
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            G.add_node(entity, type=entity_type)
    
    # Add edges (relationships)
    relationships = [
        ("John Doe", "Acme Corp", "WORKS_FOR"),
        ("Jane Smith", "Tech Solutions", "WORKS_FOR"),
        ("Bob Johnson", "Global Industries", "WORKS_FOR"),
        ("Alice Williams", "Innovation Labs", "WORKS_FOR"),
        ("Acme Corp", "New York", "LOCATED_IN"),
        ("Tech Solutions", "San Francisco", "LOCATED_IN"),
        ("Global Industries", "London", "LOCATED_IN"),
        ("Innovation Labs", "Tokyo", "LOCATED_IN"),
        ("John Doe", "AI", "EXPERT_IN"),
        ("Jane Smith", "Machine Learning", "EXPERT_IN"),
        ("Bob Johnson", "Cloud Computing", "EXPERT_IN"),
        ("Alice Williams", "Blockchain", "EXPERT_IN"),
        ("Acme Corp", "Product X", "PRODUCES"),
        ("Tech Solutions", "Service Y", "PRODUCES"),
        ("Global Industries", "Platform Z", "PRODUCES"),
        ("AI", "Machine Learning", "RELATES_TO"),
        ("Machine Learning", "Cloud Computing", "RELATES_TO"),
        ("Blockchain", "IoT", "RELATES_TO"),
        ("Product X", "AI", "USES"),
        ("Service Y", "Machine Learning", "USES"),
        ("Platform Z", "Cloud Computing", "USES"),
        # Add more connections
        ("John Doe", "Jane Smith", "COLLABORATES_WITH"),
        ("Tech Solutions", "Innovation Labs", "PARTNERS_WITH"),
        ("New York", "San Francisco", "CONNECTED_TO"),
    ]
    
    for source, target, rel_type in relationships:
        G.add_edge(source, target, relationship=rel_type)
    
    return G

# Load graph
with st.spinner("Loading knowledge graph..."):
    G = generate_sample_graph()

# Filter graph
G_filtered = G.copy()

# Filter by entity types
if entity_types:
    nodes_to_keep = [n for n in G_filtered.nodes() if G_filtered.nodes[n].get('type') in entity_types]
    G_filtered = G_filtered.subgraph(nodes_to_keep).copy()

# Filter by degree
if min_degree > 0:
    nodes_to_keep = [n for n in G_filtered.nodes() if G_filtered.degree(n) >= min_degree]
    G_filtered = G_filtered.subgraph(nodes_to_keep).copy()

# Limit number of nodes
if len(G_filtered) > max_nodes:
    # Keep nodes with highest degree
    top_nodes = sorted(G_filtered.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
    G_filtered = G_filtered.subgraph([n for n, d in top_nodes]).copy()

# Calculate graph statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Graph Statistics")
st.sidebar.metric("Nodes", len(G_filtered))
st.sidebar.metric("Edges", G_filtered.number_of_edges())
st.sidebar.metric("Density", f"{nx.density(G_filtered):.3f}")
if len(G_filtered) > 0:
    st.sidebar.metric("Avg Degree", f"{sum(dict(G_filtered.degree()).values()) / len(G_filtered):.1f}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🕸️ Graph Visualization",
    "📊 Node Analysis",
    "🔗 Relationship Browser",
    "🌐 Communities"
])

with tab1:
    st.markdown("## Interactive Knowledge Graph")
    
    if len(G_filtered) == 0:
        st.warning("No nodes match the current filters. Try adjusting the filters in the sidebar.")
    else:
        # Calculate layout
        if layout_algorithm == "Spring":
            pos = nx.spring_layout(G_filtered, seed=42, k=1/np.sqrt(len(G_filtered)))
        elif layout_algorithm == "Circular":
            pos = nx.circular_layout(G_filtered)
        elif layout_algorithm == "Kamada-Kawai":
            pos = nx.kamada_kawai_layout(G_filtered)
        else:  # Spectral
            pos = nx.spectral_layout(G_filtered)
        
        # Calculate node sizes
        if node_size_by == "Degree":
            node_sizes = dict(G_filtered.degree())
        elif node_size_by == "Betweenness":
            node_sizes = nx.betweenness_centrality(G_filtered)
        elif node_size_by == "Pagerank":
            node_sizes = nx.pagerank(G_filtered)
        else:  # Uniform
            node_sizes = {n: 1 for n in G_filtered.nodes()}
        
        # Normalize sizes
        max_size = max(node_sizes.values()) if node_sizes.values() else 1
        node_sizes = {k: (v / max_size) * 40 + 10 for k, v in node_sizes.items()}
        
        # Color by entity type
        color_map = {
            "PERSON": "blue",
            "ORGANIZATION": "green",
            "LOCATION": "red",
            "CONCEPT": "orange",
            "PRODUCT": "purple"
        }
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes_list = []
        
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_type = G_filtered.nodes[node].get('type', 'UNKNOWN')
            node_colors.append(color_map.get(node_type, 'gray'))
            
            degree = G_filtered.degree(node)
            node_text.append(f"{node}<br>Type: {node_type}<br>Connections: {degree}")
            node_sizes_list.append(node_sizes[node])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            text=[n for n in G_filtered.nodes()] if show_labels else None,
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes_list,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Knowledge Graph ({len(G_filtered)} nodes, {G_filtered.number_of_edges()} edges)',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("### 🎨 Legend")
        legend_cols = st.columns(len(color_map))
        for i, (entity_type, color) in enumerate(color_map.items()):
            with legend_cols[i]:
                st.markdown(f"<span style='color:{color}'>●</span> {entity_type}", unsafe_allow_html=True)

with tab2:
    st.markdown("## Node Analysis")
    
    if len(G_filtered) == 0:
        st.warning("No nodes to analyze with current filters.")
    else:
        # Top nodes by centrality
        st.markdown("### 🏆 Most Important Nodes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**By Degree (Connections)**")
            degree_centrality = dict(G_filtered.degree())
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, degree in top_degree:
                node_type = G_filtered.nodes[node].get('type', 'UNKNOWN')
                st.write(f"**{node}** ({node_type}): {degree}")
        
        with col2:
            st.markdown("**By Betweenness (Bridge Nodes)**")
            betweenness = nx.betweenness_centrality(G_filtered)
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, score in top_betweenness:
                node_type = G_filtered.nodes[node].get('type', 'UNKNOWN')
                st.write(f"**{node}** ({node_type}): {score:.3f}")
        
        with col3:
            st.markdown("**By PageRank (Influence)**")
            pagerank = nx.pagerank(G_filtered)
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, score in top_pagerank:
                node_type = G_filtered.nodes[node].get('type', 'UNKNOWN')
                st.write(f"**{node}** ({node_type}): {score:.3f}")
        
        # Node details
        st.markdown("### 🔍 Node Details")
        
        selected_node = st.selectbox(
            "Select a node to explore",
            sorted(G_filtered.nodes())
        )
        
        if selected_node:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Node Information**")
                node_type = G_filtered.nodes[selected_node].get('type', 'UNKNOWN')
                st.info(f"**Type:** {node_type}")
                st.info(f"**Degree:** {G_filtered.degree(selected_node)}")
                st.info(f"**Betweenness:** {betweenness.get(selected_node, 0):.3f}")
                st.info(f"**PageRank:** {pagerank.get(selected_node, 0):.3f}")
            
            with col2:
                st.markdown("**Connected Nodes**")
                neighbors = list(G_filtered.neighbors(selected_node))
                
                if neighbors:
                    neighbor_data = []
                    for neighbor in neighbors:
                        edge_data = G_filtered.get_edge_data(selected_node, neighbor)
                        relationship = edge_data.get('relationship', 'UNKNOWN') if edge_data else 'UNKNOWN'
                        neighbor_type = G_filtered.nodes[neighbor].get('type', 'UNKNOWN')
                        
                        neighbor_data.append({
                            'Node': neighbor,
                            'Type': neighbor_type,
                            'Relationship': relationship
                        })
                    
                    df_neighbors = pd.DataFrame(neighbor_data)
                    st.dataframe(df_neighbors, use_container_width=True, hide_index=True)
                else:
                    st.info("No connections found")

with tab3:
    st.markdown("## Relationship Browser")
    
    if G_filtered.number_of_edges() == 0:
        st.warning("No relationships to display with current filters.")
    else:
        # Extract all relationships
        relationships = []
        for source, target, data in G_filtered.edges(data=True):
            relationships.append({
                'Source': source,
                'Source Type': G_filtered.nodes[source].get('type', 'UNKNOWN'),
                'Relationship': data.get('relationship', 'UNKNOWN'),
                'Target': target,
                'Target Type': G_filtered.nodes[target].get('type', 'UNKNOWN')
            })
        
        df_relationships = pd.DataFrame(relationships)
        
        # Relationship type filter
        rel_types = df_relationships['Relationship'].unique()
        selected_rel_types = st.multiselect(
            "Filter by Relationship Type",
            rel_types,
            default=list(rel_types)
        )
        
        df_filtered_rel = df_relationships[df_relationships['Relationship'].isin(selected_rel_types)]
        
        # Display table
        st.dataframe(df_filtered_rel, use_container_width=True, hide_index=True)
        
        # Relationship statistics
        st.markdown("### 📊 Relationship Statistics")
        
        rel_counts = df_relationships['Relationship'].value_counts()
        fig_rel = go.Figure(data=[
            go.Bar(x=rel_counts.index, y=rel_counts.values)
        ])
        fig_rel.update_layout(
            title='Relationship Type Distribution',
            xaxis_title='Relationship Type',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig_rel, use_container_width=True)

with tab4:
    st.markdown("## Community Detection")
    
    if len(G_filtered) == 0:
        st.warning("No nodes to analyze for communities.")
    else:
        # Detect communities using Louvain algorithm
        from networkx.algorithms import community
        
        communities = community.greedy_modularity_communities(G_filtered)
        
        st.markdown(f"### Found {len(communities)} Communities")
        
        # Assign community IDs to nodes
        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = i
        
        # Display communities
        for i, comm in enumerate(communities):
            with st.expander(f"**Community {i+1}** ({len(comm)} nodes)", expanded=(i<3)):
                comm_nodes = list(comm)
                
                # Group by entity type
                type_groups = {}
                for node in comm_nodes:
                    node_type = G_filtered.nodes[node].get('type', 'UNKNOWN')
                    if node_type not in type_groups:
                        type_groups[node_type] = []
                    type_groups[node_type].append(node)
                
                for node_type, nodes in type_groups.items():
                    st.markdown(f"**{node_type}:** {', '.join(nodes)}")
        
        # Community size distribution
        community_sizes = [len(comm) for comm in communities]
        fig_comm = go.Figure(data=[
            go.Bar(x=[f"C{i+1}" for i in range(len(communities))], y=community_sizes)
        ])
        fig_comm.update_layout(
            title='Community Size Distribution',
            xaxis_title='Community',
            yaxis_title='Number of Nodes',
            height=400
        )
        st.plotly_chart(fig_comm, use_container_width=True)

# Export options
st.markdown("---")
st.markdown("### 📥 Export Graph")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Export Graph Data"):
        # Export as edge list
        edge_list = nx.to_pandas_edgelist(G_filtered)
        csv = edge_list.to_csv(index=False)
        st.download_button(
            "Download Edge List CSV",
            csv,
            "graph_edges.csv",
            "text/csv"
        )

with col2:
    if st.button("📊 Export Statistics"):
        stats = {
            'Metric': ['Nodes', 'Edges', 'Density', 'Avg Degree', 'Communities'],
            'Value': [
                len(G_filtered),
                G_filtered.number_of_edges(),
                f"{nx.density(G_filtered):.3f}",
                f"{sum(dict(G_filtered.degree()).values()) / len(G_filtered):.2f}" if len(G_filtered) > 0 else 0,
                len(communities) if len(G_filtered) > 0 else 0
            ]
        }
        df_stats = pd.DataFrame(stats)
        csv = df_stats.to_csv(index=False)
        st.download_button(
            "Download Statistics CSV",
            csv,
            "graph_statistics.csv",
            "text/csv"
        )

with col3:
    if st.button("🖼️ Save Visualization"):
        st.info("High-res visualization export coming soon...")
