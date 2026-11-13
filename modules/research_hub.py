# modules/research_hub.py
"""
Research Hub - Enhanced Literature Review Features
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.literature_db import REAL_PAPERS, get_papers_by_section, search_papers, get_reading_stats

def show_research_hub():
    """Main research hub interface"""
    
    # Use simple markdown instead of complex HTML for header
    st.markdown("# 🔬 Research Hub")
    st.markdown("*Curated cardiovascular ML literature · 15+ real peer-reviewed papers*")
    
    # Stats overview
    stats = get_reading_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Papers", stats['total'], help="peer-reviewed")
    with col2:
        st.metric("📊 Total Citations", f"{stats['total_citations']:,}", help="across all papers")
    with col3:
        st.metric("⭐ Avg Impact Factor", f"{stats['avg_if']:.1f}", help="journal score")
    with col4:
        st.metric("🎯 Research Areas", len(stats['by_section']), help="specialties")
    
    st.divider()
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs([
        "📄 Literature Library", 
        "🔍 Paper Explorer", 
        "📊 Research Analytics"
    ])
    
    with tab1:
        show_literature_library()
    
    with tab2:
        show_paper_explorer()
    
    with tab3:
        show_research_analytics()

def show_literature_library():
    """Display the literature library with filtering"""
    
    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input("🔍 Search papers", placeholder="Title, author, keyword...", key="lit_search")
    with col2:
        sections = ["All"] + sorted(list(set(p['section'] for p in REAL_PAPERS)))
        section_filter = st.selectbox("Section", sections, key="lit_section")
    with col3:
        sort_options = ["Citations (high-low)", "Year (new-old)", "Impact Factor (high-low)", "Title A-Z"]
        sort_by = st.selectbox("Sort by", sort_options, key="lit_sort")
    
    # Filter papers
    papers = REAL_PAPERS.copy()
    if search_term:
        papers = search_papers(search_term)
    if section_filter != "All":
        papers = [p for p in papers if p['section'] == section_filter]
    
    # Sort
    if sort_by == "Citations (high-low)":
        papers.sort(key=lambda x: x['citations'], reverse=True)
    elif sort_by == "Year (new-old)":
        papers.sort(key=lambda x: x['year'], reverse=True)
    elif sort_by == "Impact Factor (high-low)":
        papers.sort(key=lambda x: x['if_score'] if isinstance(x['if_score'], (int, float)) else 0, reverse=True)
    elif sort_by == "Title A-Z":
        papers.sort(key=lambda x: x['title'])
    
    st.markdown(f"### 📖 {len(papers)} Papers Found")
    
    # Display papers
    for paper in papers:
        display_paper_card(paper)

def display_paper_card(paper):
    """Display a single paper card with details using Streamlit native components"""
    
    # Determine badge color based on impact
    if isinstance(paper['if_score'], (int, float)):
        if paper['if_score'] >= 20:
            badge = "🏆 Top-tier"
            badge_color = "#f59e0b"
        elif paper['if_score'] >= 10:
            badge = "⭐ High-impact"
            badge_color = "#22c55e"
        elif paper['if_score'] >= 5:
            badge = "📊 Mid-tier"
            badge_color = "#14b8a6"
        else:
            badge = "📄 Core"
            badge_color = "#6b7280"
    else:
        badge = "🏛️ Conference"
        badge_color = "#a78bfa"
    
    # Open access badge
    oa_badge = "🔓 Open Access" if paper.get('open_access') else "🔒 Subscription"
    
    # Create expandable card for each paper
    with st.container():
        # Header with title and badges
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{paper['title']}**")
            st.caption(f"{paper['authors']} · {paper['journal']} · {paper['year']}")
        with col2:
            st.markdown(f"`{badge}`")
            st.caption(oa_badge)
        
        # Key finding in an info box
        st.info(f"📌 **KEY FINDING:** {paper['key_finding']}")
        
        # Summary
        st.markdown(paper['summary'])
        
        # Tags and citations
        cols = st.columns([3, 1])
        with cols[0]:
            tags = " ".join([f"`{tag}`" for tag in paper['tags'][:4]])
            st.markdown(tags)
        with cols[1]:
            st.markdown(f"📊 **{paper['citations']:,}** citations")
        
        # Link button
        st.link_button("View Paper →", paper['url'], use_container_width=True)
        
        st.divider()

def show_paper_explorer():
    """Interactive paper explorer with visualizations"""
    
    st.markdown("### 🔍 Interactive Paper Explorer")
    
    # Create a dataframe for analysis
    df = pd.DataFrame(REAL_PAPERS)
    
    # Year distribution
    st.markdown("#### 📅 Publication Timeline")
    year_counts = df['year'].value_counts().sort_index()
    fig = go.Figure(go.Scatter(
        x=year_counts.index.tolist(),
        y=year_counts.values.tolist(),
        mode='lines+markers',
        line=dict(color='#14b8a6', width=2),
        marker=dict(size=8, color='#5eead4'),
        fill='tozeroy',
        fillcolor='rgba(20,184,166,0.1)'
    ))
    fig.update_layout(
        title="Papers by Year",
        height=300,
        xaxis_title="Year",
        yaxis_title="Number of Papers",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Citations by section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Citations by Research Area")
        section_cites = df.groupby('section')['citations'].sum().sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(
            x=section_cites.values,
            y=section_cites.index,
            orientation='h',
            marker=dict(color='#14b8a6', line=dict(width=0)),
            text=[f'{v:,}' for v in section_cites.values],
            textposition='outside'
        ))
        fig2.update_layout(height=300, xaxis_title="Total Citations", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Top Cited Papers")
        top_papers = df.nlargest(5, 'citations')[['title', 'citations', 'year']]
        for idx, row in top_papers.iterrows():
            with st.container():
                st.markdown(f"**{row['title'][:60]}...**")
                st.caption(f"📊 {row['citations']:,} cites · {row['year']}")
                st.divider()
    
    # Tag cloud
    st.markdown("#### 🏷️ Research Topic Tags")
    all_tags = [tag for paper in REAL_PAPERS for tag in paper['tags']]
    tag_counts = pd.Series(all_tags).value_counts().head(12)
    
    fig3 = go.Figure(go.Bar(
        x=tag_counts.values,
        y=tag_counts.index,
        orientation='h',
        marker=dict(color='#a78bfa', line=dict(width=0))
    ))
    fig3.update_layout(height=300, xaxis_title="Frequency", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

def show_research_analytics():
    """Advanced research analytics"""
    
    st.markdown("### 📊 Research Analytics Dashboard")
    
    df = pd.DataFrame(REAL_PAPERS)
    
    # Impact factor distribution
    impact_scores = [p['if_score'] for p in REAL_PAPERS if isinstance(p['if_score'], (int, float))]
    if impact_scores:
        fig = go.Figure(go.Histogram(
            x=impact_scores,
            nbinsx=15,
            marker=dict(color='#14b8a6', line=dict(width=0)),
            opacity=0.7
        ))
        fig.update_layout(
            title="Impact Factor Distribution",
            height=300,
            xaxis_title="Impact Factor",
            yaxis_title="Number of Papers",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Reading progress
    st.markdown("#### 📖 Reading Progress Tracker")
    
    # Simulate reading progress (in real app, this would come from user data)
    reading_status = {
        'Read': 8,
        'Reading': 4,
        'To Read': 5
    }
    
    fig2 = go.Figure(go.Pie(
        values=list(reading_status.values()),
        labels=list(reading_status.keys()),
        hole=0.5,
        marker=dict(colors=['#22c55e', '#f59e0b', '#6b7280'])
    ))
    fig2.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Research themes
    st.markdown("#### 🎯 Research Theme Analysis")
    
    themes = {
        "Machine Learning": ["machine learning", "xgboost", "random forest", "deep learning", "neural networks"],
        "Clinical Implementation": ["clinical", "risk prediction", "ehr", "electronic health records", "real-world"],
        "ECG Analysis": ["ecg", "electrocardiogram", "atrial fibrillation", "arrhythmia"],
        "Explainability": ["shap", "xai", "interpretability", "explainable"],
        "Epidemiology": ["brfss", "global burden", "dataset", "population"]
    }
    
    theme_counts = {}
    for theme, keywords in themes.items():
        count = sum(1 for paper in REAL_PAPERS if any(kw in str(paper['tags']).lower() for kw in keywords))
        theme_counts[theme] = count
    
    fig3 = go.Figure(go.Bar(
        x=list(theme_counts.values()),
        y=list(theme_counts.keys()),
        orientation='h',
        marker=dict(color='#f59e0b', line=dict(width=0)),
        text=list(theme_counts.values()),
        textposition='outside'
    ))
    fig3.update_layout(height=300, xaxis_title="Number of Papers", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# Export for use in main app
def get_literature_for_researcher():
    """Get literature data for researcher dashboard"""
    return {
        'papers': REAL_PAPERS,
        'stats': get_reading_stats()
    }
