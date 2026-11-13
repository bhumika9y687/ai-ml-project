# modules/literature_review.py
"""
Literature Review Module - Clean, Professional UI with Genuine Papers
"""

import streamlit as st
from utils.literature_db import get_all_papers, get_papers_by_section, get_reading_stats

def show_literature_review():
    """Main literature review interface"""
    
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin: 0; 
                   background: linear-gradient(135deg, #e4ecf4, #5eead4); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent;">
            📚 Literature Review
        </h1>
        <p style="color: #8ea3b8; margin: 0.25rem 0 0;">Peer-reviewed cardiovascular AI research with direct access</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    stats = get_reading_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Papers", stats['total'], delta=None)
    with col2:
        st.metric("📊 Citations", f"{stats['total_citations']:,}", delta=None)
    with col3:
        st.metric("📚 Research Areas", len(stats['by_section']), delta=None)
    with col4:
        st.metric("⭐ Avg Impact Factor", f"{stats['avg_if']:.1f}", delta=None)
    
    st.divider()
    
    (tab1,) = st.tabs(["📖 Paper Library"])
    
    with tab1:
        show_paper_library()

def show_paper_library():
    """Display all papers in a clean format"""
    
    # Simple filters in one row
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        sections = ["All Papers"] + list(get_reading_stats()['by_section'].keys())
        selected_section = st.selectbox("Research Area", sections, label_visibility="collapsed")
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Most Cited", "Newest First", "Highest Impact"], label_visibility="collapsed")
    
    with col3:
        st.write("")  # Spacer
    
    # Filter papers
    papers = get_all_papers()
    if selected_section != "All Papers":
        papers = get_papers_by_section(selected_section)
    
    # Sort
    if sort_by == "Most Cited":
        papers.sort(key=lambda x: x['citations'], reverse=True)
    elif sort_by == "Newest First":
        papers.sort(key=lambda x: x['year'], reverse=True)
    elif sort_by == "Highest Impact":
        papers.sort(key=lambda x: x['if_score'] if isinstance(x['if_score'], (int, float)) else 0, reverse=True)
    
    st.markdown(f"**{len(papers)} papers found**")
    
    # Display papers
    for paper in papers:
        display_clean_paper_card(paper)

def display_clean_paper_card(paper):
    """Display a clean, professional paper card"""
    
    # Determine badge
    if isinstance(paper['if_score'], (int, float)):
        if paper['if_score'] >= 20:
            badge = "🏆 Top Tier"
            badge_color = "#f59e0b"
        elif paper['if_score'] >= 10:
            badge = "⭐ High Impact"
            badge_color = "#22c55e"
        elif paper['if_score'] >= 5:
            badge = "📊 Quality"
            badge_color = "#14b8a6"
        else:
            badge = "📄 Core"
            badge_color = "#6b7280"
    else:
        badge = "🏛️ Conference"
        badge_color = "#a78bfa"
    
    card_html = f'''
    <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem; border-left: 3px solid {badge_color};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 0.95rem; color: #e4ecf4; margin-bottom: 0.25rem;">
                    {paper['title']}
                </div>
                <div style="font-size: 0.75rem; color: #8ea3b8;">
                    {paper['authors']} · {paper['journal']} · {paper['year']}
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem; margin-left: 1rem;">
                <span style="background: {badge_color}20; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; color: {badge_color};">
                    {badge}
                </span>
                <a href="{paper['url']}" target="_blank" style="text-decoration: none;">
                    <span style="background: #14b8a6; padding: 2px 10px; border-radius: 6px; font-size: 0.7rem; color: white;">
                        Read →
                    </span>
                </a>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #8ea3b8; margin-top: 0.5rem; line-height: 1.5;">
            {paper['summary'][:200]}...
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.7rem;">
            <span>📊 {paper['citations']:,} citations</span>
            <span>🔑 {paper['key_finding'][:80]}...</span>
        </div>
    </div>
    '''
    
    st.markdown(card_html, unsafe_allow_html=True)
