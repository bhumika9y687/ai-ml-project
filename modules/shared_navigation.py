"""Shared role navigation for the Cardio Vue Streamlit app."""

import streamlit as st
from datetime import datetime
from utils.ml_engine import render_model_download_ui


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        render_model_download_ui()
    user = st.session_state.user
    role = user['role']
    nav = {
        'patient': [
            ('dashboard',       '⬛', 'Dashboard'),
            ('risk_prediction', '◎', 'Risk Assessment'),
            ('whatif',          '⊙', 'What-If Planner'),
            ('health_records',  '⊞', 'Health Records'),
            ('appointments',    '◫', 'Appointments'),
            ('goals',           '◈', 'Goals'),
            ('ai_assistant',    '◉', 'AI Assistant'),
            ('notifications',   '⊡', 'Notifications'),
            ('profile',         '◷', 'Profile'),
        ],
        'doctor': [
            ('dashboard',       '⬛', 'Dashboard'),
            ('patients',        '◎', 'Patients'),
            ('risk_prediction', '◉', 'Risk Analysis'),
            ('whatif',          '⊙', 'Scenario Planner'),
            ('appointments',    '◫', 'Appointments'),
            ('guidelines',      '📋', 'Guidelines'),
        ],
        'researcher': [
            ('dashboard',        '⬛', 'Dashboard'),
            ('research_hub',     '🔬', 'Research Hub'),
            ('literature_review','📚', 'Literature Review'),
            ('dataset',          '🗂️', 'Dataset Explorer'),
            ('model_lab',        '🧠', 'Model Lab'),
            ('experiments',      '🧪', 'Experiments'),
        ],
    }

    with st.sidebar:
        initials = ''.join(w[0].upper() for w in user['name'].split()[:2])
        st.markdown(f"""
        <div style="padding:1rem 0.75rem 0.75rem;border-bottom:1px solid var(--b1);margin-bottom:0.5rem">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem">
                <div style="width:22px;height:22px;background:rgba(20,184,166,0.14);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:0.8rem">🫀</div>
                <span style="font-size:0.85rem;font-weight:700;letter-spacing:-0.01em;color:var(--t1)">Cardio Vue <span style="color:var(--teal-lt)">AI</span></span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <div class="avatar" style="width:32px;height:32px;font-size:0.75rem">{initials}</div>
                <div style="min-width:0">
                    <div style="font-size:0.78rem;font-weight:600;color:var(--t1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{user['name']}</div>
                    <span class="role-badge role-{role}" style="margin-top:2px;display:inline-flex">{role.upper()}</span>
                </div>
            </div>
        </div>
        <div style="padding:0 0.375rem">
            <div style="font-size:0.63rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:0.09em;padding:0.5rem 0.625rem 0.25rem">Navigation</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.session_state.get('page', 'dashboard')
        with st.container():
            st.markdown('<div style="padding:0 0.375rem">', unsafe_allow_html=True)
            for key, _ico, label in nav.get(role, []):
                is_active = page == key
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
                    st.session_state.page = key
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider" style="margin:0.75rem 0">', unsafe_allow_html=True)

        with st.expander("⚙ AI Settings", expanded=False):
            api_key = st.text_input("Gemini API Key",
                                     value=st.session_state.get('gemini_key', ''),
                                     type="password",
                                     placeholder="Enter API key from makersuite.google.com",
                                     help="Free tier available. Enables advanced AI responses.")
            if api_key:
                st.session_state.gemini_key = api_key
                st.success("✓ Key saved securely")

        st.markdown('<div style="padding:0 0.5rem">', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Sign out", key="signout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        with col_b:
            if st.button("⟳ Clear Cache", key="clear_cache", use_container_width=True, help="Clear cached data"):
                st.cache_data.clear()
                st.success("Cache cleared!")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;padding-top:0.5rem;border-top:1px solid var(--b1)">
            <span style="font-size:0.6rem;color:var(--t3)">Cardio Vue AI · {datetime.now().strftime('%b %Y')}</span>
        </div>
        """, unsafe_allow_html=True)
