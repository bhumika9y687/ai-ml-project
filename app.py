"""
Cardio Vue AI — Intelligent Cardiovascular Risk Prediction & Monitoring
Enhanced Professional Interface with Modern Design
"""

import streamlit as st
import time
from datetime import datetime
import warnings
from modules.clinical_guidelines_viewer import show_clinical_guidelines
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio Vue AI | Intelligent Cardiovascular Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOCAL IMPORTS ─────────────────────────────────────────────────────────────
from utils.theme import CSS
from utils.database import init_db, authenticate, register_user
from modules.shared_navigation import show_sidebar
from modules.doctor_pages import show_doctor_dashboard, show_patients
from modules.researcher_pages import (
    show_dataset, show_experiments, show_literature, show_model_lab,
    show_research_hub, show_researcher_dashboard,
)
from modules.patient_pages import (
    show_ai_assistant, show_appointments, show_goals, show_health_records,
    show_notifications, show_patient_dashboard, show_profile,
    show_risk_prediction, show_whatif,
)

st.markdown(CSS, unsafe_allow_html=True)

# ─── INIT DB ───────────────────────────────────────────────────────────────────
init_db()

# ─── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'logged_in': False,
        'username': None,
        'user': None,
        'session_start': None,
        'page': 'dashboard',
        'gemini_key': '',
        'chat_history': [],
        'last_prediction': None,
        'last_features': None,
        'viewing_patient': None,
        'selected_patient': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# ─── LOADING ANIMATION ─────────────────────────────────────────────────────────
with st.spinner('🫀 Loading Cardio Vue AI...'):
    time.sleep(0.3)

# ─── CUSTOM JAVASCRIPT FOR INTERACTIONS ───────────────────────────────────────
st.markdown("""
<script>
// Smooth button interactions
document.querySelectorAll('.stButton button').forEach(button => {
    button.addEventListener('click', function() {
        this.style.transform = 'scale(0.98)';
        setTimeout(() => {
            this.style.transform = '';
        }, 150);
    });
});

// Add fade-in animation to cards
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
});

document.querySelectorAll('.card, .card-sm').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(card);
});

// Keyboard navigation for sidebar
document.querySelectorAll('[data-testid="stSidebar"] .stButton button').forEach((btn, idx) => {
    btn.setAttribute('tabindex', '0');
    btn.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            btn.click();
        }
    });
});
</script>
""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
def add_footer():
    st.markdown("""
    <div class="site-footer">
        <span>🫀 Cardio Vue AI </span>
        <span style="color:var(--b2)">·</span>
        <span>Ensemble ML · HIPAA Compliant</span>
        <span style="color:var(--b2)">·</span>
        <a href="#">Privacy</a>
        <span style="color:var(--b2)">·</span>
        <a href="#">Terms and Conditions</a>
    </div>
    """, unsafe_allow_html=True)

# ─── AUTH ──────────────────────────────────────────────────────────────────────
def show_login():
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="login-hero">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.5rem">
                <div style="width:28px;height:28px;background:rgba(20,184,166,0.18);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem">🫀</div>
                <span style="font-size:0.88rem;font-weight:700;color:var(--t1)">Cardio Vue <span style="color:var(--teal-lt)">AI</span></span>
                <span class="badge badge-teal" style="margin-left:4px">v2.0</span>
            </div>
            <h1>Clinical-grade<br><em>Heart Risk</em><br>Intelligence</h1>
            <p style="margin-top:0.75rem;max-width:340px">
                Ensemble ML trained on 253,680 patient records. Real-time cardiovascular risk prediction for patients, clinicians, and researchers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <img src="https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=700&q=80&auto=format&fit=crop"
             class="section-img" style="height:175px;object-fit:cover;margin-bottom:6px"
             alt="Cardiologist reviewing patient data"/>
        <p class="img-caption">Trusted by clinicians · Validated on 253,680 real patient records</p>
        """, unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        for col, num, lbl in [(s1,"253K","Records"),(s2,"94.2%","Accuracy"),(s3,"8","ML Models")]:
            with col:
                st.markdown(f'<div class="stat-box"><div><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:0.75rem">
            <div class="card-title">Platform capabilities</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
                <div class="feat-item"><div class="feat-dot"></div>Stacking ensemble risk prediction</div>
                <div class="feat-item"><div class="feat-dot"></div>What-If scenario planner</div>
                <div class="feat-item"><div class="feat-dot"></div>AI health assistant</div>
                <div class="feat-item" style="border:none"><div class="feat-dot"></div>SHAP explainability</div>
                <div class="feat-item" style="border:none"><div class="feat-dot"></div>Research analytics lab</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="margin-bottom:0.5rem">
            <div style="font-size:0.85rem;font-weight:600;color:var(--t1);margin-bottom:0.125rem">Sign in to your account</div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            with st.form("login"):
                username_in = st.text_input("Username", placeholder="your username")
                password_in = st.text_input("Password", type="password", placeholder="••••••••")
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    submitted = st.form_submit_button("Sign in →", type="primary", use_container_width=True)
                with col_btn2:
                    st.markdown('<div style="text-align:right;margin-top:8px"><a href="#" style="color:var(--teal);font-size:0.7rem">Forgot password?</a></div>', unsafe_allow_html=True)
                
                if submitted:
                    if username_in and password_in:
                        with st.spinner("Authenticating..."):
                            user = authenticate(username_in, password_in)
                            if user:
                                st.session_state.update({
                                    'logged_in': True, 
                                    'username': username_in,
                                    'user': user, 
                                    'session_start': datetime.now(),
                                    'page': 'dashboard'
                                })
                                st.success(f"Welcome back, {user['name']}!")
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                st.error("Invalid credentials. Please check your username and password.")
                    else:
                        st.warning("Please enter both username and password.")

        with tab2:
            with st.form("register"):
                st.markdown('<div style="margin-bottom:1rem;"><span style="font-weight: 600;">Create your account</span></div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    ru = st.text_input("Username *", placeholder="Choose username")
                    rn = st.text_input("Full Name *", placeholder="Your full name")
                    rr = st.selectbox("Role *", ["patient", "doctor", "researcher"], format_func=lambda x: x.title())
                with c2:
                    rp = st.text_input("Password *", type="password", placeholder="Strong password")
                    re = st.text_input("Email *", placeholder="you@email.com")
                    rage = st.number_input("Age", 18, 100, 35) if rr == "patient" else None
                
                st.markdown("""
                <div style="font-size:0.7rem;color:var(--t3);margin:0.5rem 0;">
                    By creating an account, you agree to our Terms of Service and Privacy Policy.
                </div>
                """, unsafe_allow_html=True)
                
                if st.form_submit_button("Create account →", type="primary", use_container_width=True):
                    if ru and rp and rn and re:
                        if len(rp) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            extra = {'age': rage} if rage else {}
                            ok, msg = register_user(ru, rp, rr, rn, re, extra)
                            if ok:
                                st.success(msg)
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        st.warning("Please fill all required fields.")



# ─── RISK PREDICTION PAGE ──────────────────────────────────────────────────────
















# ─── MAIN ROUTER ───────────────────────────────────────────────────────────────
def main():
    if st.session_state.get('session_start'):
        elapsed = (datetime.now() - st.session_state['session_start']).seconds
        if elapsed > 2700:
            st.warning("⏰ Session expired due to inactivity. Please sign in again.")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    if not st.session_state.get('logged_in'):
        show_login()
        add_footer()
        return

    user = st.session_state.user
    role = user['role']
    page = st.session_state.get('page', 'dashboard')
    initials = ''.join(w[0].upper() for w in user['name'].split()[:2])

    role_badge_cls = {'patient': 'badge-sky', 'doctor': 'badge-green', 'researcher': 'badge-amber'}.get(role, 'badge-teal')

    st.markdown(f"""
    <div class="top-nav">
        <div class="top-nav-brand">
            <div style="width:24px;height:24px;background:rgba(20,184,166,0.15);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:0.85rem">🫀</div>
            Cardio Vue&nbsp;<span>AI</span>
            <span class="badge badge-teal" style="margin-left:4px">v2.0</span>
        </div>
        <div class="top-nav-right">
            <span class="badge {role_badge_cls}">{role.upper()}</span>
            <div class="avatar">{initials}</div>
            <span style="font-size:0.78rem;color:var(--t2)">{user['name'].split()[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    show_sidebar()
    
    ROUTES = {
        'patient': {
            'dashboard': show_patient_dashboard,
            'risk_prediction': show_risk_prediction,
            'whatif': show_whatif,
            'health_records': show_health_records,
            'appointments': show_appointments,
            'goals': show_goals,
            'profile': show_profile,
            'ai_assistant': show_ai_assistant,
            'notifications': show_notifications,
        },
        'doctor': {
            'dashboard': show_doctor_dashboard,
            'patients': show_patients,
            'risk_prediction': lambda: show_risk_prediction(st.session_state.get('selected_patient')) if st.session_state.get('selected_patient') else show_risk_prediction(),
            'whatif': show_whatif,
            'appointments': show_appointments,
            'guidelines': show_clinical_guidelines,
        },
        'researcher': {
            'dashboard': show_researcher_dashboard,
            'research_hub': show_research_hub,
            'literature_review': show_literature,
            'dataset': show_dataset,
            'model_lab': show_model_lab,
            'experiments': show_experiments,
        }
    }

    handler = ROUTES.get(role, {}).get(page)
    if handler:
        handler()
    else:
        ROUTES.get(role, {}).get('dashboard', show_patient_dashboard)()
    
    add_footer()

if __name__ == "__main__":
    main()
