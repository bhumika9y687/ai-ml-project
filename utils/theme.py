"""
CardioVue AI v2.0 — Enhanced Professional Medical SaaS Theme
────────────────────────────────────────────────────────────
Upgrades over v1:
• Syne display font for hero headings (bolder, more distinctive)
• Animated hero card with pulse dot
• Enhanced KPI cards (v2) with hover lift + bg icons
• Better paper/literature cards
• Richer gradient backgrounds
• Improved alert boxes with left-accent borders
• Smoother animations and micro-interactions
"""

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#8ea3b8', size=11),
    title_font=dict(family='Syne, sans-serif', color='#e4ecf4', size=13),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=10)),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.07)', borderwidth=1, font=dict(size=10)),
    margin=dict(l=4, r=4, t=36, b=4),
    hoverlabel=dict(bgcolor='#152235', bordercolor='rgba(20,184,166,0.4)',
                    font=dict(family='Inter', color='#e4ecf4', size=11)),
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* Backgrounds — deep slate-blue */
    --bg-950:  #060d18;
    --bg-900:  #0a1525;
    --bg-800:  #0f1e35;
    --bg-700:  #142540;
    --bg-600:  #1a2e4d;

    /* Surfaces */
    --s1:  rgba(255,255,255,0.030);
    --s2:  rgba(255,255,255,0.055);
    --s3:  rgba(255,255,255,0.085);
    --b1:  rgba(255,255,255,0.065);
    --b2:  rgba(255,255,255,0.120);

    /* Brand — teal/cyan */
    --teal:     #14b8a6;
    --teal-lt:  #5eead4;
    --teal-dim: rgba(20,184,166,0.14);

    /* Accent — warm amber */
    --amber:     #f59e0b;
    --amber-dim: rgba(245,158,11,0.14);

    /* Status */
    --s-low:      #22c55e;
    --s-moderate: #f59e0b;
    --s-high:     #f97316;
    --s-critical: #ef4444;

    /* Neutrals */
    --sky:     #38bdf8;
    --violet:  #a78bfa;
    --rose:    #f87171;
    --emerald: #34d399;

    /* Text */
    --t1:  #e4ecf4;
    --t2:  #8ea3b8;
    --t3:  #44607a;

    /* Radii */
    --r-sm: 6px;
    --r-md: 10px;
    --r-lg: 14px;
    --r-xl: 18px;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--t1);
    -webkit-font-smoothing: antialiased;
    font-size: 14px;
}
.stApp { background: var(--bg-950); }

.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    max-width: 1440px !important;
}

/* ── Top nav ──────────────────────────────────── */
.top-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.5rem; height: 52px;
    background: linear-gradient(180deg, #0a1830 0%, #0a1525 100%);
    border-bottom: 1px solid var(--b1);
    position: sticky; top: 0; z-index: 100;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(12px);
}
.top-nav-brand {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700; color: var(--t1); letter-spacing: -0.01em;
}
.top-nav-brand span { color: var(--teal-lt); }
.top-nav-right { display: flex; align-items: center; gap: 10px; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(20,184,166,0.25), rgba(56,189,248,0.15));
    border: 1px solid rgba(20,184,166,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: var(--teal-lt);
    transition: transform 0.15s;
}
.avatar:hover { transform: scale(1.08); }

/* ── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-900) !important;
    border-right: 1px solid var(--b1) !important;
    box-shadow: none;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left;
    background: transparent; border: none;
    color: var(--t2); padding: 7px 12px;
    border-radius: var(--r-md);
    font-size: 0.8rem; font-weight: 500;
    transition: background 0.15s, color 0.15s; margin: 1.5px 0;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: var(--s2); color: var(--t1);
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: var(--teal-dim) !important;
    color: var(--teal-lt) !important;
    border: 1px solid rgba(20,184,166,0.12) !important;
    box-shadow: none !important;
}

/* ── Cards ────────────────────────────────────── */
.card {
    background: var(--s1); border: 1px solid var(--b1);
    border-radius: var(--r-lg); padding: 1rem 1.125rem; margin-bottom: 0.75rem;
    transition: border-color 0.15s, background 0.15s;
}
.card:hover { border-color: var(--b2); }
.card-sm {
    background: var(--s1); border: 1px solid var(--b1);
    border-radius: var(--r-md); padding: 0.625rem 0.875rem; margin-bottom: 0.5rem;
}
.card-title {
    font-size: 0.67rem; font-weight: 600; color: var(--t3);
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.5rem;
}

/* ── KPI (original) ───────────────────────────── */
.kpi {
    background: var(--s1); border: 1px solid var(--b1);
    border-radius: var(--r-lg); padding: 0.875rem 1rem;
    display: flex; align-items: flex-start; gap: 0.75rem;
    transition: border-color 0.15s, transform 0.15s;
}
.kpi:hover { border-color: var(--b2); transform: translateY(-1px); }
.kpi-icon {
    width: 36px; height: 36px; border-radius: var(--r-md);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.kpi-val { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.025em; }
.kpi-label { font-size: 0.68rem; color: var(--t3); margin-top: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-sub { font-size: 0.7rem; color: var(--t2); margin-top: 3px; }

/* ── Page header ──────────────────────────────── */
.page-header {
    margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--b1);
    display: flex; align-items: center; justify-content: space-between;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 800; color: var(--t1); margin: 0; letter-spacing: -0.02em;
}
.page-sub { font-size: 0.76rem; color: var(--t3); margin-top: 3px; }
.sec-title {
    font-size: 0.7rem; font-weight: 700; color: var(--t3);
    text-transform: uppercase; letter-spacing: 0.1em; margin: 0.875rem 0 0.5rem;
    display: flex; align-items: center; gap: 6px;
}

/* ── Alert boxes ──────────────────────────────── */
.alert-box {
    padding: 0.625rem 0.875rem;
    border-radius: var(--r-md); margin: 0.5rem 0;
    border-left: 3px solid; font-size: 0.8rem; line-height: 1.55;
}
.alert-info     { background: rgba(56,189,248,0.07);  border-color: var(--sky);     color: #7dd3fc; }
.alert-warning  { background: rgba(245,158,11,0.08);  border-color: var(--amber);   color: #fde68a; }
.alert-danger   { background: rgba(239,68,68,0.08);   border-color: var(--rose);    color: #fca5a5; }
.alert-success  { background: rgba(52,211,153,0.08);  border-color: var(--emerald); color: #6ee7b7; }
.alert-critical { background: rgba(239,68,68,0.10);   border-color: var(--rose);    color: #fca5a5; }

/* ── Badges ───────────────────────────────────── */
.badge {
    display: inline-flex; align-items: center;
    padding: 2px 8px; border-radius: 999px;
    font-size: 0.62rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.badge-teal   { background: var(--teal-dim);                  color: var(--teal-lt); }
.badge-amber  { background: var(--amber-dim);                 color: #fde68a; }
.badge-green  { background: rgba(34,197,94,0.12);             color: #86efac; }
.badge-rose   { background: rgba(239,68,68,0.12);             color: #fca5a5; }
.badge-sky    { background: rgba(56,189,248,0.12);            color: #7dd3fc; }
.badge-violet { background: rgba(167,139,250,0.12);           color: #c4b5fd; }
.badge-indigo { background: rgba(99,102,241,0.12);            color: #a5b4fc; }
.badge-orange { background: rgba(249,115,22,0.12);            color: #fdba74; }

/* ── Chat bubbles ─────────────────────────────── */
.bubble-user { display:flex; justify-content:flex-end; margin:0.35rem 0; }
.bubble-user > div {
    background: var(--teal-dim); border:1px solid rgba(20,184,166,0.2);
    border-radius:12px 12px 2px 12px; padding:0.55rem 0.8rem;
    max-width:68%; font-size:0.81rem; line-height:1.55;
}
.bubble-ai { display:flex; justify-content:flex-start; margin:0.35rem 0; }
.bubble-ai > div {
    background: var(--s2); border:1px solid var(--b1);
    border-radius:12px 12px 12px 2px; padding:0.55rem 0.8rem;
    max-width:72%; font-size:0.81rem; line-height:1.6;
}

/* ── Insight card ─────────────────────────────── */
.insight-card {
    background: var(--s1); border:1px solid var(--b1);
    border-radius: var(--r-lg); padding:0.875rem; height:100%;
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
}
.insight-card:hover { border-color: var(--b2); background: var(--s2); transform: translateY(-2px); }
.insight-icon  { font-size:1.4rem; margin-bottom:0.4rem; }
.insight-title { font-size:0.83rem; font-weight:600; margin-bottom:0.3rem; color:var(--t1); }
.insight-body  { font-size:0.76rem; color:var(--t2); line-height:1.55; }
.insight-impact{ font-size:0.7rem; color:var(--emerald); font-weight:600; margin-top:0.4rem; }

/* ── What-if card ─────────────────────────────── */
.whatif-card { background:var(--s1); border:1px solid var(--b1); border-radius:var(--r-md); padding:0.7rem 0.875rem; margin-bottom:0.4rem; transition: border-color 0.15s; }
.whatif-card:hover { border-color:var(--b2); }
.delta-pos { color:#f97316; font-weight:600; font-size:0.8rem; }
.delta-neg { color:var(--emerald); font-weight:600; font-size:0.8rem; }

/* ── Progress ─────────────────────────────────── */
.progress-track { background:rgba(255,255,255,0.06); border-radius:999px; overflow:hidden; height:5px; }
.progress-fill  { background:var(--teal); height:100%; border-radius:999px; transition:width 0.5s ease; }

/* ── Wearable tiles ───────────────────────────── */
.wear-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; }
.wear-tile {
    background:var(--s2); border:1px solid var(--b1); border-radius:var(--r-md);
    padding:0.625rem 0.75rem; transition: border-color 0.15s;
}
.wear-tile:hover { border-color: var(--b2); }
.wear-val { font-family: 'Syne', sans-serif; font-size:1.1rem; font-weight:800; color:var(--t1); line-height:1.1; }
.wear-lbl { font-size:0.64rem; color:var(--t3); margin-top:2px; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; }

/* ── Login hero ───────────────────────────────── */
.login-hero {
    background: linear-gradient(150deg, #0a1525 0%, #0d2040 55%, #0a1525 100%);
    border: 1px solid var(--b1); border-radius: var(--r-xl); padding: 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.login-hero::before {
    content: '';
    position: absolute; top:-80px; right:-80px;
    width:250px; height:250px;
    background: radial-gradient(circle, rgba(20,184,166,0.1) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.login-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem; font-weight: 800; margin: 0 0 0.25rem; letter-spacing: -0.03em; color: var(--t1);
}
.login-hero h1 em { font-style: normal; color: var(--teal-lt); }
.login-hero p { font-size: 0.82rem; color: var(--t2); margin: 0; line-height: 1.6; }
.stat-box {
    background:var(--s2); border:1px solid var(--b1); border-radius:var(--r-md);
    padding:0.75rem 1rem; display:flex; align-items:center; gap:0.75rem;
    transition: border-color 0.15s, transform 0.15s;
}
.stat-box:hover { border-color: var(--b2); transform: translateY(-1px); }
.stat-num { font-family: 'Syne', sans-serif; font-size:1.25rem; font-weight:800; color:var(--teal-lt); line-height:1; }
.stat-lbl { font-size:0.72rem; color:var(--t2); }

/* ── Trow ─────────────────────────────────────── */
.trow { display:flex; align-items:center; padding:0.45rem 0; border-bottom:1px solid var(--b1); font-size:0.8rem; }
.trow:last-child { border-bottom:none; }

/* ── Buttons ──────────────────────────────────── */
.stButton button {
    border-radius:var(--r-md) !important; font-family:'Inter',sans-serif !important;
    font-weight:500 !important; font-size:0.8rem !important; transition:all 0.15s !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #14b8a6, #0e9e8c) !important;
    border:none !important; color:#fff !important; font-weight:600 !important;
    box-shadow:0 2px 8px rgba(20,184,166,0.25) !important;
}
.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0e9e8c, #0d8f7f) !important;
    box-shadow:0 4px 14px rgba(20,184,166,0.35) !important; transform:translateY(-1px) !important;
}
.stButton button[kind="secondary"] {
    background:var(--s1) !important; border:1px solid var(--b1) !important; color:var(--t2) !important;
}
.stButton button[kind="secondary"]:hover {
    background:var(--s2) !important; border-color:var(--b2) !important; color:var(--t1) !important;
}

/* ── Inputs ───────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background:var(--s1) !important; border:1px solid var(--b1) !important;
    border-radius:var(--r-md) !important; color:var(--t1) !important;
    font-family:'Inter',sans-serif !important; font-size:0.8rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color:var(--teal) !important; box-shadow:0 0 0 3px rgba(20,184,166,0.1) !important;
}
[data-testid="stWidgetLabel"] p, label { font-size:0.76rem !important; color:var(--t2) !important; }

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--b1) !important; gap:0 !important; padding:0 !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--t2) !important; border-radius:0 !important; font-size:0.8rem !important; font-weight:500 !important; padding:0.5rem 1rem !important; border-bottom:2px solid transparent !important; margin-bottom:-1px !important; transition:color 0.15s !important; }
.stTabs [data-baseweb="tab"]:hover { color:var(--t1) !important; }
.stTabs [aria-selected="true"] { color:var(--t1) !important; border-bottom-color:var(--teal) !important; background:transparent !important; }

/* ── DataFrames ───────────────────────────────── */
.stDataFrame { border:1px solid var(--b1) !important; border-radius:var(--r-lg) !important; overflow:hidden !important; }

/* ── Expanders ────────────────────────────────── */
.streamlit-expanderHeader { background:var(--s1) !important; border:1px solid var(--b1) !important; border-radius:var(--r-md) !important; font-size:0.8rem !important; font-weight:500 !important; transition: background 0.15s !important; }
.streamlit-expanderHeader:hover { background:var(--s2) !important; }

/* ── Divider ──────────────────────────────────── */
.divider { border:none; border-top:1px solid var(--b1); margin:0.875rem 0; }
.divider::after { display:none; }
hr.divider { border:none; border-top:1px solid var(--b1); margin:1rem 0; }

/* ── Scrollbar ────────────────────────────────── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(20,184,166,0.2); border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:rgba(20,184,166,0.35); }

/* ── Spinner ──────────────────────────────────── */
.stSpinner > div { border-top-color:var(--teal) !important; }

/* ── Footer ───────────────────────────────────── */
.site-footer {
    position:fixed; bottom:0; left:0; right:0; height:36px;
    background: linear-gradient(180deg, rgba(10,21,37,0) 0%, #0a1525 100%);
    border-top:1px solid var(--b1);
    display:flex; align-items:center; justify-content:center;
    font-size:0.68rem; color:var(--t3); z-index:99; gap:1rem;
    backdrop-filter: blur(8px);
}
.site-footer a { color:var(--t3); text-decoration:none; transition:color 0.15s; }
.site-footer a:hover { color:var(--t2); }

/* ── Mono ─────────────────────────────────────── */
.mono { font-family:'JetBrains Mono',monospace; font-size:0.78rem; }

/* ── Feature grid (login) ─────────────────────── */
.feat-item { font-size:0.78rem; color:var(--t2); padding:5px 0; border-bottom:1px solid var(--b1); display:flex; align-items:center; gap:6px; }
.feat-item:last-child { border-bottom:none; }
.feat-dot { width:6px; height:6px; border-radius:50%; background:var(--teal); flex-shrink:0; animation: pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.6;transform:scale(0.8)} }

/* ── Image section ────────────────────────────── */
.hero-img { border-radius:var(--r-xl); overflow:hidden; border:1px solid var(--b1); }
.section-img { width:100%; border-radius:var(--r-lg); border:1px solid var(--b1); display:block; }
.img-caption { font-size:0.7rem; color:var(--t3); margin-top:4px; }

/* ── Research stat card ───────────────────────── */
.res-stat {
    background:var(--s1); border:1px solid var(--b1); border-radius:var(--r-lg);
    padding:1rem; text-align:center; transition: border-color 0.15s, transform 0.15s;
}
.res-stat:hover { border-color: var(--b2); transform: translateY(-2px); }
.res-stat-num { font-family: 'Syne', sans-serif; font-size:1.65rem; font-weight:800; color:var(--teal-lt); line-height:1.1; letter-spacing:-0.025em; }
.res-stat-lbl { font-size:0.7rem; color:var(--t3); margin-top:3px; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; }

/* ── Hide nonessential Streamlit chrome ───────── */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }

/* ── Animations ───────────────────────────────── */
@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.fade-up { animation:fadeUp 0.25s ease both; }

@keyframes slideIn { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
.slide-in { animation:slideIn 0.2s ease both; }

@keyframes heartbeat {
    0%{transform:scale(1)} 14%{transform:scale(1.08)}
    28%{transform:scale(1)} 42%{transform:scale(1.05)}
    70%{transform:scale(1)} 100%{transform:scale(1)}
}
.heart-beat { animation:heartbeat 1.6s ease infinite; display:inline-block; }

/* ── Select slider ────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
}

/* ── Metric ───────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

/* ── Alert box (new style) ────────────────────── */
.alert-box {
    padding: 0.625rem 0.875rem;
    border-radius: var(--r-md);
    margin: 0.5rem 0;
    border-left: 3px solid;
    font-size: 0.8rem;
    line-height: 1.55;
}
</style>
"""

def risk_label(score):
    if score < 25:   return 'Low'
    elif score < 50: return 'Moderate'
    elif score < 75: return 'High'
    else:            return 'Critical'

def risk_color(label):
    return {'Low':'#22c55e','Moderate':'#f59e0b','High':'#f97316','Critical':'#ef4444'}.get(label,'#8ea3b8')

def risk_icon(label):
    return {'Low':'💚','Moderate':'💛','High':'🟠','Critical':'🔴'}.get(label,'⚪')

def kpi_card(icon, title, value, sub, color='#e4ecf4', badge=None):
    bg_map = {
        '#22c55e':'rgba(34,197,94,0.1)',   '#f59e0b':'rgba(245,158,11,0.1)',
        '#f97316':'rgba(249,115,22,0.1)',   '#ef4444':'rgba(239,68,68,0.1)',
        '#14b8a6':'rgba(20,184,166,0.1)',   '#38bdf8':'rgba(56,189,248,0.1)',
        '#0ea5e9':'rgba(14,165,233,0.1)',   '#6366f1':'rgba(99,102,241,0.1)',
        '#10b981':'rgba(16,185,129,0.1)',   '#F59E0B':'rgba(245,158,11,0.1)',
        '#EF4444':'rgba(239,68,68,0.1)',    '#3B82F6':'rgba(59,130,246,0.1)',
        '#a78bfa':'rgba(167,139,250,0.1)',
    }
    icon_bg = bg_map.get(color, 'rgba(255,255,255,0.06)')
    badge_html = f'<span class="badge badge-teal" style="margin-left:5px">{badge}</span>' if badge else ''
    return f"""
    <div class="kpi fade-up">
        <div class="kpi-icon" style="background:{icon_bg};color:{color}">{icon}</div>
        <div>
            <div class="kpi-label">{title}{badge_html}</div>
            <div class="kpi-val" style="color:{color}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    </div>"""

def section_heading(icon, title, sub=''):
    sub_html = f'<p class="page-sub">{sub}</p>' if sub else ''
    return f'<div class="page-header"><h1 class="page-title"><span class="heart-beat">{icon}</span>&nbsp; {title}</h1>{sub_html}</div>'