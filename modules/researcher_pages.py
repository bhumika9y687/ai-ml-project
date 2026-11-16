"""Researcher dashboard, dataset, model, and experiment pages."""

from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.literature_review import show_literature_review
from modules.research_hub import show_research_hub
from utils.database import create_experiment, get_experiments, get_research_records
from utils.theme import PLOT_LAYOUT, kpi_card, section_heading


# ─── MODEL LABORATORY ──────────────────────────────────────────────────────────
def show_model_lab():
    st.markdown(section_heading("🔬", "Model Laboratory", "Experiment with and compare ML models for cardiovascular risk prediction"), unsafe_allow_html=True)

    results_file = Path(__file__).resolve().parent.parent / "results" / "model_comparison.csv"
    if not results_file.exists():
        st.info("No trained-model performance results are available yet. Run model training to generate them.")
        return

    df = pd.read_csv(results_file).rename(columns={
        'F1-Score': 'F1_Score',
        'ROC-AUC': 'ROC_AUC',
    })
    df['Model'] = df['Model'].str.replace('_', ' ', regex=False)
    df['Training_Time_s'] = None
    (tab1,) = st.tabs(["  Performance Comparison  "])

    with tab1:
        metric = st.selectbox("Sort & compare by", ['F1_Score','Accuracy','ROC_AUC','Recall','Precision','Balanced_Accuracy'])
        sorted_df = df.sort_values(metric, ascending=True)
        fig = go.Figure(go.Bar(x=sorted_df[metric], y=sorted_df['Model'], orientation='h', marker=dict(color=sorted_df[metric], colorscale=[[0,'#0ea5e9'],[0.5,'#f59e0b'],[1,'#14b8a6']], showscale=True), text=[f'{v:.3f}' for v in sorted_df[metric]], textposition='outside'))
        fig.update_layout(**PLOT_LAYOUT, title=f'🏆 Model Comparison — {metric}', height=360)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        display_df = df[['Model','Accuracy','Precision','Recall','F1_Score','ROC_AUC','Balanced_Accuracy']].copy()
        display_df.columns = ['Model','Accuracy','Precision','Recall','F1 Score','ROC-AUC','Balanced Acc']
        st.dataframe(display_df.sort_values('F1 Score', ascending=False), use_container_width=True, hide_index=True)

# ─── DATASET EXPLORER ─────────────────────────────────────────────────────────
def show_dataset():
    st.markdown(section_heading("🗂️", "Dataset Explorer", "Access and filter anonymized cardiovascular research datasets"), unsafe_allow_html=True)

    anon = pd.DataFrame(get_research_records())
    if anon.empty:
        st.info("No de-identified patient health records are available yet. Records appear here after patients complete a risk assessment.")
        return

    anon['AgeGroup'] = pd.cut(anon['age'], bins=[0, 30, 45, 60, 75, float('inf')], labels=['18-30', '31-45', '46-60', '61-75', '75+']).astype(str)
    anon['HighRisk'] = anon['risk_label'].isin(['High', 'Critical'])

    c1, c2, c3, c4 = st.columns(4)
    with c1: f_age = st.multiselect("Age Group", anon['AgeGroup'].unique().tolist(), default=anon['AgeGroup'].unique().tolist())
    with c2: f_gender = st.multiselect("Gender", sorted(anon['gender'].dropna().unique().tolist()), default=sorted(anon['gender'].dropna().unique().tolist()))
    with c3: f_hd = st.selectbox("High-risk status", ["All","High / Critical","Low / Moderate"])
    with c4: f_risk = st.selectbox("Risk Level", ["All","Low (<25%)","Moderate (25-50%)","High (50-75%)","Critical (>75%)"])

    filt = anon[anon['AgeGroup'].isin(f_age) & anon['gender'].isin(f_gender)]
    if f_hd == "High / Critical": filt = filt[filt['HighRisk']]
    elif f_hd == "Low / Moderate": filt = filt[~filt['HighRisk']]
    if f_risk == "Low (<25%)": filt = filt[filt['risk_score']<25]
    elif f_risk == "Moderate (25-50%)": filt = filt[(filt['risk_score']>=25)&(filt['risk_score']<50)]
    elif f_risk == "High (50-75%)": filt = filt[(filt['risk_score']>=50)&(filt['risk_score']<75)]
    elif f_risk == "Critical (>75%)": filt = filt[filt['risk_score']>=75]

    st.markdown(f'<div style="font-size:0.82rem;color:var(--t3);margin-bottom:0.5rem">Showing <strong>{len(filt):,}</strong> of {len(anon):,} anonymized records</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(100), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        csv = filt.to_csv(index=False)
        st.download_button("📥 Download Dataset (CSV)", csv, f"cardiovascular_dataset_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    with c2:
        if st.button("📊 Run Statistical Summary", use_container_width=True):
            st.dataframe(filt.describe().round(2), use_container_width=True)

# ─── RESEARCHER DASHBOARD ─────────────────────────────────────────────────────
def show_researcher_dashboard():
    name_short = st.session_state.user['name'].split()[0]
    research_df = pd.DataFrame(get_research_records())
    record_count = len(research_df)
    patient_count = research_df['patient_id'].nunique() if not research_df.empty else 0
    high_risk_count = int(research_df['risk_label'].isin(['High', 'Critical']).sum()) if not research_df.empty else 0
    average_risk = research_df['risk_score'].mean() if not research_df.empty else None
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">{greeting}, Dr. {name_short}</h1>
            <p class="page-sub">Research workspace · de-identified Cardio Vue patient records</p>
        </div>
        <span class="badge badge-amber" data-tooltip="Researcher access level">Researcher Access</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("🗄️", "Health Records", str(record_count), "de-identified", "#14b8a6"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("👥", "Patients", str(patient_count), "with recorded data", "#22c55e"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("⚠️", "High Risk", str(high_risk_count), "high or critical", "#f59e0b"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("📊", "Average Risk", f"{average_risk:.1f}%" if average_risk is not None else "—", "recorded assessments", "#a78bfa"), unsafe_allow_html=True)

    if research_df.empty:
        st.info("No patient assessments are available yet. Patient data will appear after a patient completes a risk assessment.")
    else:
        st.markdown('<div class="sec-title">Latest de-identified patient records</div>', unsafe_allow_html=True)
        preview = research_df[['patient_id', 'date', 'age', 'gender', 'bmi', 'risk_score', 'risk_label']].copy()
        preview.columns = ['Patient ID', 'Date', 'Age', 'Gender', 'BMI', 'Risk Score', 'Risk Level']
        st.dataframe(preview.head(20), use_container_width=True, hide_index=True)

# ─── RESEARCHER: LITERATURE TRACKER ──────────────────────────────────────────
def show_literature():
    show_literature_review()

# ─── RESEARCHER: EXPERIMENT NOTES ─────────────────────────────────────────────
def show_experiments():
    st.markdown(section_heading("🧪", "Experiment Notes", "Track your model experiments, hyperparameters, and results"), unsafe_allow_html=True)

    experiments = get_experiments(st.session_state.username)
    colors_by_status = {"Complete": "#14b8a6", "Running": "#f59e0b", "Failed": "#f43f5e"}
    for experiment in experiments:
        experiment['color'] = colors_by_status.get(experiment['status'], '#0ea5e9')
        experiment['date'] = experiment['created_at']
        experiment['params'] = experiment['parameters']

    complete = [e for e in experiments if e['status'] == 'Complete' and e['f1'] is not None]
    best = max(complete, key=lambda x: x['f1']) if complete else None
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("🧪", "Experiments", str(len(experiments)), "total runs", "#14b8a6"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("✅", "Complete", str(len(complete)), "finished", "#22c55e"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("🏆", "Best F1", f"{best['f1']:.3f}" if best else "--", best['name'] if best else "--", "#f59e0b"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("🔄", "Running", str(sum(1 for e in experiments if e['status']=='Running')), "in progress", "#f97316"), unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:0.5rem">Experiment Log</div>', unsafe_allow_html=True)

    complete_exps = [e for e in experiments if e['f1'] is not None]
    if complete_exps:
        names = [e['name'] for e in complete_exps]
        f1s   = [e['f1']   for e in complete_exps]
        aucs  = [e['auc']  for e in complete_exps]
        colors_exp = [e['color'] for e in complete_exps]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='F1 Score', x=names, y=f1s,
                             marker_color=colors_exp, text=[f'{v:.3f}' for v in f1s],
                             textposition='outside'))
        fig.add_trace(go.Scatter(name='ROC-AUC', x=names, y=aucs,
                                 mode='lines+markers', line=dict(color='#f59e0b', width=2),
                                 marker=dict(size=8)))
        _pl = {k: v for k, v in PLOT_LAYOUT.items() if k != 'yaxis'}
        fig.update_layout(**_pl, title='Experiment Comparison', height=280,
                          yaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.04)'),
                          barmode='group')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    status_cls = {"Complete": "badge-green", "Running": "badge-amber", "Failed": "badge-rose"}
    for e in experiments:
        f1_str  = f"{e['f1']:.3f}"  if e['f1']  else "—"
        auc_str = f"{e['auc']:.3f}" if e['auc'] else "—"
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {e['color']}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <div style="display:flex;align-items:center;gap:0.75rem">
                    <span class="mono" style="color:var(--t3);font-size:0.72rem">{e['id']}</span>
                    <span style="font-weight:600;font-size:0.88rem">{e['name']}</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.75rem">
                    <span style="font-size:0.75rem;color:var(--t3)">{e['date']}</span>
                    <span class="badge {status_cls.get(e['status'],'badge-teal')}">{e['status']}</span>
                </div>
            </div>
            <div style="display:flex;gap:1.5rem;margin-bottom:0.4rem">
                <div><span style="font-size:0.68rem;color:var(--t3)">F1 SCORE</span><br><span style="font-weight:700;color:{e['color']};font-size:0.95rem">{f1_str}</span></div>
                <div><span style="font-size:0.68rem;color:var(--t3)">ROC-AUC</span><br><span style="font-weight:700;color:var(--t2);font-size:0.95rem">{auc_str}</span></div>
                <div style="flex:1"><span style="font-size:0.68rem;color:var(--t3)">PARAMETERS</span><br><span class="mono" style="font-size:0.72rem;color:var(--t2)">{e['params']}</span></div>
            </div>
            <div style="font-size:0.78rem;color:var(--t2);line-height:1.5;border-top:1px solid var(--b1);padding-top:0.4rem">{e['notes']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Log New Experiment</div>', unsafe_allow_html=True)
    with st.expander("+ New experiment entry"):
        with st.form("new_exp"):
            c1, c2 = st.columns(2)
            with c1:
                exp_name   = st.text_input("Experiment Name")
                exp_model  = st.selectbox("Model Type", ["Extreme Random Forest","XGBoost","CatBoost","Neural Network","Stacking Ensemble","Other"])
                exp_f1     = st.number_input("F1 Score", 0.0, 1.0, 0.90, step=0.001, format="%.3f")
            with c2:
                exp_auc    = st.number_input("ROC-AUC", 0.0, 1.0, 0.90, step=0.001, format="%.3f")
                exp_status = st.selectbox("Status", ["Running","Complete","Failed"])
                exp_params = st.text_input("Key Parameters")
            exp_notes = st.text_area("Notes", height=80)
            if st.form_submit_button("Log Experiment →", type="primary"):
                if not exp_name.strip():
                    st.error("Enter an experiment name.")
                else:
                    metrics = {'f1': exp_f1, 'auc': exp_auc} if exp_status == 'Complete' else {'f1': None, 'auc': None}
                    create_experiment(st.session_state.username, {
                        'name': exp_name.strip(),
                        'model_type': exp_model,
                        'status': exp_status,
                        'parameters': exp_params,
                        'notes': exp_notes,
                        **metrics,
                    })
                    st.success(f"Experiment '{exp_name}' logged successfully.")
                    st.rerun()
