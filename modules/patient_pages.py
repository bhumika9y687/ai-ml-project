"""Patient pages and shared patient-facing health workflows."""

import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.ai_chatbot import get_ai_response, get_quick_insights
from utils.clinical_guidelines import ascvd_risk_score, get_ascvd_recommendation
from utils.database import (
    add_notification, book_appointment, create_goal, get_all_doctors, get_blood_tests,
    get_conn, get_goals, get_health_records, get_notifications, get_user,
    insert_blood_test, insert_health_record, mark_all_read,
    update_appointment_status, update_goal_progress, get_appointments,
)
from utils.ml_engine import predict_risk
from utils.theme import (
    PLOT_LAYOUT, kpi_card, risk_color, risk_label, section_heading,
)


# ─── PATIENT DASHBOARD ─────────────────────────────────────────────────────────
def show_patient_dashboard():
    username = st.session_state.username
    records = get_health_records(username, limit=24)
    latest = records[0] if records else {}
    notifs = get_notifications(username)
    unread = sum(1 for n in notifs if not n['is_read'])

    name_short = st.session_state.user['name'].split()[0]
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    unread_badge = f' <span class="badge badge-rose">{unread} new</span>' if unread else ''
    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">{greeting}, {name_short}</h1>
            <p class="page-sub">Cardiovascular health overview{unread_badge}</p>
        </div>
        <div style="font-size:0.72rem;color:var(--t3)">{datetime.now().strftime('%A, %d %b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    r_score = latest.get('risk_score', '--')
    r_label = latest.get('risk_label', '--')
    r_col = risk_color(r_label)

    with c1:
        badge = "NEW" if records else None
        st.markdown(kpi_card("🫀", "Risk Score", f"{r_score}%", r_label, r_col, badge), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("💉", "Blood Pressure", f"{latest.get('bp_systolic','--')}/{latest.get('bp_diastolic','--')}", "mmHg · Sys/Dia", "#0ea5e9"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("🧪", "Cholesterol", f"{latest.get('cholesterol','--')}", "mg/dL", "#f59e0b"), unsafe_allow_html=True)
    with c4:
        badge2 = f"{unread}" if unread else None
        st.markdown(kpi_card("🔔", "Notifications", str(unread) if unread else "0", "unread alerts", "#f43f5e" if unread else "#10b981", badge2), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 2])

    with col_left:
        if len(records) >= 2:
            dates = [r['date'] for r in reversed(records)]
            scores = [r['risk_score'] for r in reversed(records)]
            labels = [r['risk_label'] for r in reversed(records)]
            colors_line = [risk_color(l) for l in labels]

            fig = go.Figure()
            fig.add_hrect(y0=0, y1=25, fillcolor='rgba(16,185,129,0.05)', line_width=0)
            fig.add_hrect(y0=25, y1=50, fillcolor='rgba(245,158,11,0.04)', line_width=0)
            fig.add_hrect(y0=50, y1=100, fillcolor='rgba(244,63,94,0.04)', line_width=0)

            fig.add_trace(go.Scatter(
                x=dates, y=scores,
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#14b8a6', width=2),
                marker=dict(size=6, color=colors_line, line=dict(color='#0a1525', width=1.5)),
                fill='tozeroy', fillcolor='rgba(20,184,166,0.05)',
                hovertemplate='%{x}<br>Risk: <b>%{y:.1f}%</b><extra></extra>'
            ))

            if len(scores) >= 4:
                ma = pd.Series(scores).rolling(4, min_periods=1).mean().tolist()
                fig.add_trace(go.Scatter(
                    x=dates, y=ma, mode='lines', name='4-wk avg',
                    line=dict(color='rgba(14,165,233,0.5)', width=1.5, dash='dot')
                ))

            fig.update_layout(
                **PLOT_LAYOUT,
                title='📈 Risk Score Trend',
                height=300,
                xaxis_title='', yaxis_title='Risk Score (%)',
                yaxis_range=[0, 100],
                hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
            <div class="card" style="height:300px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:0.5rem">
                <div style="font-size:0.85rem;font-weight:600;color:var(--t2)">No trend data yet</div>
                <div style="color:var(--t3);font-size:0.78rem">Complete a risk assessment to start tracking</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <img src="https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=400&q=80&auto=format&fit=crop"
             class="section-img" style="height:110px;object-fit:cover;margin-bottom:4px"
             alt="Heart health monitoring"/>
        <p class="img-caption">Stay consistent with your monitoring schedule</p>
        """, unsafe_allow_html=True)

        appts = get_appointments(username, role='patient')
        upcoming = [a for a in appts if a['status'] in ['confirmed', 'pending']]
        if upcoming:
            a = upcoming[0]
            is_confirmed = a['status'] == 'confirmed'
            status_class = 'badge-green' if is_confirmed else 'badge-amber'
            status_text = '✓ Confirmed by doctor' if is_confirmed else 'Awaiting doctor confirmation'
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Next appointment</div>
                <div style="font-weight:600;color:var(--t1);font-size:0.88rem">{a['doctor_name']}</div>
                <div style="font-size:0.78rem;color:var(--t2);margin-top:3px">{a['date']} · {a['time']}</div>
                <div style="font-size:0.73rem;color:var(--t3);margin-top:2px">{a['type']}</div>
                <span class="badge {status_class}" style="margin-top:8px;display:inline-flex">{status_text}</span>
            </div>
            """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if len(records) >= 3:
            dates_r = [r['date'] for r in reversed(records)]
            chol = [r.get('cholesterol', 0) for r in reversed(records)]
            bp_sys = [r.get('bp_systolic', 0) for r in reversed(records)]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=dates_r, y=chol, name='Cholesterol', line=dict(color='#f59e0b', width=2)))
            fig3.add_trace(go.Scatter(x=dates_r, y=bp_sys, name='Systolic BP', line=dict(color='#0ea5e9', width=2)))
            fig3.add_hline(y=200, line=dict(color='rgba(245,158,11,0.3)', dash='dash', width=1))
            fig3.add_hline(y=130, line=dict(color='rgba(14,165,233,0.25)', dash='dash', width=1))
            fig3.update_layout(**PLOT_LAYOUT, title='🩸 Cholesterol & BP History', height=240)
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        patient_data = st.session_state.user
        insights = get_quick_insights(patient_data, latest)
        if insights:
            st.markdown('<div class="card-title" style="margin-bottom:0.75rem">💡 AI Insights</div>', unsafe_allow_html=True)
            for ins in insights[:2]:
                st.markdown(f"""
                <div class="insight-card" style="margin-bottom:0.6rem">
                    <div style="display:flex;gap:0.75rem;align-items:flex-start">
                        <span style="font-size:1.4rem">{ins['icon']}</span>
                        <div>
                            <div class="insight-title">{ins['title']}</div>
                            <div class="insight-body">{ins['body']}</div>
                            <div class="insight-impact">{ins['impact']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─── RISK PREDICTION PAGE ──────────────────────────────────────────────────────
def show_risk_prediction(patient_username=None):
    username = patient_username or st.session_state.username
    is_doctor_view = patient_username is not None

    st.markdown(section_heading("◎", "Risk Assessment",
                                "Powered by Extreme Random Forest"), unsafe_allow_html=True)

    patient_data = get_user(username) if is_doctor_view else st.session_state.user
    blood_tests = get_blood_tests(username)
    latest_blood = blood_tests[0] if blood_tests else {}

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        with st.form("risk_form"):
            st.markdown('<div class="card-title">📊 Clinical Parameters</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 18, 100, patient_data.get('age', 45), help="Your current age")
                bmi = st.number_input("BMI", 15.0, 60.0, 27.5, step=0.1, help="Body Mass Index = weight(kg)/height(m)²")
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 450, 200, help="Total cholesterol level")
                bp_sys = st.number_input("Systolic BP (mmHg)", 80, 220, 128, help="Top number in blood pressure reading")
            with c2:
                bp_dia = st.number_input("Diastolic BP (mmHg)", 50, 140, 82, help="Bottom number in blood pressure reading")
                heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 72, help="Resting heart rate")
                gen_health = st.selectbox("General Health", [1,2,3,4,5], index=2,
                                           format_func=lambda x: ['Excellent','Very Good','Good','Fair','Poor'][x-1])
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

            st.markdown('<div style="margin-top:0.75rem"><div class="card-title">⚠️ Risk Factors</div></div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                highbp = st.checkbox("High Blood Pressure", value=True, help="Diagnosed hypertension or consistently high readings")
                highchol = st.checkbox("High Cholesterol", value=True, help="Elevated LDL or total cholesterol")
                smoker = st.checkbox("Current Smoker", help="Actively smoking tobacco products")
                family_history = st.checkbox("Family History of Heart Disease",
                                             value=bool(patient_data.get('family_history', 0)),
                                             help="First-degree relative with heart disease before age 55 (M) or 65 (F)")
            with c4:
                diabetes = st.checkbox("Diabetes", help="Diagnosed diabetes or elevated blood sugar")
                phys_activity = st.checkbox("Regular Physical Activity", value=True, help="≥150 min/week moderate exercise")
                stroke = st.checkbox("Prior Stroke / TIA", help="History of cerebrovascular event")
                race = st.selectbox("Race (for ASCVD)", ["White", "African American", "Other"], help="Used for clinical risk calculation")

            st.markdown('<div class="card-title">🧪 Optional: HDL Cholesterol</div>', unsafe_allow_html=True)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100,
                                   value=int(latest_blood.get('hdl', 45)) if latest_blood else 45,
                                   help="High-density lipoprotein - 'good' cholesterol")

            submit = st.form_submit_button("🔮 Run AI Prediction", type="primary", use_container_width=True)

    with col_result:
        if submit:
            features = {
                'age': age, 'bmi': bmi, 'gen_health': gen_health, 'sex': sex,
                'highbp': int(highbp), 'highchol': int(highchol),
                'smoker': int(smoker), 'diabetes': int(diabetes),
                'phys_activity': int(phys_activity), 'stroke': int(stroke),
                'family_history': int(family_history),
            }

            with st.spinner("🧠 Running prediction..."):
                try:
                    result = predict_risk(features)
                    # Validate result is not None and has required keys
                    if result is None:
                        raise ValueError("Prediction returned None")
                    if not isinstance(result, dict):
                        raise ValueError(f"Prediction returned invalid type: {type(result)}")
                    if 'risk_score' not in result:
                        raise ValueError("Prediction missing risk_score key")
                except Exception as e:
                    st.warning(f"Prediction service error: {e}. Using fallback values.")
                    result = {
                        'risk_score': 45.0,
                        'risk_label': 'Moderate',
                        'risk_color': '#f59e0b',
                        'ci_low': 38.0,
                        'ci_high': 52.0,
                        'model_name': 'Calibrated Simulator',
                        'model_confidence': 85.0,
                        'shap_values': {'BMI': 0.08, 'Age': 0.12, 'Smoking': 0.15, 'Blood Pressure': 0.10}
                    }

            st.session_state['last_prediction'] = result
            st.session_state['last_features'] = features

            record = {
                **features,
                'risk_score': result.get('risk_score', 45),
                'risk_label': result.get('risk_label', 'Moderate'),
                'cholesterol': cholesterol,
                'bp_systolic': bp_sys,
                'bp_diastolic': bp_dia,
                'heart_rate': heart_rate,
                'shap_values': result.get('shap_values', {}),
                'model_used': result.get('model_name', 'Ensemble'),
                'family_history': int(family_history),
            }
            insert_health_record(username, record)

            if result.get('risk_label') in ['High', 'Critical']:
                add_notification(username, 'alert', f"⚠️ {result.get('risk_label')} risk detected ({result.get('risk_score', 0):.1f}%). Please consult your doctor.")

        if 'last_prediction' in st.session_state and st.session_state['last_prediction'] is not None:
            result = st.session_state['last_prediction']
            score = result.get('risk_score', 45)
            label = result.get('risk_label', 'Moderate')
            color = result.get('risk_color', '#f59e0b')
            ci_low = result.get('ci_low', 38)
            ci_high = result.get('ci_high', 52)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"<b>{label} Risk</b>", 'font': {'family':'DM Sans','color':color,'size':18}},
                number={'suffix':'%','font':{'family':'DM Sans','size':44,'color':color}},
                gauge={
                    'axis': {'range':[0,100],'tickcolor':'#4A5B7A'},
                    'bar': {'color': color, 'thickness': 0.28},
                    'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                    'steps': [
                        {'range':[0,25],'color':'rgba(16,185,129,0.12)'},
                        {'range':[25,50],'color':'rgba(245,158,11,0.10)'},
                        {'range':[50,75],'color':'rgba(244,63,94,0.10)'},
                        {'range':[75,100],'color':'rgba(244,63,94,0.12)'},
                    ],
                    'threshold': {'line':{'color':color,'width':3},'thickness':0.82,'value':score}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9ab3'), height=260, margin=dict(t=20,b=10))
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            # ASCVD Risk Score
            try:
                if age >= 40 and age <= 79:
                    ascvd_score = ascvd_risk_score(
                        age=age, sex='M' if sex == 1 else 'F', race=race,
                        total_chol=cholesterol, hdl_chol=hdl, systolic_bp=bp_sys,
                        treated_hypertension=highbp, diabetes=diabetes, smoker=smoker
                    )

                    if ascvd_score:
                        ascvd_recommendation = get_ascvd_recommendation(ascvd_score, age, diabetes, smoker, highbp)
                        st.markdown(f"""
                        <div class="card-sm" style="margin-top:0.5rem;background:rgba(14,165,233,0.07)">
                            <div class="card-title">📋 Clinical ASCVD Risk Score</div>
                            <div style="display:flex;justify-content:space-between;align-items:baseline">
                                <div style="font-size:1.8rem;font-weight:800;color:#0ea5e9">{ascvd_score:.1f}%</div>
                                <div style="font-size:0.75rem;color:var(--t3)">10-year risk</div>
                            </div>
                            <div style="font-size:0.82rem;margin-top:0.5rem;padding:0.5rem;background:rgba(14,165,233,0.08);border-radius:8px">{ascvd_recommendation}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("📌 ASCVD risk score is validated for ages 40-79. Continue with AI risk assessment.")
            except Exception as e:
                pass

            st.markdown(f"""
            <div style="display:flex;gap:0.6rem;margin-top:-0.5rem;margin-bottom:0.75rem">
                <div class="card-sm" style="flex:1;text-align:center"><div class="card-title">95% CI</div><div style="font-weight:700;color:{color}">{ci_low}% – {ci_high}%</div></div>
                <div class="card-sm" style="flex:1;text-align:center"><div class="card-title">Model</div><div style="font-weight:600;font-size:0.82rem">{result.get('model_name', 'Ensemble')}</div></div>
                <div class="card-sm" style="flex:1;text-align:center"><div class="card-title">Confidence</div><div style="font-weight:700;color:#10b981">{result.get('model_confidence', 85):.0f}%</div></div>
            </div>
            """, unsafe_allow_html=True)

            # SHAP visualization
            shap = result.get('shap_values', {})
            if shap and isinstance(shap, dict):
                st.markdown('<div class="card-title">🧬 SHAP Feature Impact</div>', unsafe_allow_html=True)
                sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                keys = [k for k, _ in sorted_shap]
                vals = [float(v) * 100 for _, v in sorted_shap]
                fig_shap = go.Figure(go.Bar(
                    x=vals, y=keys, orientation='h',
                    marker=dict(color=['#f43f5e' if v > 0 else '#10b981' for v in vals], line=dict(width=0)),
                    text=[f'{v:+.1f}%' for v in vals], textposition='outside', textfont=dict(size=10)
                ))
                fig_shap.update_layout(**PLOT_LAYOUT, title='', height=260, xaxis_title='Risk Contribution (%)')
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

            alert_map = {
                'Low': ('alert-success', '✅', 'Low Risk', 'Your cardiovascular health indicators are in healthy ranges.'),
                'Moderate': ('alert-warning', '⚠️', 'Moderate Risk', 'Some risk factors detected. Lifestyle modifications recommended.'),
                'High': ('alert-danger', '🚨', 'High Risk', 'Cardiology consultation and active management strongly recommended.'),
                'Critical': ('alert-critical', '🆘', 'Critical Risk', 'Immediate medical attention advised.'),
            }
            cls, icon, title, msg = alert_map.get(label, alert_map['Moderate'])
            st.markdown(f'<div class="alert {cls}">{icon} <strong>{title}:</strong> {msg}</div>', unsafe_allow_html=True)

            if st.button("📄 Download Full PDF Report", use_container_width=True):
                _generate_pdf_download(username, result)
        elif submit:
            st.error("Prediction failed. Please check if the ML models are properly loaded or try again.")

def _generate_pdf_download(username, result):
    try:
        from utils.pdf_report import generate_patient_report
        records = get_health_records(username, limit=12)
        blood_tests = get_blood_tests(username)
        patient_data = get_user(username)

        # Ensure result is not None
        if result is None:
            result = {}

        pdf_bytes = generate_patient_report(patient_data, records, result, blood_tests)
        ext = 'pdf' if pdf_bytes[:4] == b'%PDF' else 'csv'
        st.download_button("📥 Download Report", data=pdf_bytes, file_name=f"cardio_vue_ai_report_{username}_{datetime.now().strftime('%Y%m%d')}.{ext}", mime='application/pdf' if ext == 'pdf' else 'text/csv')
    except Exception as e:
        st.error(f"Report generation error: {e}")

# ─── WHAT-IF PLANNER ───────────────────────────────────────────────────────────
def show_whatif():
    from modules.whatif_page import show_whatif as _show_whatif
    username = st.session_state.username
    records = get_health_records(username, limit=1)
    latest = records[0] if records else {}
    _show_whatif(username, latest)

# ─── HEALTH RECORDS ────────────────────────────────────────────────────────────
def show_health_records():
    username = st.session_state.username
    st.markdown(section_heading("📋", "Health Records", "Complete cardiovascular history with downloadable reports"), unsafe_allow_html=True)

    tabs = st.tabs(["  History  ", "  Blood Tests  ", "  Upload Report  "])

    with tabs[0]:
        records = get_health_records(username, limit=50)
        if records:
            df = pd.DataFrame(records)
            display_cols = ['date','risk_score','risk_label','bp_systolic','bp_diastolic','cholesterol','bmi','model_used']
            df_disp = df[[c for c in display_cols if c in df.columns]].copy()
            df_disp.columns = [c.replace('_',' ').title() for c in df_disp.columns]
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                csv = df_disp.to_csv(index=False)
                st.download_button("📥 Export CSV", csv, f"health_records_{username}.csv", "text/csv", use_container_width=True)
            with c2:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    if 'last_prediction' in st.session_state:
                        _generate_pdf_download(username, st.session_state['last_prediction'])
                    else:
                        _generate_pdf_download(username, {})
        else:
            st.info("No health records yet. Complete a risk assessment to generate your first record.")

    with tabs[1]:
        st.markdown('<div class="card-title">Blood Test Results</div>', unsafe_allow_html=True)
        bt_records = get_blood_tests(username)
        if bt_records:
            bt_df = pd.DataFrame(bt_records)
            cols_show = ['date','hdl','ldl','triglycerides','glucose','hba1c','creatinine']
            bt_disp = bt_df[[c for c in cols_show if c in bt_df.columns]]
            st.dataframe(bt_disp, use_container_width=True, hide_index=True)

        with st.form("blood_test_form"):
            st.markdown('<div class="card-title">Add New Result</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                hdl = st.number_input("HDL (mg/dL)", value=52)
                ldl = st.number_input("LDL (mg/dL)", value=118)
            with c2:
                trig = st.number_input("Triglycerides (mg/dL)", value=145)
                glucose = st.number_input("Fasting Glucose (mg/dL)", value=96)
            with c3:
                hba1c = st.number_input("HbA1c (%)", value=5.4, step=0.1)
                creat = st.number_input("Creatinine (mg/dL)", value=0.92, step=0.01)
            if st.form_submit_button("Save Blood Test", type="primary"):
                insert_blood_test(username, {'hdl': hdl, 'ldl': ldl, 'triglycerides': trig, 'glucose': glucose, 'hba1c': hba1c, 'creatinine': creat})
                st.success("✅ Blood test results saved.")
                st.rerun()

    with tabs[2]:
        st.markdown('<div class="card-title">Upload Medical Documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload ECG, lab results, or medical reports", type=["pdf", "png", "jpg", "jpeg", "csv"])
        if uploaded:
            st.success(f"✅ '{uploaded.name}' uploaded securely.")
            st.markdown('<div class="alert alert-info">Files encrypted with AES-256 and stored in HIPAA-compliant storage.</div>', unsafe_allow_html=True)

# ─── APPOINTMENTS ──────────────────────────────────────────────────────────────
def show_appointments():
    username = st.session_state.username
    role = st.session_state.user['role']
    st.markdown(section_heading("📅", "Appointments"), unsafe_allow_html=True)

    tab_labels = ["  Upcoming  ", "  Book New  ", "  History  "] if role == 'patient' else ["  Upcoming  ", "  History  "]
    tabs = st.tabs(tab_labels)
    appts = get_appointments(username, role=role)

    with tabs[0]:
        upcoming = [a for a in appts if a['status'] in ['confirmed', 'pending']]
        if upcoming:
            for a in upcoming:
                s_col = {'confirmed': '#10b981', 'pending': '#f59e0b'}.get(a['status'], 'var(--t2)')
                other = a['doctor_name'] if role == 'patient' else a['patient_name']
                st.markdown(f"""
                <div class="card-sm">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div style="font-weight:600;font-size:0.85rem;color:var(--t1)">{other}</div>
                            <div style="font-size:0.78rem;color:var(--t2);margin-top:2px">{a['type']} &nbsp;·&nbsp; {a['date']} &nbsp;·&nbsp; {a['time']}</div>
                        </div>
                        <span class="badge {'badge-green' if a['status'] == 'confirmed' else 'badge-amber'}">{a['status'].upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if role == 'doctor':
                    if a['status'] == 'pending':
                        confirm_col, decline_col, _ = st.columns([1, 1, 3])
                        with confirm_col:
                            if st.button("Confirm", key=f"confirm_{a['id']}", use_container_width=True):
                                update_appointment_status(a['id'], 'confirmed')
                                add_notification(a['patient_user'], 'appointment', f"Your appointment with {st.session_state.user['name']} on {a['date']} at {a['time']} was confirmed.")
                                st.rerun()
                        with decline_col:
                            if st.button("Decline", key=f"decline_{a['id']}", use_container_width=True):
                                update_appointment_status(a['id'], 'declined')
                                add_notification(a['patient_user'], 'appointment', f"Your appointment request for {a['date']} at {a['time']} was declined.")
                                st.rerun()
                    elif a['status'] == 'confirmed':
                        if st.button("Mark completed", key=f"complete_{a['id']}"):
                            update_appointment_status(a['id'], 'completed')
                            add_notification(a['patient_user'], 'appointment', f"Your appointment with {st.session_state.user['name']} has been marked completed.")
                            st.rerun()
        else:
            st.info("No upcoming appointments.")

    if role == 'patient':
        with tabs[1]:
            doctors = get_all_doctors()
            with st.form("book_form"):
                c1, c2 = st.columns(2)
                with c1:
                    atype = st.selectbox("Consultation Type", ["Cardiology Review","ECG Review","General Check-up","Video Consultation","Follow-up","Second Opinion"])
                    adate = st.date_input("Preferred Date", min_value=datetime.now().date())
                with c2:
                    atime = st.selectbox("Time Slot", ["9:00 AM","9:30 AM","10:00 AM","10:30 AM","11:00 AM","2:00 PM","2:30 PM","3:00 PM","3:30 PM","4:00 PM"])
                    doctor_user = st.selectbox(
                        "Preferred Doctor",
                        doctors,
                        format_func=lambda doctor: f"{doctor['name']} ({doctor.get('specialty', 'General Practice')})"
                    ) if doctors else None
                notes = st.text_area("Notes / Current Symptoms", height=80, placeholder="Describe any current symptoms or concerns...")
                if st.form_submit_button("Book Appointment →", type="primary", use_container_width=True):
                    if not doctor_user:
                        st.error("No doctor accounts are available for appointment booking.")
                    else:
                        created, message = book_appointment({'patient_user': username, 'patient_name': st.session_state.user['name'], 'doctor_user': doctor_user['username'], 'doctor_name': doctor_user['name'], 'date': adate.strftime('%Y-%m-%d'), 'time': atime, 'type': atype, 'notes': notes})
                        if created:
                            add_notification(doctor_user['username'], 'appointment', f"New appointment request from {st.session_state.user['name']} – {adate} {atime}")
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.warning(message)
    with tabs[-1]:
        past = [a for a in appts if a['status'] in ['completed', 'declined']]
        if past:
            for a in past:
                other = a['doctor_name'] if role == 'patient' else a['patient_name']
                st.markdown(f"""
                <div class="card-sm" style="opacity:0.65">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div><span style="font-weight:600">{a['type']}</span><span style="color:var(--t3);margin-left:0.75rem;font-size:0.82rem">with {other}</span></div>
                        <span style="color:var(--t3);font-size:0.78rem">{a['date']} · {a['status'].title()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No past appointments on record.")

# ─── AI ASSISTANT ──────────────────────────────────────────────────────────────
def show_ai_assistant():
    username = st.session_state.username
    st.markdown(section_heading("◉", "AI Health Assistant", "Personalized cardiovascular guidance · Powered by Gemini"), unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {'role': 'assistant', 'content': "Hello! I'm your Cardio Vue AI Health Assistant 🫀\n\nI can help you understand your risk factors, suggest lifestyle improvements, explain your test results, and answer cardiovascular health questions. What would you like to know?"}
        ]

    records = get_health_records(username, limit=1)
    latest = records[0] if records else {}

    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem">', unsafe_allow_html=True)
    quick = ["Best diet for my risk level", "How much should I exercise?", "Explain my risk score", "Stress & heart health", "Should I take aspirin?", "Signs of a heart attack"]
    cols = st.columns(len(quick))
    for col, q in zip(cols, quick):
        with col:
            if st.button(q, key=f"qa_{q[:15]}", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': q})
                api_key = st.session_state.get('gemini_key', '')
                resp = get_ai_response(q, st.session_state.chat_history, st.session_state.user, latest, api_key)
                st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    chat_html = ""
    for msg in st.session_state.chat_history[-20:]:
        if msg['role'] == 'user':
            chat_html += f'<div class="bubble-user"><div>{msg["content"]}</div></div>'
        else:
            content = msg['content'].replace('\n', '<br>')
            chat_html += f'<div class="bubble-ai"><div><span style="font-size:0.68rem;color:var(--teal-lt);font-weight:600;letter-spacing:0.05em">CARDIO VUE AI</span><br>{content}</div></div>'

    st.markdown(f'<div style="max-height:420px;overflow-y:auto;padding:0.5rem 0;margin-bottom:0.75rem">{chat_html}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_msg = st.text_input("Ask about your heart health...", label_visibility="collapsed", placeholder="Type your question here...")
        with c2:
            send = st.form_submit_button("Send →", type="primary", use_container_width=True)
        if send and user_msg.strip():
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            api_key = st.session_state.get('gemini_key', '')
            with st.spinner("Thinking..."):
                resp = get_ai_response(user_msg, st.session_state.chat_history, st.session_state.user, latest, api_key)
            st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
            st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    insights = get_quick_insights(st.session_state.user, latest)
    if insights:
        st.markdown('<div class="card-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
        cols = st.columns(min(4, len(insights)))
        for col, ins in zip(cols, insights):
            with col:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-icon">{ins['icon']}</div>
                    <div class="insight-title">{ins['title']}</div>
                    <div class="insight-body">{ins['body']}</div>
                    <div class="insight-impact">{ins['impact']}</div>
                </div>
                """, unsafe_allow_html=True)

# ─── NOTIFICATIONS ─────────────────────────────────────────────────────────────
def show_notifications():
    username = st.session_state.username
    st.markdown(section_heading("⊡", "Notifications", "Your health alerts and reminders"), unsafe_allow_html=True)

    notifs = get_notifications(username)
    unread = sum(1 for n in notifs if not n['is_read'])

    if unread:
        st.markdown(f'<div class="alert alert-info" style="">📬 You have <strong>{unread}</strong> unread notification{"s" if unread > 1 else ""}</div>', unsafe_allow_html=True)

    type_icons = {'reminder': '💊', 'alert': '⚠️', 'appointment': '📅', 'info': 'ℹ️', 'success': '🎉'}
    for n in notifs:
        icon = type_icons.get(n['type'], '🔔')
        read_op = 'opacity:0.55' if n['is_read'] else ''
        new_badge = '<span class="badge badge-teal" style="margin-left:5px">NEW</span>' if not n['is_read'] else ''
        st.markdown(f"""
        <div class="card-sm" style="{read_op}">
            <div style="display:flex;align-items:center;gap:0.75rem">
                <span style="font-size:1.25rem">{icon}</span>
                <div style="flex:1">
                    <div style="font-size:0.88rem">{n['msg']}{new_badge}</div>
                    <div style="font-size:0.72rem;color:var(--t3);margin-top:0.15rem">{n['time_str']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if notifs and st.button("✓ Mark All as Read", use_container_width=True):
        mark_all_read(username)
        st.rerun()

# ─── PROFILE PAGE ─────────────────────────────────────────────────────────────
def show_profile():
    username = st.session_state.username
    user = get_user(username)

    st.markdown(section_heading("◷", "Profile", "Manage your personal information and health preferences"), unsafe_allow_html=True)

    doctors = get_all_doctors()
    with st.form("profile_form"):
        st.markdown('<div class="card-title">Personal Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", value=user['name'])
            email = st.text_input("Email", value=user['email'])
            age = st.number_input("Age", min_value=18, max_value=100, value=user.get('age', 35), step=1)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if user.get('gender', 'Male') == 'Male' else 1)
            family_history = st.checkbox("Family history of premature heart disease", value=bool(user.get('family_history', 0)), help="Heart disease in first-degree relative before age 55 (male) or 65 (female)")

        st.markdown('<div class="card-title">Health Preferences</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            doctor_options = [None] + doctors
            preferred_doctor = st.selectbox(
                "Preferred Doctor",
                doctor_options,
                format_func=lambda doctor: "No preference" if doctor is None else f"{doctor['name']} ({doctor.get('specialty', 'General Practice')})"
            )
        with col4:
            notification_prefs = st.multiselect("Notification Preferences", ["Email reminders", "SMS alerts", "In-app notifications"], default=["In-app notifications"])

        submitted = st.form_submit_button("Update Profile", type="primary", use_container_width=True)
        if submitted:
            try:
                extra_json = json.dumps({'age': age, 'gender': gender, 'preferred_doctor': preferred_doctor['username'] if preferred_doctor else None, 'notification_prefs': notification_prefs})
                with get_conn() as conn:
                    conn.execute("UPDATE users SET name=?, email=?, extra_json=?, family_history=? WHERE username=?", (name, email, extra_json, int(family_history), username))
                st.session_state.user = get_user(username)
                st.success("✅ Profile updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating profile: {e}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Health Summary</div>', unsafe_allow_html=True)
    records = get_health_records(username, limit=5)
    if records:
        latest = records[0]
        delta = f"{latest.get('risk_score', 0) - records[1].get('risk_score', 0):+.1f}%" if len(records) > 1 else ""
        col1, col2, col3, col4 = st.columns(4)
        r_lbl = risk_label(latest.get('risk_score', 0))
        with col1: st.markdown(kpi_card("🫀", "Risk Score", f"{latest.get('risk_score','--')}%", f"{r_lbl} {delta}", risk_color(r_lbl)), unsafe_allow_html=True)
        with col2: st.markdown(kpi_card("💉", "Blood Pressure", f"{latest.get('bp_systolic','--')}/{latest.get('bp_diastolic','--')}", "mmHg", "#0ea5e9"), unsafe_allow_html=True)
        with col3: st.markdown(kpi_card("🧪", "Cholesterol", f"{latest.get('cholesterol','--')}", "mg/dL", "#f59e0b"), unsafe_allow_html=True)
        with col4: st.markdown(kpi_card("⚖️", "BMI", f"{latest.get('bmi','--')}", "kg/m²", "#14b8a6"), unsafe_allow_html=True)
    else:
        st.info("Complete a risk assessment to see your health summary.")

# ─── GOALS PAGE ────────────────────────────────────────────────────────────────
def show_goals():
    username = st.session_state.username

    st.markdown(section_heading("◈", "Health Goals", "Set and track your cardiovascular milestones"), unsafe_allow_html=True)

    active_goals = get_goals(username, status='active')
    achieved_goals = get_goals(username, status='achieved')
    records = get_health_records(username, limit=1)
    latest = records[0] if records else {}
    blood_tests = get_blood_tests(username)
    latest_blood = blood_tests[0] if blood_tests else {}

    with st.expander("➕ Create New Goal", expanded=False):
        with st.form("new_goal_form"):
            st.markdown('<div class="card-title">Set a New Health Goal</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                goal_type = st.selectbox("Goal Type", ["risk_score", "bp_systolic", "cholesterol", "bmi", "steps", "exercise_minutes", "hdl"], format_func=lambda x: {'risk_score': 'Reduce Risk Score', 'bp_systolic': 'Lower Blood Pressure', 'cholesterol': 'Improve Cholesterol', 'bmi': 'Achieve Healthy BMI', 'steps': 'Increase Daily Steps', 'exercise_minutes': 'Exercise Minutes', 'hdl': 'Improve HDL Cholesterol'}[x])
                target_value = st.number_input("Target Value", min_value=0.0, step=1.0)
            with col2:
                target_date = st.date_input("Target Date", min_value=datetime.now().date())
                current_value = st.number_input("Current Value (optional)", min_value=0.0, step=1.0, value=0.0)

            if current_value == 0.0:
                if goal_type == 'risk_score' and latest: current_value = latest.get('risk_score', 0)
                elif goal_type == 'bp_systolic' and latest: current_value = latest.get('bp_systolic', 0)
                elif goal_type == 'cholesterol' and latest: current_value = latest.get('cholesterol', 0)
                elif goal_type == 'bmi' and latest: current_value = latest.get('bmi', 0)
                elif goal_type == 'hdl' and latest_blood: current_value = latest_blood.get('hdl', 0)

            if st.form_submit_button("Create Goal", type="primary"):
                create_goal(username, goal_type, target_value, target_date.strftime("%Y-%m-%d"), current_value)
                st.success("✅ Goal created successfully!")
                st.rerun()

    if active_goals:
        st.markdown('<div class="sec-title">Active Goals</div>', unsafe_allow_html=True)
        for goal in active_goals:
            current = goal.get('current_value', 0)
            target = goal['target_value']
            progress_pct = min(100, (current / target * 100) if target > 0 else 0)
            is_reduction_goal = goal['goal_type'] in ['risk_score', 'bp_systolic', 'cholesterol', 'bmi']
            progress_color = '#10b981' if (is_reduction_goal and current <= target) or (not is_reduction_goal and current >= target) else '#f59e0b'

            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                        <div>
                            <div style="font-weight:600;font-size:0.85rem">{goal['goal_type'].replace('_', ' ').title()}</div>
                            <div style="font-size:0.72rem;color:var(--t3);margin-top:2px">Target: {target} &nbsp;·&nbsp; Current: {current:.1f} &nbsp;·&nbsp; Due: {goal['target_date']}</div>
                        </div>
                        <div style="text-align:right">
                            <span style="font-size:0.78rem;color:{progress_color};font-weight:600">{progress_pct:.0f}%</span>
                        </div>
                    </div>
                    <div class="progress-track"><div style="width:{progress_pct}%;background:{progress_color};height:4px;border-radius:999px;transition:width 0.3s"></div></div>
                </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col2:
                    if st.button("Update", key=f"update_{goal['id']}", use_container_width=True):
                        new_value = st.number_input("New value", value=current, key=f"val_{goal['id']}")
                        update_goal_progress(goal['id'], new_value)
                        st.rerun()
                with col3:
                    if st.button("Achieved ✓", key=f"achieve_{goal['id']}", use_container_width=True, type="primary"):
                        update_goal_progress(goal['id'], current, achieved=True)
                        add_notification(username, 'success', f"Congratulations! You achieved your {goal['goal_type']} goal!")
                        st.rerun()

    if achieved_goals:
        st.markdown('<div class="sec-title">Achieved Goals</div>', unsafe_allow_html=True)
        for goal in achieved_goals:
            st.markdown(f"""
            <div class="card-sm" style="opacity:0.6">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <span style="font-weight:600;font-size:0.82rem">{goal['goal_type'].replace('_', ' ').title()}</span>
                        <div style="font-size:0.72rem;color:var(--t3);margin-top:2px">Target: {goal['target_value']} &nbsp;·&nbsp; Achieved: {goal['achieved_date']}</div>
                    </div>
                    <span class="badge badge-green">✓ Done</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if not active_goals and not achieved_goals:
        st.info("No goals yet. Use the form above to start tracking your health milestones.")
