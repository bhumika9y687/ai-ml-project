"""Doctor dashboard and patient-management pages."""

from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.database import (
    get_appointments, get_blood_tests, get_doctor_patients, get_health_records,
)
from utils.theme import PLOT_LAYOUT, kpi_card, risk_color, section_heading


# ─── DOCTOR DASHBOARD ──────────────────────────────────────────────────────────
def show_doctor_dashboard():
    username = st.session_state.username
    patients = get_doctor_patients(username)
    appts = get_appointments(username, role='doctor')
    upcoming_appts = [a for a in appts if a['status'] in ['confirmed', 'pending']]

    all_records = []
    for p in patients:
        all_records.extend(get_health_records(p['username'], limit=1))

    high_risk_count = sum(1 for r in all_records if r.get('risk_label') in ['High', 'Critical'])
    critical_count = sum(1 for r in all_records if r.get('risk_label') == 'Critical')

    name_short = st.session_state.user['name'].split()[0]
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">{greeting}, Dr. {name_short}</h1>
            <p class="page-sub">Clinical overview · {len(patients)} patients · {len(upcoming_appts)} upcoming appointments</p>
        </div>
        <div style="font-size:0.72rem;color:var(--t3)">{datetime.now().strftime('%A, %d %b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("👥", "Total Patients", str(len(patients)), "under care", "#0ea5e9"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("⚠️", "High Risk", str(high_risk_count), f"incl. {critical_count} critical", "#f43f5e"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("📅", "Today's Appointments", str(len([a for a in upcoming_appts if a['date'] == datetime.now().strftime('%Y-%m-%d')])), "scheduled", "#10b981"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("📊", "Pending Reviews", str(len([a for a in appts if a['status'] == 'pending'])), "awaiting confirmation", "#f59e0b"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        risk_dist = {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0}
        for r in all_records:
            lbl = r.get('risk_label', 'Low')
            risk_dist[lbl] = risk_dist.get(lbl, 0) + 1
        fig = go.Figure(go.Pie(labels=list(risk_dist.keys()), values=list(risk_dist.values()), hole=0.52, marker=dict(colors=['#10b981','#f59e0b','#f43f5e','#14b8a6'])))
        fig.update_layout(**PLOT_LAYOUT, title='🎯 Patient Risk Distribution', height=300)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown('<div class="card-title">📅 Upcoming Schedule</div>', unsafe_allow_html=True)
        if upcoming_appts:
            for a in upcoming_appts[:5]:
                s_col = {'confirmed': '#10b981', 'pending': '#f59e0b'}.get(a['status'], 'var(--t2)')
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0;border-bottom:1px solid var(--b1)">
                    <div>
                        <div style="font-weight:600;font-size:0.9rem">{a['patient_name']}</div>
                        <div style="font-size:0.78rem;color:var(--t3)">{a['type']}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.84rem;color:var(--t2)">{a['time']}</div>
                        <div style="font-size:0.7rem;color:{s_col}">{a['date']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No appointments scheduled for today.")

# ─── DOCTOR: PATIENT LIST ──────────────────────────────────────────────────────
def show_patients():
    st.markdown(section_heading("👥", "Patient Management"), unsafe_allow_html=True)
    doctor_username = st.session_state.username
    patients = get_doctor_patients(doctor_username)
    patient_by_username = {patient['username']: patient for patient in patients}
    viewing_username = st.session_state.get('viewing_patient')

    if viewing_username:
        patient = patient_by_username.get(viewing_username)
        if not patient:
            st.session_state.viewing_patient = None
            st.warning("That patient is not assigned to your care.")
            return

        if st.button("← Back to patient list"):
            st.session_state.viewing_patient = None
            st.rerun()

        st.markdown(section_heading("👤", patient['name'], f"Patient history · {patient.get('email', 'No email on file')}"), unsafe_allow_html=True)
        history = get_health_records(viewing_username)
        blood_tests = get_blood_tests(viewing_username)
        patient_appts = [appointment for appointment in get_appointments(doctor_username, role='doctor') if appointment['patient_user'] == viewing_username]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Health records", len(history))
        with c2:
            st.metric("Blood tests", len(blood_tests))
        with c3:
            st.metric("Appointments", len(patient_appts))

        if st.button("Run risk assessment", type="primary"):
            st.session_state.selected_patient = viewing_username
            st.session_state.page = 'risk_prediction'
            st.rerun()

        st.markdown('<div class="card-title">Health-record history</div>', unsafe_allow_html=True)
        if history:
            history_df = pd.DataFrame(history)[['date', 'risk_score', 'risk_label', 'bmi', 'bp_systolic', 'bp_diastolic', 'cholesterol', 'heart_rate', 'model_used', 'notes']]
            history_df.columns = ['Date', 'Risk score', 'Risk level', 'BMI', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Heart rate', 'Model', 'Notes']
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No health records have been submitted by this patient.")

        st.markdown('<div class="card-title">Blood-test history</div>', unsafe_allow_html=True)
        if blood_tests:
            blood_df = pd.DataFrame(blood_tests)[['date', 'hdl', 'ldl', 'triglycerides', 'glucose', 'hba1c', 'creatinine', 'notes']]
            blood_df.columns = ['Date', 'HDL', 'LDL', 'Triglycerides', 'Glucose', 'HbA1c', 'Creatinine', 'Notes']
            st.dataframe(blood_df, use_container_width=True, hide_index=True)
        else:
            st.info("No blood-test results have been submitted by this patient.")

        st.markdown('<div class="card-title">Appointment history</div>', unsafe_allow_html=True)
        if patient_appts:
            appointment_df = pd.DataFrame(patient_appts)[['date', 'time', 'type', 'status', 'notes']]
            appointment_df.columns = ['Date', 'Time', 'Type', 'Status', 'Patient notes']
            st.dataframe(appointment_df, use_container_width=True, hide_index=True)
        else:
            st.info("No appointments with this patient yet.")
        return

    search = st.text_input("🔍 Search patients by name", placeholder="Type to filter...")

    if not patients:
        st.info("No patients have booked an appointment with you yet.")
        return

    for p in patients:
        if search and search.lower() not in p.get('name','').lower():
            continue
        records = get_health_records(p['username'], limit=1)
        latest = records[0] if records else {}
        r_score = latest.get('risk_score', '--')
        r_label = latest.get('risk_label', 'No data')
        r_col = risk_color(r_label)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;gap:1rem;align-items:center">
                    <div style="width:42px;height:42px;background:rgba(20,184,166,0.12);border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'DM Sans',sans-serif;font-weight:800;color:#14b8a6;font-size:1rem;flex-shrink:0">{p.get('name','?')[0].upper()}</div>
                    <div style="flex:1">
                        <div style="font-family:'DM Sans',sans-serif;font-weight:700">{p.get('name','')}</div>
                        <div style="font-size:0.82rem;color:var(--t2)">{p.get('email','')}</div>
                        <div style="font-size:0.76rem;color:var(--t3);margin-top:0.15rem">Age {p.get('age','?')} · Joined {p.get('joined','?')} · {len(records)} records</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0">
                        <div style="font-family:'DM Sans',sans-serif;font-weight:800;font-size:1.5rem;color:{r_col}">{r_score}{'%' if r_score != '--' else ''}</div>
                        <div style="font-size:0.78rem;color:{r_col};font-weight:600">{r_label}</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid var(--b1)">
                    <div style="font-size:0.78rem;color:var(--t3)">BP: <span style="color:var(--t2)">{latest.get('bp_systolic','--')}/{latest.get('bp_diastolic','--')}</span></div>
                    <div style="font-size:0.78rem;color:var(--t3)">Chol: <span style="color:var(--t2)">{latest.get('cholesterol','--')} mg/dL</span></div>
                    <div style="font-size:0.78rem;color:var(--t3)">BMI: <span style="color:var(--t2)">{latest.get('bmi','--')}</span></div>
                    <div style="font-size:0.78rem;color:var(--t3)">Model: <span style="color:var(--t2)">{latest.get('model_used','--')}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button("View History", key=f"vp_{p['username']}", use_container_width=True):
                st.session_state.viewing_patient = p['username']
                st.rerun()
            if st.button("Risk Assessment", key=f"ra_{p['username']}", use_container_width=True):
                st.session_state.selected_patient = p['username']
                st.session_state.page = 'risk_prediction'
                st.rerun()
