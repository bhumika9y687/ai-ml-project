"""CardioVue AI patient cardiovascular risk assessment."""

from datetime import datetime
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.database import (
    authenticate,
    get_health_records,
    init_db,
    insert_health_record,
    register_user,
)
from utils.advanced_ml import render_shap_waterfall
from utils.chatbot import get_ai_response
from utils.ml_engine import predict_risk
from utils.theme import CSS, PLOT_LAYOUT, kpi_card, risk_color, section_heading

st.set_page_config(
    page_title="CardioVue AI | Heart Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)
init_db()


def show_login():
    st.markdown(section_heading("🫀", "CardioVue AI", "Patient cardiovascular risk assessment"), unsafe_allow_html=True)
    st.info("This academic project is for demonstration and is not a medical diagnosis.")

    sign_in, create_account = st.tabs(["Sign In", "Create Patient Account"])
    with sign_in:
        with st.form("patient_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            user = authenticate(username.strip(), password)
            if user and user.get("role") == "patient":
                st.session_state.update(
                    logged_in=True,
                    username=user["username"],
                    user=user,
                    page="dashboard",
                    last_prediction=None,
                )
                st.rerun()
            st.error("Invalid credentials or this account is not a patient account.")

    with create_account:
        with st.form("patient_registration"):
            username = st.text_input("Username")
            name = st.text_input("Full name")
            email = st.text_input("Email")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if not all((username.strip(), name.strip(), email.strip(), password)):
                st.warning("Please complete all fields.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, message = register_user(
                    username.strip(),
                    password,
                    name.strip(),
                    email.strip(),
                    {"age": int(age)},
                )
                (st.success if ok else st.error)(message)


def show_sidebar():
    st.sidebar.markdown("## 🫀 CardioVue AI")
    st.sidebar.caption(f"Signed in as patient: {st.session_state.user['name']}")
    preview_role = st.sidebar.selectbox(
        "Preview interface",
        ["Patient", "Doctor", "Researcher"],
        index=["patient", "doctor", "researcher"].index(st.session_state.get("preview_role", "patient")),
        key="preview_role_select",
        help="Doctor and researcher areas are static UI previews only.",
    ).lower()
    st.session_state.preview_role = preview_role

    nav_by_role = {
        "patient": {
            "dashboard": "Dashboard",
            "risk_assessment": "Risk Assessment",
            "history": "Assessment History",
            "appointments_preview": "Appointments",
            "goals_preview": "Health Goals",
            "notifications_preview": "Notifications",
            "profile_preview": "Profile",
            "blood_tests_preview": "Blood Tests & Uploads",
            "ai_assistant_preview": "AI Assistant",
            "what_if_preview": "What-If Planner",
            "ecg_preview": "ECG Viewer",
        },
        "doctor": {
            "doctor_dashboard_preview": "Dashboard",
            "doctor_patients_preview": "Patients",
            "doctor_risk_preview": "Risk Analysis",
            "doctor_appointments_preview": "Appointments",
            "doctor_telemedicine_preview": "Telemedicine",
            "doctor_guidelines_preview": "Guidelines",
            "doctor_analytics_preview": "Analytics",
        },
        "researcher": {
            "research_dashboard_preview": "Dashboard",
            "research_hub_preview": "Research Hub",
            "literature_preview": "Literature Review",
            "dataset_preview": "Dataset Explorer",
            "research_analytics_preview": "Analytics",
            "model_lab_preview": "Model Lab",
            "experiments_preview": "Experiments",
            "collaboration_preview": "Collaboration",
        },
    }
    pages = nav_by_role[preview_role]
    if st.session_state.page not in pages:
        st.session_state.page = next(iter(pages))
    for page, label in pages.items():
        if st.sidebar.button(
            label,
            key=f"nav_{page}",
            type="primary" if st.session_state.page == page else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = page
            st.rerun()
    if st.sidebar.button("Sign out", use_container_width=True):
        for key in ("logged_in", "username", "user", "page", "last_prediction", "preview_role"):
            st.session_state.pop(key, None)
        st.rerun()


def preview_notice():
    st.info("UI preview only — controls and sample information are not connected to application logic or saved data.")


def show_patient_extras(page):
    titles = {
        "appointments_preview": ("Appointments", "Upcoming and previous care visits"),
        "goals_preview": ("Health Goals", "Personal cardiovascular health milestones"),
        "notifications_preview": ("Notifications", "Reminders and health updates"),
        "profile_preview": ("Patient Profile", "Personal details and preferences"),
        "blood_tests_preview": ("Blood Tests & Uploads", "Laboratory results and medical documents"),
    }
    title, subtitle = titles[page]
    st.markdown(section_heading("◈", title, subtitle), unsafe_allow_html=True)
    preview_notice()

    if page == "appointments_preview":
        st.tabs(["Upcoming", "Book New", "History"])
        st.dataframe(
            pd.DataFrame(
                [
                    {"Date": "Sample date", "Time": "10:30 AM", "Clinician": "Sample cardiologist", "Visit type": "Follow-up", "Status": "Preview"},
                    {"Date": "Sample date", "Time": "2:00 PM", "Clinician": "Sample clinician", "Visit type": "Risk review", "Status": "Preview"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.button("Request appointment", disabled=True)
    elif page == "goals_preview":
        for label, progress in [("Weekly activity", 0.55), ("Blood pressure tracking", 0.7), ("Healthy habits", 0.35)]:
            st.markdown(f"**{label}**")
            st.progress(progress)
        st.text_input("New goal", placeholder="Goal name", disabled=True)
        st.button("Add goal", disabled=True)
    elif page == "notifications_preview":
        for message in [
            "Sample: Your next health review is approaching.",
            "Sample: Remember to record your blood pressure.",
            "Sample: Your assessment history is available.",
        ]:
            st.info(message)
        st.button("Mark all as read", disabled=True)
    elif page == "profile_preview":
        with st.form("profile_preview_form"):
            st.text_input("Full name", value=st.session_state.user.get("name", ""), disabled=True)
            st.text_input("Email", value="sample@example.com", disabled=True)
            st.number_input("Age", min_value=18, max_value=100, value=45, disabled=True)
            st.form_submit_button("Save profile", disabled=True)
    else:
        st.subheader("Sample laboratory results")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Test": "HDL cholesterol", "Sample result": "52 mg/dL"},
                    {"Test": "LDL cholesterol", "Sample result": "118 mg/dL"},
                    {"Test": "HbA1c", "Sample result": "5.4%"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.file_uploader("Choose a report (preview only)", disabled=True)


def show_ai_assistant():
    st.markdown(
        section_heading("◉", "AI Health Assistant", "Groq-powered answers to general heart-health questions"),
        unsafe_allow_html=True,
    )
    st.info(
        "This assistant uses Groq's cloud API. Your message, recent chat, and "
        "latest saved assessment are sent to Groq. Do not share information you "
        "do not want sent to a cloud service. It cannot diagnose or recommend treatment."
    )

    chat_key = f"chat_messages_{st.session_state.username}"
    messages = st.session_state.setdefault(
        chat_key,
        [
            {
                "role": "assistant",
                "content": (
                    "Hello! I can help with general heart-health information and explain "
                    "how this app displays assessment results. What would you like to know?"
                ),
            }
        ],
    )

    if st.button("Clear conversation", key="clear_health_chat"):
        st.session_state[chat_key] = []
        messages = st.session_state[chat_key]

    if not messages:
        with st.chat_message("assistant"):
            st.write("Conversation cleared. Ask a general heart-health question to begin.")

    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a general heart-health question")
    if prompt:
        latest_records = get_health_records(st.session_state.username, limit=1)
        latest_record = latest_records[0] if latest_records else None
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.spinner("Generating a response with Groq..."):
                response = get_ai_response(prompt, messages[:-1], latest_record)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)


def show_what_if_preview():
    st.markdown(section_heading("⊙", "What-If Planner", "Explore how a scenario-planning interface could look"), unsafe_allow_html=True)
    preview_notice()
    col1, col2 = st.columns(2)
    with col1:
        st.slider("BMI", 16.0, 50.0, 27.0, disabled=True)
        st.slider("Systolic blood pressure", 80, 220, 128, disabled=True)
        st.toggle("Current smoker", disabled=True)
        st.toggle("Regular physical activity", disabled=True)
    with col2:
        st.metric("Illustrative risk score", "—", help="No risk calculation runs on this preview page.")
        st.caption("Scenario output would appear here when this feature is implemented.")


def show_ecg_preview():
    st.markdown(section_heading("〰", "ECG Viewer", "Static interface preview with a synthetic waveform"), unsafe_allow_html=True)
    preview_notice()
    times = [index / 100 for index in range(600)]
    signal = [
        0.04 * math.sin(2 * math.pi * 1.2 * time)
        + 0.75 * math.exp(-((time % 0.83 - 0.35) ** 2) / 0.0015)
        - 0.18 * math.exp(-((time % 0.83 - 0.30) ** 2) / 0.001)
        for time in times
    ]
    fig = go.Figure(go.Scatter(x=times, y=signal, mode="lines", name="Synthetic sample"))
    fig.update_layout(**PLOT_LAYOUT, title="Synthetic ECG illustration (not a patient recording)", xaxis_title="Time (seconds)", yaxis_title="Relative signal")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    metric_cols = st.columns(3)
    metric_cols[0].metric("Heart rate", "—")
    metric_cols[1].metric("Rhythm", "Preview")
    metric_cols[2].metric("Analysis", "Not connected")


def show_role_preview(page):
    preview_notice()
    if page.endswith("dashboard_preview"):
        st.markdown(section_heading("📊", "Workspace Dashboard", "Static sample overview"), unsafe_allow_html=True)
        cols = st.columns(3)
        for col, title, value in zip(cols, ["Assessments", "Active items", "Reports"], ["—", "—", "—"]):
            col.metric(title, value)
        st.bar_chart(pd.DataFrame({"Sample": [4, 7, 5, 8, 6]}))
    elif page in {"doctor_patients_preview", "doctor_risk_preview"}:
        st.markdown(section_heading("◎", "Patient Overview" if page == "doctor_patients_preview" else "Risk Analysis", "Sample clinician workspace"), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Patient": "Sample patient A", "Latest assessment": "Preview", "Risk category": "—"},
                    {"Patient": "Sample patient B", "Latest assessment": "Preview", "Risk category": "—"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.button("Open patient record", disabled=True)
    elif page == "doctor_appointments_preview":
        st.markdown(section_heading("◫", "Appointments", "Clinician schedule preview"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Time": "10:30 AM", "Patient": "Sample patient", "Visit": "Consultation", "Status": "Preview"}]), use_container_width=True, hide_index=True)
        st.button("Update appointment", disabled=True)
    elif page == "doctor_telemedicine_preview":
        st.markdown(section_heading("◈", "Telemedicine", "Chat and consultation interface preview"), unsafe_allow_html=True)
        st.tabs(["Secure Chat", "Video Consultation"])
        st.text_area("Message", placeholder="Messaging is not connected.", disabled=True)
        st.button("Start consultation", disabled=True)
    elif page == "doctor_guidelines_preview":
        st.markdown(section_heading("📋", "Clinical Guidelines", "Reference interface preview"), unsafe_allow_html=True)
        st.tabs(["Blood Pressure", "Cholesterol", "Diabetes", "Lifestyle"])
        st.dataframe(pd.DataFrame([{"Category": "Sample category", "Reference": "Guideline content placeholder"}]), use_container_width=True, hide_index=True)
    elif page == "doctor_analytics_preview" or page == "research_analytics_preview":
        st.markdown(section_heading("📈", "Analytics", "Static chart preview"), unsafe_allow_html=True)
        st.line_chart(pd.DataFrame({"Sample metric": [3, 5, 4, 7, 6]}))
    elif page == "research_hub_preview":
        st.markdown(section_heading("🔬", "Research Hub", "Research workspace interface preview"), unsafe_allow_html=True)
        st.tabs(["Papers", "Research Areas", "Citations"])
        st.dataframe(pd.DataFrame([{"Title": "Sample cardiovascular ML paper", "Year": "—", "Status": "Preview"}]), use_container_width=True, hide_index=True)
    elif page == "literature_preview":
        st.markdown(section_heading("📚", "Literature Review", "Search and filter interface preview"), unsafe_allow_html=True)
        st.text_input("Search papers", disabled=True)
        st.selectbox("Research area", ["All areas"], disabled=True)
        st.dataframe(pd.DataFrame([{"Paper": "Sample reference", "Finding": "Placeholder", "Year": "—"}]), use_container_width=True, hide_index=True)
    elif page == "dataset_preview":
        st.markdown(section_heading("🗂️", "Dataset Explorer", "Dataset browsing interface preview"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Feature": "Age group", "Type": "Numeric", "Missing": "—"}, {"Feature": "Blood pressure", "Type": "Categorical", "Missing": "—"}]), use_container_width=True, hide_index=True)
        st.button("Load dataset", disabled=True)
    elif page == "model_lab_preview":
        st.markdown(section_heading("🧠", "Model Laboratory", "Model comparison interface preview"), unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Illustrative score": [0.82, 0.88, 0.91]}, index=["Model A", "Model B", "Model C"]))
        st.button("Run comparison", disabled=True)
    elif page == "experiments_preview":
        st.markdown(section_heading("🧪", "Experiments", "Experiment log interface preview"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Experiment": "Sample baseline", "Status": "Preview", "Metric": "—"}]), use_container_width=True, hide_index=True)
        st.button("Create experiment", disabled=True)
    elif page == "collaboration_preview":
        st.markdown(section_heading("🤝", "Collaboration", "Team workspace interface preview"), unsafe_allow_html=True)
        st.columns(2)[0].markdown("**Team activity**\n\nSample activity item\n\nSample shared note")
        st.columns(2)[1].markdown("**Team members**\n\nMember A · Preview\n\nMember B · Preview")


def show_dashboard():
    records = get_health_records(st.session_state.username, limit=24)
    latest = records[0] if records else {}
    first_name = st.session_state.user["name"].split()[0]
    st.markdown(
        section_heading("🫀", f"Welcome, {first_name}", "Your cardiovascular assessment overview"),
        unsafe_allow_html=True,
    )

    if records:
        score = latest.get("risk_score")
        label = latest.get("risk_label", "Not available")
        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                kpi_card("🫀", "Latest Risk Score", f"{score}%", label, risk_color(label)),
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                kpi_card("💉", "Blood Pressure", f"{latest.get('bp_systolic', '--')}/{latest.get('bp_diastolic', '--')}", "mmHg", "#0ea5e9"),
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                kpi_card("🧪", "Cholesterol", str(latest.get("cholesterol", "--")), "mg/dL", "#f59e0b"),
                unsafe_allow_html=True,
            )

        if len(records) > 1:
            trend = list(reversed(records))
            fig = go.Figure(
                go.Scatter(
                    x=[record["date"] for record in trend],
                    y=[record["risk_score"] for record in trend],
                    mode="lines+markers",
                    name="Risk score",
                    line=dict(color="#14b8a6", width=2),
                )
            )
            fig.update_layout(**PLOT_LAYOUT, title="Assessment History", yaxis_title="Risk score (%)", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No assessments yet. Start a risk assessment to see your results here.")

    if st.button("Start Risk Assessment", type="primary"):
        st.session_state.page = "risk_assessment"
        st.rerun()


def show_risk_assessment():
    st.markdown(
        section_heading("◎", "Heart Risk Assessment", "Enter health information to estimate cardiovascular risk"),
        unsafe_allow_html=True,
    )
    st.warning("This estimate is for educational use only. It is not a diagnosis or substitute for professional care.")

    with st.form("risk_assessment"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(st.session_state.user.get("age", 45)))
            bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=27.5, step=0.1)
            cholesterol = st.number_input("Total cholesterol (mg/dL)", min_value=100, max_value=450, value=200)
            systolic = st.number_input("Systolic blood pressure (mmHg)", min_value=80, max_value=220, value=128)
            diastolic = st.number_input("Diastolic blood pressure (mmHg)", min_value=50, max_value=140, value=82)
        with col2:
            heart_rate = st.number_input("Heart rate (bpm)", min_value=40, max_value=180, value=72)
            general_health = st.selectbox(
                "General health",
                [1, 2, 3, 4, 5],
                index=2,
                format_func=lambda value: ["Excellent", "Very Good", "Good", "Fair", "Poor"][value - 1],
            )
            sex = st.selectbox("Sex", [0, 1], format_func=lambda value: "Female" if value == 0 else "Male")
            high_bp = st.checkbox("High blood pressure")
            high_chol = st.checkbox("High cholesterol")
            smoker = st.checkbox("Current smoker")
            diabetes = st.checkbox("Diabetes")
            physical_activity = st.checkbox("Regular physical activity")
            stroke = st.checkbox("Prior stroke / TIA")
        submitted = st.form_submit_button("Calculate Risk", type="primary", use_container_width=True)

    if submitted:
        features = {
            "age": int(age),
            "bmi": float(bmi),
            "gen_health": int(general_health),
            "sex": int(sex),
            "highbp": int(high_bp),
            "highchol": int(high_chol),
            "smoker": int(smoker),
            "diabetes": int(diabetes),
            "phys_activity": int(physical_activity),
            "stroke": int(stroke),
        }
        try:
            with st.spinner("Calculating risk..."):
                result = predict_risk(features)
            if not isinstance(result, dict) or "risk_score" not in result:
                raise ValueError("The prediction engine returned an incomplete result.")
            record = {
                **features,
                "risk_score": result["risk_score"],
                "risk_label": result.get("risk_label", "Unknown"),
                "cholesterol": int(cholesterol),
                "bp_systolic": int(systolic),
                "bp_diastolic": int(diastolic),
                "heart_rate": int(heart_rate),
                "shap_values": result.get("shap_values", {}),
                "model_used": result.get("model_name", "Unknown"),
            }
            insert_health_record(st.session_state.username, record)
            st.session_state.last_prediction = result
            st.session_state.page = "risk_assessment"
            st.rerun()
        except Exception as exc:
            st.error(f"Risk assessment could not be completed: {exc}")

    result = st.session_state.get("last_prediction")
    if result:
        score = float(result["risk_score"])
        label = result.get("risk_label", "Unknown")
        color = result.get("risk_color", risk_color(label))
        st.subheader("Assessment Result")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%", "font": {"color": color}},
                title={"text": f"{label} risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(16,185,129,0.15)"},
                        {"range": [25, 50], "color": "rgba(245,158,11,0.15)"},
                        {"range": [50, 75], "color": "rgba(244,63,94,0.12)"},
                        {"range": [75, 100], "color": "rgba(244,63,94,0.2)"},
                    ],
                },
            )
        )
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=45, b=10))
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Model: {result.get('model_name', 'Unknown')} · The output is an estimate, not medical advice.")

        shap_values = result.get("shap_values", {})
        if isinstance(shap_values, dict) and shap_values:
            st.markdown("#### Feature contributions")
            render_shap_waterfall(shap_values)


def show_history():
    st.markdown(section_heading("📋", "Assessment History", "Previously saved risk assessments"), unsafe_allow_html=True)
    records = get_health_records(st.session_state.username, limit=100)
    if not records:
        st.info("No assessment records yet.")
        return
    columns = ["date", "risk_score", "risk_label", "bp_systolic", "bp_diastolic", "cholesterol", "bmi", "model_used"]
    display = pd.DataFrame(records)
    display = display[[column for column in columns if column in display.columns]]
    display.columns = [column.replace("_", " ").title() for column in display.columns]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download assessment history (CSV)",
        display.to_csv(index=False),
        file_name=f"assessment_history_{st.session_state.username}_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )


def main():
    if not st.session_state.get("logged_in"):
        show_login()
        return

    st.session_state.setdefault("page", "dashboard")
    show_sidebar()
    preview_role = st.session_state.get("preview_role", "patient")
    if preview_role == "patient":
        pages = {
            "dashboard": show_dashboard,
            "risk_assessment": show_risk_assessment,
            "history": show_history,
            "appointments_preview": lambda: show_patient_extras("appointments_preview"),
            "goals_preview": lambda: show_patient_extras("goals_preview"),
            "notifications_preview": lambda: show_patient_extras("notifications_preview"),
            "profile_preview": lambda: show_patient_extras("profile_preview"),
            "blood_tests_preview": lambda: show_patient_extras("blood_tests_preview"),
            "ai_assistant_preview": show_ai_assistant,
            "what_if_preview": show_what_if_preview,
            "ecg_preview": show_ecg_preview,
        }
        fallback = show_dashboard
    else:
        pages = {
            "doctor_dashboard_preview": show_role_preview,
            "doctor_patients_preview": show_role_preview,
            "doctor_risk_preview": show_role_preview,
            "doctor_appointments_preview": show_role_preview,
            "doctor_telemedicine_preview": show_role_preview,
            "doctor_guidelines_preview": show_role_preview,
            "doctor_analytics_preview": show_role_preview,
            "research_dashboard_preview": show_role_preview,
            "research_hub_preview": show_role_preview,
            "literature_preview": show_role_preview,
            "dataset_preview": show_role_preview,
            "research_analytics_preview": show_role_preview,
            "model_lab_preview": show_role_preview,
            "experiments_preview": show_role_preview,
            "collaboration_preview": show_role_preview,
        }
        fallback = show_role_preview

    handler = pages.get(st.session_state.page, fallback)
    if handler == show_role_preview:
        handler(st.session_state.page)
    else:
        handler()


if __name__ == "__main__":
    main()
