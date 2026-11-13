"""
Cardio Vue AI — What-If Scenario Planner Page
The most visually impressive feature: live sliders updating risk score in real-time.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.ml_engine import predict_risk, get_intervention_scenarios, whatif_delta, _risk_label, _risk_color
from utils.theme import PLOT_LAYOUT, section_heading


def show_whatif(username: str, latest_record: dict = None):
    st.markdown(section_heading("🔮", "What-If Scenario Planner",
                                 "Drag sliders to see how lifestyle changes affect your risk score in real-time"),
                unsafe_allow_html=True)

    # Base features from latest record or defaults
    defaults = latest_record or {}

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card-title">Adjust Your Health Parameters</div>', unsafe_allow_html=True)

        age = st.slider("Age (years)", 18, 85,
                        int(defaults.get('age', 45)), key="wi_age")
        bmi = st.slider("BMI", 16.0, 50.0,
                        float(defaults.get('bmi', 27.0)), step=0.1, key="wi_bmi")
        gen_health = st.select_slider(
            "General Health",
            options=[1, 2, 3, 4, 5],
            value=int(defaults.get('gen_health', 3)),
            format_func=lambda x: ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x - 1],
            key="wi_gh"
        )

        st.markdown('<div style="margin-top:0.75rem"><div class="card-title">Risk Factors (Toggle)</div></div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            highbp = st.toggle("High Blood Pressure", value=bool(defaults.get('highbp', 0)), key="wi_hbp")
            highchol = st.toggle("High Cholesterol", value=bool(defaults.get('highchol', 0)), key="wi_hcl")
            smoker = st.toggle("Smoker", value=bool(defaults.get('smoker', 0)), key="wi_smk")
        with c2:
            diabetes = st.toggle("Diabetes", value=bool(defaults.get('diabetes', 0)), key="wi_dia")
            phys = st.toggle("Regular Exercise", value=bool(defaults.get('phys_activity', 0)), key="wi_phy")
            stroke = st.toggle("Prior Stroke", value=bool(defaults.get('stroke', 0)), key="wi_str")

    # Build current feature dict
    current_features = {
        'age': age, 'bmi': bmi, 'gen_health': gen_health,
        'highbp': int(highbp), 'highchol': int(highchol),
        'smoker': int(smoker), 'diabetes': int(diabetes),
        'phys_activity': int(phys), 'stroke': int(stroke),
    }

    result = predict_risk(current_features)
    score = result['risk_score']
    label = result['risk_label']
    color = result['risk_color']
    ci_low = result['ci_low']
    ci_high = result['ci_high']

    # Baseline from original record
    if latest_record:
        orig_features = {
            'age': defaults.get('age', 45), 'bmi': defaults.get('bmi', 27),
            'gen_health': defaults.get('gen_health', 3),
            'highbp': defaults.get('highbp', 0), 'highchol': defaults.get('highchol', 0),
            'smoker': defaults.get('smoker', 0), 'diabetes': defaults.get('diabetes', 0),
            'phys_activity': defaults.get('phys_activity', 0), 'stroke': defaults.get('stroke', 0),
        }
        orig_result = predict_risk(orig_features)
        baseline_score = orig_result['risk_score']
        delta = score - baseline_score
    else:
        baseline_score = score
        delta = 0.0

    with col_right:
        # Big animated gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={
                'reference': baseline_score,
                'valueformat': '.1f',
                'increasing': {'color': '#EF4444'},
                'decreasing': {'color': '#10B981'},
                'suffix': '%',
            },
            title={
                'text': f"<b>{label} Risk</b>",
                'font': {'family': 'Syne', 'color': color, 'size': 20}
            },
            number={
                'suffix': '%',
                'font': {'family': 'Syne', 'size': 48, 'color': color}
            },
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#4A5B7A', 'tickwidth': 1},
                'bar': {'color': color, 'thickness': 0.28},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16,185,129,0.12)'},
                    {'range': [25, 50], 'color': 'rgba(245,158,11,0.10)'},
                    {'range': [50, 75], 'color': 'rgba(239,68,68,0.10)'},
                    {'range': [75, 100], 'color': 'rgba(200,16,46,0.14)'},
                ],
                'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.85, 'value': score}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8A9BBE'),
            height=280, margin=dict(t=20, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        # CI and model info
        st.markdown(f"""
        <div style="display:flex;gap:0.75rem;margin-top:-0.5rem">
            <div class="card-sm" style="flex:1;text-align:center">
                <div class="card-title">95% CI</div>
                <div style="font-weight:700;font-size:1rem;color:{color}">{ci_low}% – {ci_high}%</div>
            </div>
            <div class="card-sm" style="flex:1;text-align:center">
                <div class="card-title">Model</div>
                <div style="font-weight:700;font-size:0.85rem;color:var(--text-secondary)">{result['model_name']}</div>
            </div>
            <div class="card-sm" style="flex:1;text-align:center">
                <div class="card-title">Confidence</div>
                <div style="font-weight:700;font-size:1rem;color:#10B981">{result['model_confidence']:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Delta message
        if abs(delta) > 0.5:
            d_color = '#10B981' if delta < 0 else '#EF4444'
            d_icon = '↓' if delta < 0 else '↑'
            d_word = 'reduced' if delta < 0 else 'increased'
            st.markdown(f"""
            <div class="alert-box {'alert-success' if delta < 0 else 'alert-danger'}" style="margin-top:0.5rem">
                <strong>{d_icon} Risk {d_word} by {abs(delta):.1f}%</strong> compared to your baseline ({baseline_score:.1f}%)
            </div>
            """, unsafe_allow_html=True)

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_shap, col_scenarios = st.columns([1, 1], gap="large")

    with col_shap:
        st.markdown('<div class="card-title">SHAP Feature Impact (Current Settings)</div>', unsafe_allow_html=True)
        shap = result.get('shap_values', {})
        if shap:
            sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            keys = [k for k, _ in sorted_shap]
            vals = [float(v) * 100 for _, v in sorted_shap]
            colors_shap = ['#EF4444' if v > 0 else '#10B981' for v in vals]

            fig_shap = go.Figure(go.Bar(
                x=vals, y=keys, orientation='h',
                marker=dict(color=colors_shap, line=dict(width=0)),
                text=[f'{v:+.1f}%' for v in vals],
                textposition='outside',
                textfont=dict(size=10, color='#8A9BBE'),
                hovertemplate='%{y}: %{x:+.1f}%<extra></extra>'
            ))
            _pl = {k: v for k, v in PLOT_LAYOUT.items() if k != 'yaxis'}
            fig_shap.update_layout(
                **_pl,
                title='',
                height=280,
                xaxis_title='Risk Contribution (%)',
                yaxis=dict(**PLOT_LAYOUT['yaxis'], categoryorder='total ascending'),
            )
            st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

    with col_scenarios:
        st.markdown('<div class="card-title">Pre-built Intervention Scenarios</div>', unsafe_allow_html=True)
        scenarios = get_intervention_scenarios(current_features)

        for sc in scenarios:
            delta_sc = sc['delta']
            d_color = '#10B981' if delta_sc < 0 else '#EF4444'
            d_icon = '↓' if delta_sc < 0 else '↑'
            st.markdown(f"""
            <div class="whatif-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.92rem">{sc['name']}</div>
                        <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.2rem">{sc['description']}</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;margin-left:1rem">
                        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;color:{d_color}">{d_icon} {abs(delta_sc):.1f}%</div>
                        <div style="font-size:0.75rem;color:var(--text-muted)">{sc['new_score']:.1f}% risk</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Factor Comparison Bar ─────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Per-Factor Impact of Your Current Changes vs Baseline</div>',
                unsafe_allow_html=True)

    if latest_record:
        whatif_result = whatif_delta(orig_features, current_features)
        impacts = whatif_result.get('per_factor_impacts', {})
        if impacts:
            fig_impact = go.Figure(go.Bar(
                x=list(impacts.keys()),
                y=list(impacts.values()),
                marker=dict(
                    color=['#10B981' if v < 0 else '#EF4444' for v in impacts.values()],
                    line=dict(width=0)
                ),
                text=[f'{v:+.1f}%' for v in impacts.values()],
                textposition='outside',
                textfont=dict(size=11)
            ))
            fig_impact.update_layout(
                **PLOT_LAYOUT,
                title='',
                height=230,
                yaxis_title='Risk Change (%)',
                xaxis_tickangle=-20,
            )
            fig_impact.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1))
            st.plotly_chart(fig_impact, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown('<div class="alert-box alert-info">Change one or more parameters above to see the per-factor breakdown.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-info">Complete a risk assessment first to enable comparison vs baseline.</div>',
                    unsafe_allow_html=True)
