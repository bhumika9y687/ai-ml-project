"""Model explanation visualization used by the patient risk assessment."""

import plotly.graph_objects as go
import streamlit as st

from utils.theme import PLOT_LAYOUT


def render_shap_waterfall(shap_values: dict):
    """Show the strongest model feature contributions for one prediction."""
    if not shap_values:
        return

    items = sorted(shap_values.items(), key=lambda item: abs(float(item[1])), reverse=True)[:8]
    names = [name for name, _ in items]
    values = [float(value) * 100 for _, value in items]
    colors = ["#ef4444" if value > 0 else "#10b981" for value in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.1f}%" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(**PLOT_LAYOUT, title="Risk Contribution Factors", height=280, xaxis_title="Contribution")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
