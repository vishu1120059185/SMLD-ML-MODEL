"""KPI cards, evidence cards, recommendation cards with provenance badges."""
from __future__ import annotations

import streamlit as st

STATE_COLORS = {
    "HEALTHY": "#2E9E6B",
    "WATCH": "#E0B93B",
    "DEGRADING": "#E8862F",
    "CRITICAL": "#D64545",
    "FAILURE-LIKELY": "#8E1B3A",
}

PROVENANCE_COLORS = {
    "OBSERVED": "#4C8BF5",
    "PREDICTED": "#F5A623",
    "SIMULATED": "#9B6BFF",
}


def _badge(label: str, bg: str, fg: str = "#FFFFFF") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:0.72rem;font-weight:600;'
        f'letter-spacing:0.4px;">{label}</span>'
    )


def state_badge(state: str) -> str:
    colour = STATE_COLORS.get(state.upper(), "#555555")
    return _badge(state.upper(), colour)


def provenance_badge(kind: str) -> str:
    colour = PROVENANCE_COLORS.get(kind.upper(), "#555555")
    return _badge(kind.upper(), colour)


def kpi_card(
    label: str,
    value: str | float | int,
    *,
    delta: str | None = None,
    delta_color: str = "normal",
    provenance: str | None = None,
):
    """Render a KPI metric card."""
    prov_html = f" {provenance_badge(provenance)}" if provenance else ""
    st.markdown(
        f"""
        <div style="
            background:#161B22; border:1px solid #21262D; border-radius:8px;
            padding:16px 18px; margin-bottom:12px;
        ">
            <div style="font-size:0.75rem; color:#8B949E; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:6px;">
                {label}{prov_html}
            </div>
            <div style="font-size:1.6rem; font-weight:700; color:#C9D1D9;
                        font-family:'JetBrains Mono','Fira Code',monospace;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if delta:
        st.caption(delta)


def evidence_card(
    title: str,
    body: str,
    *,
    sensor: str | None = None,
    provenance: str = "OBSERVED",
    severity: str = "info",
):
    """Evidence card with provenance badge."""
    sev_colours = {
        "info": "#4C8BF5",
        "warning": "#E0B93B",
        "danger": "#D64545",
        "success": "#2E9E6B",
    }
    border = sev_colours.get(severity, "#21262D")
    sensor_html = (
        f'<span style="color:#8B949E;font-size:0.72rem;"> | {sensor}</span>'
        if sensor else ""
    )
    st.markdown(
        f"""
        <div style="
            background:#161B22; border-left:3px solid {border};
            border-radius:0 6px 6px 0; padding:12px 16px; margin-bottom:10px;
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span style="font-weight:600; color:#C9D1D9; font-size:0.88rem;">
                    {title}
                </span>
                {provenance_badge(provenance)}
                {sensor_html}
            </div>
            <div style="color:#8B949E; font-size:0.82rem; line-height:1.45;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_card(
    text: str,
    *,
    priority: str = "medium",
    provenance: str = "PREDICTED",
):
    """Actionable recommendation card."""
    prio_colors = {
        "high": "#D64545",
        "medium": "#E0B93B",
        "low": "#2E9E6B",
    }
    border = prio_colors.get(priority, "#4C8BF5")
    st.markdown(
        f"""
        <div style="
            background:#161B22; border-left:3px solid {border};
            border-radius:0 6px 6px 0; padding:12px 16px; margin-bottom:10px;
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span style="font-weight:600; color:#C9D1D9; font-size:0.88rem;">
                    Recommendation
                </span>
                {_badge(priority.upper(), border)}
                {provenance_badge(provenance)}
            </div>
            <div style="color:#8B949E; font-size:0.82rem; line-height:1.45;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def delta_card(label: str, original: float, simulated: float, *, fmt: str = ".3f"):
    """Show original → simulated with delta."""
    delta = simulated - original
    sign = "+" if delta >= 0 else ""
    colour = "#D64545" if delta < 0 else "#2E9E6B"
    st.markdown(
        f"""
        <div style="
            background:#161B22; border:1px solid #21262D; border-radius:8px;
            padding:14px 16px; margin-bottom:10px;
        ">
            <div style="font-size:0.72rem; color:#8B949E; text-transform:uppercase;
                        letter-spacing:0.6px; margin-bottom:6px;">{label}</div>
            <div style="display:flex; align-items:baseline; gap:10px;">
                <span style="font-size:1.1rem; color:#C9D1D9;
                             font-family:'JetBrains Mono',monospace;">
                    {original:{fmt}}
                </span>
                <span style="color:#8B949E;">→</span>
                <span style="font-size:1.1rem; color:#C9D1D9;
                             font-family:'JetBrains Mono',monospace;">
                    {simulated:{fmt}}
                </span>
                <span style="font-size:0.85rem; font-weight:600; color:{colour};">
                    ({sign}{delta:{fmt}})
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer_banner(text: str = "SIMULATION — not real operational data"):
    st.markdown(
        f"""
        <div style="
            background:rgba(155,107,255,0.15); border:1px solid #9B6BFF;
            border-radius:6px; padding:10px 16px; margin-bottom:14px;
            color:#C9D1D9; font-size:0.82rem; text-align:center;
            font-weight:600; letter-spacing:0.4px;">
            ⚠ {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    sub_html = (
        f'<span style="color:#8B949E;font-size:0.82rem;margin-left:8px;">{subtitle}</span>'
        if subtitle else ""
    )
    st.markdown(
        f'<h3 style="color:#C9D1D9;margin-bottom:4px;">{title}{sub_html}</h3>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
