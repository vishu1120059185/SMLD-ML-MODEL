"""KPI cards, evidence cards, recommendation cards with provenance badges."""
from __future__ import annotations

import streamlit as st

# ── Design tokens (matches app.py industrial dark theme) ────────────────────
CARD_BG = "#111827"
CARD_BORDER = "#1F2937"
TEXT_PRIMARY = "#F9FAFB"
TEXT_SECONDARY = "#9CA3AF"
ACCENT = "#3B82F6"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
SHADOW = "0 4px 6px -1px rgba(0,0,0,0.45), 0 2px 4px -2px rgba(0,0,0,0.35)"

STATE_COLORS = {
    "HEALTHY": SUCCESS,
    "WATCH": WARNING,
    "DEGRADING": "#F97316",
    "CRITICAL": DANGER,
    "FAILURE-LIKELY": "#991B1B",
}

PROVENANCE_COLORS = {
    "OBSERVED": ACCENT,
    "PREDICTED": WARNING,
    "SIMULATED": "#8B5CF6",
}


def _badge(label: str, bg: str, fg: str = "#FFFFFF") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:0.72rem;font-weight:600;'
        f'letter-spacing:0.4px;">{label}</span>'
    )


def state_badge(state: str) -> str:
    colour = STATE_COLORS.get(state.upper(), "#4B5563")
    return _badge(state.upper(), colour)


def provenance_badge(kind: str) -> str:
    colour = PROVENANCE_COLORS.get(kind.upper(), "#4B5563")
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
            background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px;
            padding:16px 18px; margin-bottom:12px; box-shadow:{SHADOW};
        ">
            <div style="font-size:0.72rem; color:{TEXT_SECONDARY}; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:6px; font-weight:600;">
                {label}{prov_html}
            </div>
            <div style="font-size:1.6rem; font-weight:700; color:{TEXT_PRIMARY};
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
        "info": ACCENT,
        "warning": WARNING,
        "danger": DANGER,
        "success": SUCCESS,
    }
    border = sev_colours.get(severity, CARD_BORDER)
    sensor_html = (
        f'<span style="color:{TEXT_SECONDARY};font-size:0.72rem;"> | {sensor}</span>'
        if sensor else ""
    )
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG}; border-left:3px solid {border};
            border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;
            box-shadow:{SHADOW};
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span style="font-weight:600; color:{TEXT_PRIMARY}; font-size:0.88rem;">
                    {title}
                </span>
                {provenance_badge(provenance)}
                {sensor_html}
            </div>
            <div style="color:{TEXT_SECONDARY}; font-size:0.82rem; line-height:1.45;">
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
        "high": DANGER,
        "medium": WARNING,
        "low": SUCCESS,
    }
    border = prio_colors.get(priority, ACCENT)
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG}; border-left:3px solid {border};
            border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;
            box-shadow:{SHADOW};
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span style="font-weight:600; color:{TEXT_PRIMARY}; font-size:0.88rem;">
                    Recommendation
                </span>
                {_badge(priority.upper(), border)}
                {provenance_badge(provenance)}
            </div>
            <div style="color:{TEXT_SECONDARY}; font-size:0.82rem; line-height:1.45;">
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
    colour = DANGER if delta < 0 else SUCCESS
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px;
            padding:14px 16px; margin-bottom:10px; box-shadow:{SHADOW};
        ">
            <div style="font-size:0.72rem; color:{TEXT_SECONDARY}; text-transform:uppercase;
                        letter-spacing:0.8px; margin-bottom:6px; font-weight:600;">{label}</div>
            <div style="display:flex; align-items:baseline; gap:10px;">
                <span style="font-size:1.1rem; color:{TEXT_PRIMARY};
                             font-family:'JetBrains Mono',monospace;">
                    {original:{fmt}}
                </span>
                <span style="color:{TEXT_SECONDARY};">→</span>
                <span style="font-size:1.1rem; color:{TEXT_PRIMARY};
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
            background:rgba(139,92,246,0.12); border:1px solid #8B5CF6;
            border-radius:8px; padding:10px 16px; margin-bottom:14px;
            color:{TEXT_PRIMARY}; font-size:0.82rem; text-align:center;
            font-weight:600; letter-spacing:0.4px;">
            ⚠ {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    sub_html = (
        f'<span style="color:{TEXT_SECONDARY};font-size:0.82rem;margin-left:8px;">{subtitle}</span>'
        if subtitle else ""
    )
    st.markdown(
        f'<h3 style="color:{TEXT_PRIMARY};margin-bottom:4px;">{title}{sub_html}</h3>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
