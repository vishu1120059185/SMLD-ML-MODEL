"""Page 1 — OVERVIEW: machine health at a glance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from ..components.charts import gauge_chart, timeline_chart, bar_chart
from ..components.cards import (
    kpi_card,
    state_badge,
    evidence_card,
    recommendation_card,
    section_header,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def _load(name: str, machine: str | None = None):
    """Load JSON artifact; try machine dir first, then global, then root results."""
    candidates = []
    if machine:
        candidates.append(RESULTS_DIR / machine / name)
    candidates.append(RESULTS_DIR / "global" / name)
    candidates.append(RESULTS_DIR / name)
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def render():
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    st.markdown(f"# ⚙ Overview — {machine}")

    # ── Load artifacts ──────────────────────────────────────────────────────
    health = _load("health_summary.json", machine)
    forecast = _load("failure_forecast.json", machine)
    recommendations = _load("recommendations.json", machine)
    dq = _load("data_quality.json", machine)
    timeline_data = _load("shi_timeline.json", machine)
    risk_data = _load("risk_timeline.json", machine)

    # ── KPI row ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    state = health.get("state", "HEALTHY") if health else "HEALTHY"
    shi = health.get("shi", 0.0) if health else 0.0
    failure_prob = forecast.get("failure_prob_30", 0.0) if forecast else 0.0
    rul = forecast.get("rul", None) if forecast else None
    rul_ci = forecast.get("rul_ci", [None, None]) if forecast else [None, None]

    with col1:
        st.markdown(
            f"<div style='background:#161B22;border:1px solid #21262D;border-radius:8px;"
            f"padding:14px 16px;'>"
            f"<div style='font-size:0.72rem;color:#8B949E;text-transform:uppercase;"
            f"letter-spacing:0.8px;margin-bottom:6px;'>State</div>"
            f"<div style='margin-bottom:4px;'>{state_badge(state)}</div></div>",
            unsafe_allow_html=True,
        )

    with col2:
        kpi_card("SHI", f"{shi:.3f}", provenance="OBSERVED")

    with col3:
        kpi_card("P(Failure +30d)", f"{failure_prob:.1%}", provenance="PREDICTED")

    with col4:
        if rul is not None:
            ci_str = f"[{rul_ci[0]:.0f} – {rul_ci[1]:.0f}]" if rul_ci[0] is not None else ""
            kpi_card("RUL (days)", f"{rul:.0f} {ci_str}", provenance="PREDICTED")
        else:
            kpi_card("RUL (days)", "—")

    # ── SHI Gauge + Risk Timeline ──────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        section_header("SHI Gauge")
        fig = gauge_chart(shi, title="Statistical Health Index")
        st.plotly_chart(fig, use_container_width=True, key="shi_gauge")

    with right:
        section_header("Risk Timeline")
        if risk_data:
            timestamps = risk_data.get("timestamps", [])
            risk_vals = risk_data.get("risk", [])
            bands_data = []
            if "upper" in risk_data and "lower" in risk_data:
                bands_data.append({
                    "upper": risk_data["upper"],
                    "lower": risk_data["lower"],
                    "label": "95 % CI",
                    "fill": "rgba(76,139,245,0.10)",
                })
            fig = timeline_chart(
                timestamps, risk_vals,
                title="Failure Risk Over Time",
                y_label="Risk",
                bands=bands_data,
            )
            st.plotly_chart(fig, use_container_width=True, key="risk_timeline")
        else:
            st.info("Risk timeline data not available in results/.")

    # ── DQ Status ───────────────────────────────────────────────────────────
    section_header("Data Quality")
    if dq:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            kpi_card("Completeness", f"{dq.get('completeness', 0):.1%}")
        with col_b:
            kpi_card("Timeliness", f"{dq.get('timeliness', 0):.1%}")
        with col_c:
            kpi_card("Plausibility", f"{dq.get('plausibility', 0):.1%}")
        with col_d:
            kpi_card("Overall DQ", f"{dq.get('overall', 0):.1%}")
    else:
        st.info("Data-quality metrics not available in results/.")

    # ── Recommendation ──────────────────────────────────────────────────────
    section_header("Recommendations")
    if recommendations:
        recs = recommendations if isinstance(recommendations, list) else [recommendations]
        for rec in recs[:3]:
            recommendation_card(
                rec.get("text", str(rec)),
                priority=rec.get("priority", "medium"),
                provenance=rec.get("provenance", "PREDICTED"),
            )
    else:
        recommendation_card(
            "All parameters nominal. Continue routine monitoring.",
            priority="low",
            provenance="OBSERVED",
        )

    # ── SHI Timeline ────────────────────────────────────────────────────────
    section_header("SHI Over Time")
    if timeline_data:
        fig = timeline_chart(
            timeline_data.get("timestamps", []),
            timeline_data.get("shi", []),
            title="SHI Trajectory",
            y_label="SHI",
        )
        st.plotly_chart(fig, use_container_width=True, key="shi_timeline")
    else:
        st.info("SHI timeline data not available in results/. Run the pipeline first.")
