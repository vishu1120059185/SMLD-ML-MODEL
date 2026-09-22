"""Page 4 — FAILURE FORECAST: multi-horizon probability, RUL, conformal bands."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from stattwin.dashboard.components.charts import timeline_chart, multi_line_chart, gauge_chart
from stattwin.dashboard.components.cards import kpi_card, provenance_badge, section_header

RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"


def _load(name: str, machine: str | None = None):
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
    st.markdown(f"# 🔮 Failure Forecast — {machine}")

    forecast = _load("failure_forecast.json", machine)
    warning = _load("warning_timeline.json", machine)
    conformal = _load("conformal_bands.json", machine)

    if forecast is None:
        st.warning(
            "Forecast data not found. Place **failure_forecast.json** with keys "
            "`failure_prob_30`, `rul`, `rul_ci`, `horizon_probs`, `timestamps` in results/."
        )
        forecast = _demo_forecast()
        st.info("Showing **demo** data for UI validation.")

    # ── KPI row ─────────────────────────────────────────────────────────────
    fp30 = forecast.get("failure_prob_30", 0.0)
    rul = forecast.get("rul", 0.0)
    rul_ci = forecast.get("rul_ci", [None, None])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("P(Failure +30d)", f"{fp30:.1%}", provenance="PREDICTED")
    with c2:
        ci_str = f"[{rul_ci[0]:.0f} – {rul_ci[1]:.0f}]" if rul_ci[0] is not None else ""
        kpi_card("RUL (days)", f"{rul:.0f} {ci_str}", provenance="PREDICTED")
    with c3:
        kpi_card("Model", forecast.get("model", "ensemble"), provenance="PREDICTED")
    with c4:
        st.plotly_chart(
            gauge_chart(fp30, title="Failure Risk", max_val=1.0),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Multi-horizon probability curve ──────────────────────────────────────
    section_header("Multi-Horizon Failure Probability")
    horizon_probs = forecast.get("horizon_probs", {})
    if horizon_probs:
        horizons = sorted(horizon_probs.keys(), key=lambda x: int(x))
        probs = [horizon_probs[h] for h in horizons]
        traces = [{
            "x": [int(h) for h in horizons],
            "y": probs,
            "name": "P(failure)",
            "color": "#D64545",
        }]
        fig = multi_line_chart(traces, title="P(Failure) by Forecast Horizon", y_label="Probability")
        # Conformal band overlay
        if conformal and "upper" in conformal and "lower" in conformal:
            fig.add_trace(go.Scatter(
                x=[int(h) for h in horizons],
                y=conformal.get("upper", []),
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[int(h) for h in horizons],
                y=conformal.get("lower", []),
                fill="tonexty", mode="lines", line=dict(width=0),
                fillcolor="rgba(214,69,69,0.12)", name="Conformal 90%",
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#E0B93B",
                      annotation_text="Warning threshold")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Horizon probabilities not available.")

    # ── RUL prediction point ─────────────────────────────────────────────────
    section_header("RUL Prediction")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        if rul_ci[0] is not None and rul_ci[1] is not None:
            rul_fig = go.Figure()
            rul_fig.add_trace(go.Indicator(
                mode="number+delta",
                value=rul,
                number=dict(suffix=" days", font=dict(size=36)),
                delta=dict(reference=forecast.get("rul_prev", rul + 5),
                           valueformat=".0f", suffix=" d"),
                title=dict(text="Remaining Useful Life", font=dict(size=14)),
            ))
            rul_fig.update_layout(
                paper_bgcolor="#161B22",
                font=dict(color="#C9D1D9"),
                height=200,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(rul_fig, use_container_width=True)
        else:
            kpi_card("RUL", "—")

    with col_right:
        if rul_ci[0] is not None and rul_ci[1] is not None:
            ci_fig = go.Figure()
            ci_fig.add_trace(go.Bar(
                x=[rul_ci[1] - rul_ci[0]],
                y=["RUL"],
                base=[rul_ci[0]],
                orientation="h",
                marker_color="#4C8BF5",
                name="90% CI",
            ))
            ci_fig.add_trace(go.Scatter(
                x=[rul], y=["RUL"],
                mode="markers",
                marker=dict(size=14, color="#D64545", symbol="diamond"),
                name="Point estimate",
            ))
            ci_fig.update_layout(
                paper_bgcolor="#161B22",
                plot_bgcolor="#0E1117",
                font=dict(color="#C9D1D9"),
                height=200,
                margin=dict(l=20, r=20, t=30, b=10),
                xaxis=dict(title="Days", gridcolor="#21262D"),
                yaxis=dict(gridcolor="#21262D"),
                showlegend=True,
            )
            st.plotly_chart(ci_fig, use_container_width=True)

    st.markdown("---")

    # ── Warning timeline ─────────────────────────────────────────────────────
    section_header("Warning Timeline")
    if warning:
        timestamps_w = warning.get("timestamps", [])
        alert_level = warning.get("alert_level", [])
        risk_vals = warning.get("risk", [])
        bands = []
        if "upper" in warning and "lower" in warning:
            bands.append({
                "upper": warning["upper"],
                "lower": warning["lower"],
                "label": "90% CI",
                "fill": "rgba(214,69,69,0.10)",
            })
        fig = timeline_chart(
            timestamps_w, risk_vals,
            title="Risk Trajectory with Alert Levels",
            y_label="Risk",
            bands=bands,
        )
        # Shade alert regions
        alert_colors = {"GREEN": "#2E9E6B", "YELLOW": "#E0B93B", "RED": "#D64545"}
        if alert_level and timestamps_w:
            prev_al = alert_level[0]
            start_i = 0
            for i in range(1, len(alert_level)):
                if alert_level[i] != prev_al:
                    fig.add_vrect(
                        x0=timestamps_w[start_i], x1=timestamps_w[i],
                        fillcolor=alert_colors.get(prev_al, "rgba(0,0,0,0)"),
                        opacity=0.08, layer="below", line_width=0,
                    )
                    prev_al = alert_level[i]
                    start_i = i
            fig.add_vrect(
                x0=timestamps_w[start_i], x1=timestamps_w[-1],
                fillcolor=alert_colors.get(prev_al, "rgba(0,0,0,0)"),
                opacity=0.08, layer="below", line_width=0,
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Warning timeline data not available.")

    # ── Explanation ──────────────────────────────────────────────────────────
    section_header("Forecast Methodology")
    st.markdown(
        f"""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;
                    padding:14px 18px;font-size:0.82rem;color:#8B949E;line-height:1.55;">
            <b style="color:#C9D1D9;">Ensemble approach:</b> Cox PH + RSF + LSTM + Survival SVM
            combined via stacking. Conformal prediction provides distribution-free coverage
            guarantees. RUL estimated as conditional expected lifetime given current state.
            {provenance_badge('PREDICTED')}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _demo_forecast() -> dict:
    rng = np.random.default_rng(42)
    horizons = list(range(1, 91))
    probs = [min(1.0, 0.02 + 0.008 * h + rng.normal(0, 0.005)) for h in horizons]
    return {
        "failure_prob_30": probs[29] if len(probs) > 29 else 0.5,
        "rul": 42,
        "rul_ci": [28, 58],
        "model": "ensemble (demo)",
        "horizon_probs": {str(h): p for h, p in zip(horizons, probs)},
        "timestamps": list(range(len(horizons))),
    }
