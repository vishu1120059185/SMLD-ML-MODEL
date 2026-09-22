"""Page 2 — SENSOR MONITORING: raw signals, rolling stats, EWMA, Z-scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from stattwin.dashboard.components.cards import kpi_card, provenance_badge, section_header
from stattwin.dashboard.components.charts import sparkline, timeline_chart

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
    st.markdown(f"# 📡 Sensor Monitoring — {machine}")

    sensor_data = _load("sensor_data.json", machine)
    dq_flags = _load("dq_flags.json", machine)

    if sensor_data is None:
        st.warning(
            "Sensor data not found in `results/`. "
            "Place **sensor_data.json** with keys `sensors` (dict of name→{timestamps, values}) "
            "in the machine's results directory."
        )
        # Generate demo data so the UI is functional
        sensor_data = _generate_demo()
        st.info("Showing **demo** data for UI validation.")

    sensors: dict = sensor_data.get("sensors", {})
    if not sensors:
        st.error("No sensor series found in the data.")
        return

    sensor_names = sorted(sensors.keys())

    # ── Sidebar filters ─────────────────────────────────────────────────────
    selected = st.multiselect(
        "Select sensors",
        sensor_names,
        default=sensor_names[: min(3, len(sensor_names))],
        key="sensor_sel",
    )
    window = st.slider("Rolling window", 5, 200, 30, key="roll_win")
    ewma_alpha = st.slider("EWMA α", 0.01, 0.5, 0.1, 0.01, key="ewma_a")

    if not selected:
        st.info("Select at least one sensor above.")
        return

    # ── KPI summary row ─────────────────────────────────────────────────────
    cols = st.columns(min(len(selected), 4))
    for i, sname in enumerate(selected[:4]):
        vals = sensors[sname].get("values", [])
        if vals:
            with cols[i]:
                kpi_card(
                    sname,
                    f"{vals[-1]:.3f}",
                    delta=f"mean={np.mean(vals):.3f}  σ={np.std(vals):.3f}",
                    provenance="OBSERVED",
                )

    st.markdown("---")

    # ── Per-sensor charts ───────────────────────────────────────────────────
    for sname in selected:
        sdata = sensors[sname]
        timestamps = sdata.get("timestamps", list(range(len(sdata.get("values", [])))))
        values = np.array(sdata.get("values", []), dtype=float)

        if len(values) == 0:
            continue

        # Rolling mean / std band
        rmean = np.convolve(values, np.ones(window) / window, mode="same")
        rstd = np.array([
            np.std(values[max(0, j - window): j + 1]) for j in range(len(values))
        ])
        upper = rmean + 2 * rstd
        lower = rmean - 2 * rstd

        # EWMA
        ewma = np.zeros_like(values)
        ewma[0] = values[0]
        for j in range(1, len(values)):
            ewma[j] = ewma_alpha * values[j] + (1 - ewma_alpha) * ewma[j - 1]

        # Z-score (rolling)
        overall_mean = np.mean(values)
        overall_std = np.std(values) or 1.0
        zscore = (values - overall_mean) / overall_std

        col_chart, col_z = st.columns([3, 1])

        with col_chart:
            bands = [{
                "upper": upper, "lower": lower,
                "label": f"±2σ ({window})",
                "fill": "rgba(59,130,246,0.10)",
            }]
            fig = timeline_chart(
                timestamps, values.tolist(),
                title=f"{sname} — Raw + Rolling + EWMA",
                y_label="Value",
                bands=bands,
                extra_traces=[
                    go.Scatter(
                        x=list(timestamps), y=ewma.tolist(),
                        mode="lines", name="EWMA",
                        line=dict(color="#F5A623", width=1.5, dash="dot"),
                    )
                ],
            )
            st.plotly_chart(fig, use_container_width=True, key=f"sensor_{sname}")

        with col_z:
            fig_z = timeline_chart(
                timestamps, zscore.tolist(),
                title=f"{sname} — Z-Score",
                y_label="Z",
            )
            fig_z.add_hline(y=2, line_dash="dash", line_color="#D64545", line_width=1)
            fig_z.add_hline(y=-2, line_dash="dash", line_color="#D64545", line_width=1)
            st.plotly_chart(fig_z, use_container_width=True, key=f"zscore_{sname}")

        # ── DQ-flag markers ─────────────────────────────────────────────────
        if dq_flags and sname in dq_flags:
            flags = dq_flags[sname]
            flagged_idx = flags.get("flagged_indices", [])
            if flagged_idx:
                st.markdown(
                    f"⚠ {sname}: **{len(flagged_idx)}** DQ-flagged points "
                    f"({provenance_badge('OBSERVED')})",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

    # ── Sparkline strip ─────────────────────────────────────────────────────
    section_header("Sensor Sparklines")
    spark_cols = st.columns(min(len(selected), 6))
    for i, sname in enumerate(selected[:6]):
        vals = sensors[sname].get("values", [])
        if vals:
            with spark_cols[i]:
                st.markdown(f"<div style='text-align:center;font-size:0.72rem;"
                            f"color:#8B949E;'>{sname}</div>", unsafe_allow_html=True)
                fig = sparkline(vals)
                st.plotly_chart(fig, use_container_width=True, key=f"spark_{sname}")


def _generate_demo() -> dict:
    """Generate synthetic sensor data for UI validation."""
    rng = np.random.default_rng(42)
    n = 500
    t = list(range(n))
    sensors = {}
    names = ["VIBRATION_X", "VIBRATION_Y", "TEMPERATURE", "PRESSURE", "CURRENT"]
    for name in names:
        base = rng.uniform(0.5, 5.0)
        vals = (base + rng.normal(0, base * 0.05, n)).tolist()
        sensors[name] = {"timestamps": t, "values": vals}
    return {"sensors": sensors}
