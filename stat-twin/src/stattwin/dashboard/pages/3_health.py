"""Page 3 — STATISTICAL HEALTH: SHI over time, Z-heatmap, variance, distributions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from stattwin.dashboard.components.charts import timeline_chart, heatmap_chart, bar_chart
from stattwin.dashboard.components.cards import kpi_card, state_badge, section_header

RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"

STATE_COLORS = {
    "HEALTHY": "#2E9E6B",
    "WATCH": "#E0B93B",
    "DEGRADING": "#E8862F",
    "CRITICAL": "#D64545",
    "FAILURE-LIKELY": "#8E1B3A",
}


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
    st.markdown(f"# 🏥 Statistical Health — {machine}")

    health = _load("health_summary.json", machine)
    shi_timeline = _load("shi_timeline.json", machine)
    z_heatmap = _load("z_heatmap.json", machine)
    variance_data = _load("variance_change.json", machine)
    dist_shift = _load("distribution_shift.json", machine)
    corr_matrix = _load("correlation_matrix.json", machine)

    # ── SHI + state bands ───────────────────────────────────────────────────
    section_header("SHI Over Time", "with state-transition bands")
    if shi_timeline:
        timestamps = shi_timeline.get("timestamps", [])
        shi_vals = shi_timeline.get("shi", [])
        states = shi_timeline.get("states", [])

        state_changes = []
        if states and timestamps:
            prev_state = states[0]
            start_idx = 0
            for i in range(1, len(states)):
                if states[i] != prev_state:
                    state_changes.append({
                        "start": timestamps[start_idx],
                        "end": timestamps[i - 1],
                        "state": prev_state,
                    })
                    prev_state = states[i]
                    start_idx = i
            state_changes.append({
                "start": timestamps[start_idx],
                "end": timestamps[-1],
                "state": prev_state,
            })

        fig = timeline_chart(
            timestamps, shi_vals,
            title="Statistical Health Index",
            y_label="SHI",
            state_changes=state_changes if state_changes else None,
        )

        # Add horizontal threshold lines
        fig.add_hline(y=0.25, line_dash="dot", line_color="#E0B93B", line_width=1,
                      annotation_text="WATCH", annotation_position="right")
        fig.add_hline(y=0.50, line_dash="dot", line_color="#E8862F", line_width=1,
                      annotation_text="DEGRADING", annotation_position="right")
        fig.add_hline(y=0.75, line_dash="dot", line_color="#D64545", line_width=1,
                      annotation_text="CRITICAL", annotation_position="right")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHI timeline not available. Run the pipeline first.")

    # ── Current state ────────────────────────────────────────────────────────
    if health:
        state = health.get("state", "HEALTHY")
        shi = health.get("shi", 0.0)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div style='background:#161B22;border:1px solid #21262D;border-radius:8px;"
                f"padding:14px;text-align:center;'>"
                f"<div style='font-size:0.72rem;color:#8B949E;text-transform:uppercase;"
                f"letter-spacing:0.8px;margin-bottom:6px;'>Current State</div>"
                f"{state_badge(state)}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            kpi_card("SHI", f"{shi:.4f}", provenance="OBSERVED")
        with c3:
            kpi_card("State", state, provenance="OBSERVED")

    # ── Z-Score Heatmap ──────────────────────────────────────────────────────
    section_header("Z-Score Heatmap", "sensor × time")
    if z_heatmap:
        z = np.array(z_heatmap.get("z", [[]]))
        x_labels = z_heatmap.get("time_labels", [f"t{i}" for i in range(z.shape[1])])
        y_labels = z_heatmap.get("sensor_labels", [f"s{i}" for i in range(z.shape[0])])
        fig = heatmap_chart(z, x_labels, y_labels, title="Rolling Z-Score Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Z-heatmap data not available.")

    # ── Variance-change chart ────────────────────────────────────────────────
    section_header("Variance-Change Detection")
    if variance_data:
        sensors_v = variance_data.get("sensors", {})
        if sensors_v:
            tab_names = list(sensors_v.keys())[:6]
            tabs = st.tabs(tab_names)
            for idx, sname in enumerate(tab_names):
                with tabs[idx]:
                    svar = sensors_v[sname]
                    timestamps_v = svar.get("timestamps", [])
                    var_vals = svar.get("variance", [])
                    changepoints = svar.get("changepoints", [])
                    fig = timeline_chart(
                        timestamps_v, var_vals,
                        title=f"{sname} — Rolling Variance",
                        y_label="Variance",
                    )
                    for cp in changepoints:
                        cp_idx = cp if isinstance(cp, (int, float)) else cp.get("index", 0)
                        if cp_idx < len(timestamps_v):
                            fig.add_vline(
                                x=timestamps_v[cp_idx],
                                line_dash="dash", line_color="#D64545", line_width=1.5,
                                annotation_text="Change",
                            )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Variance data empty.")
    else:
        st.info("Variance-change data not available.")

    # ── Distribution-shift table ─────────────────────────────────────────────
    section_header("Distribution-Shift Table")
    if dist_shift:
        rows = dist_shift if isinstance(dist_shift, list) else dist_shift.get("rows", [])
        if rows:
            st.dataframe(
                rows,
                use_container_width=True,
                column_config={
                    "sensor": st.column_config.TextColumn("Sensor"),
                    "ks_stat": st.column_config.NumberColumn("KS Statistic", format="%.4f"),
                    "ks_p": st.column_config.NumberColumn("KS p-value", format="%.4f"),
                    "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                    "shifted": st.column_config.CheckboxColumn("Shifted"),
                },
            )
        else:
            st.info("No distribution-shift rows found.")
    else:
        st.info("Distribution-shift table not available.")

    # ── Correlation matrix ───────────────────────────────────────────────────
    section_header("Correlation Matrix")
    if corr_matrix:
        c = np.array(corr_matrix.get("matrix", [[]]))
        labels = corr_matrix.get("labels", [f"s{i}" for i in range(c.shape[0])])
        fig = heatmap_chart(
            c, labels, labels,
            title="Sensor Correlation Matrix",
            colorscale="RdBu_r",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Correlation matrix not available.")
