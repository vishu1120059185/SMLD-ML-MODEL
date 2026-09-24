"""Page 2 — SENSOR MONITORING: live signals, rolling stats, EWMA, Z-scores."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from stattwin.dashboard.components.artifacts import load_artifact as _load
from stattwin.dashboard.components.cards import (
    kpi_card,
    page_header,
    provenance_badge,
    section_header,
)
from stattwin.dashboard.components.charts import sparkline, timeline_chart
from stattwin.dashboard.components.live import (
    LIVE_INTERVAL,
    current_tick,
    live_series,
    live_status,
    live_window,
)


def _rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    w = max(1, min(int(window), len(values)))
    kernel = np.ones(w) / w
    rmean = np.convolve(values, kernel, mode="same")
    rstd = np.array(
        [np.std(values[max(0, j - w) : j + 1]) for j in range(len(values))],
        dtype=float,
    )
    return rmean, rstd


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    ewma = np.zeros_like(values)
    if len(values) == 0:
        return ewma
    ewma[0] = values[0]
    for j in range(1, len(values)):
        ewma[j] = alpha * values[j] + (1.0 - alpha) * ewma[j - 1]
    return ewma


def render() -> None:
    """Render the live Sensor Monitoring page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    page_header("Sensor Monitoring", "raw · rolling · EWMA · z-score · DQ flags", machine)

    sensor_data = _load("sensor_data.json", machine)
    demo_mode = sensor_data is None
    if demo_mode:
        sensor_data = _generate_demo()
        st.info("Sensor data not found in `results/` — showing **demo** data.")

    sensors: dict = sensor_data.get("sensors", {})
    if not sensors:
        st.error("No sensor series found in the data.")
        return

    sensor_names = sorted(sensors.keys())

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

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_sensors() -> None:
        live_status("sensors", feed="sliding sensor window · rolling · EWMA @ 2s")
        tick = current_tick()

        raw = _load("sensor_data.json", machine)
        is_demo = raw is None
        dq_flags = _load("dq_flags.json", machine)
        data = raw if raw is not None else _generate_demo()
        live_map: dict = data.get("sensors", {})
        prov = "SIMULATED" if is_demo else "OBSERVED"

        cols = st.columns(min(len(selected), 4))
        for i, sname in enumerate(selected[:4]):
            sdata = live_map.get(sname)
            if not sdata:
                continue
            vals = live_series(
                f"{machine}:{sname}",
                tick,
                n=150,
                base_values=sdata.get("values", []),
            )
            with cols[i]:
                kpi_card(
                    sname,
                    f"{vals[-1]:.3f}",
                    delta=f"mean={np.mean(vals):.3f}  σ={np.std(vals):.3f}  tick #{tick}",
                    provenance=prov,
                )

        st.markdown("---")

        for sname in selected:
            sdata = live_map.get(sname)
            if not sdata:
                continue
            timestamps, values = live_window(
                f"{machine}:{sname}",
                tick,
                n=150,
                base_values=sdata.get("values", []),
                timestamps=sdata.get("timestamps"),
            )
            if len(values) == 0:
                continue

            rmean, rstd = _rolling(values, window)
            upper = rmean + 2 * rstd
            lower = rmean - 2 * rstd
            ewma = _ewma(values, ewma_alpha)
            overall_mean = float(np.mean(values))
            overall_std = float(np.std(values)) or 1.0
            zscore = (values - overall_mean) / overall_std

            col_chart, col_z = st.columns([3, 1])
            with col_chart:
                bands = [
                    {
                        "upper": upper,
                        "lower": lower,
                        "label": f"±2σ ({min(window, len(values))})",
                        "fill": "rgba(59,130,246,0.10)",
                    }
                ]
                fig = timeline_chart(
                    timestamps,
                    values.tolist(),
                    title=f"{sname} — Raw + Rolling + EWMA",
                    y_label="Value",
                    bands=bands,
                    extra_traces=[
                        go.Scatter(
                            x=list(timestamps),
                            y=ewma.tolist(),
                            mode="lines",
                            name="EWMA",
                            line=dict(color="#F5A623", width=1.5, dash="dot"),
                        ),
                        go.Scatter(
                            x=list(timestamps),
                            y=rmean.tolist(),
                            mode="lines",
                            name="Rolling mean",
                            line=dict(color="#10B981", width=1.2),
                        ),
                    ],
                )
                st.plotly_chart(fig, width="stretch", key=f"sensor_{sname}_{tick}")

            with col_z:
                fig_z = timeline_chart(
                    timestamps,
                    zscore.tolist(),
                    title=f"{sname} — Z-Score",
                    y_label="Z",
                )
                fig_z.add_hline(
                    y=2, line_dash="dash", line_color="#EF4444", line_width=1
                )
                fig_z.add_hline(
                    y=-2, line_dash="dash", line_color="#EF4444", line_width=1
                )
                st.plotly_chart(fig_z, width="stretch", key=f"zscore_{sname}_{tick}")

            if dq_flags and sname in dq_flags:
                flagged_idx = dq_flags[sname].get("flagged_indices", [])
                if flagged_idx:
                    st.markdown(
                        f"⚠ {sname}: **{len(flagged_idx)}** DQ-flagged points "
                        f"({provenance_badge('OBSERVED')})",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

        section_header("Sensor Sparklines")
        spark_cols = st.columns(min(len(selected), 6))
        for i, sname in enumerate(selected[:6]):
            sdata = live_map.get(sname)
            if not sdata:
                continue
            vals = live_series(
                f"{machine}:{sname}",
                tick,
                n=150,
                base_values=sdata.get("values", []),
            )
            with spark_cols[i]:
                st.markdown(
                    f"<div style='text-align:center;font-size:0.72rem;"
                    f"color:#9CA3AF;font-family:JetBrains Mono,monospace;"
                    f"letter-spacing:0.5px;'>{sname}</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    sparkline(vals),
                    width="stretch",
                    key=f"spark_{sname}_{tick}",
                )

    live_sensors()


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
