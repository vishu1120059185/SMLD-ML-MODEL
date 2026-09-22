"""Page 6 — WHAT-IF SIMULATOR: sensor sliders, original vs simulated, deltas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ..components.charts import timeline_chart, gauge_chart, multi_line_chart
from ..components.cards import (
    delta_card,
    disclaimer_banner,
    kpi_card,
    section_header,
    provenance_badge,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


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


def _simulate(sensors: dict, adjustments: dict) -> dict:
    """
    Simple what-if engine: adjusts sensor values and re-estimates SHI & RUL.
    In production this would call the actual pipeline; here we do a heuristic proxy.
    """
    modified = {}
    for name, data in sensors.items():
        vals = list(data.get("values", []))
        if name in adjustments:
            delta = adjustments[name]
            vals = [v + delta for v in vals]
        modified[name] = vals

    # Heuristic SHI estimation: weighted std-dev degradation
    all_stds = []
    weights = {"VIBRATION_X": 1.5, "VIBRATION_Y": 1.5, "TEMPERATURE": 1.0,
               "PRESSURE": 0.8, "CURRENT": 1.2}
    for name, vals in modified.items():
        w = weights.get(name, 1.0)
        all_stds.append(np.std(vals) * w)
    shi_sim = min(1.0, np.mean(all_stds) * 0.3) if all_stds else 0.0

    # State from SHI
    if shi_sim < 0.25:
        state = "HEALTHY"
    elif shi_sim < 0.50:
        state = "WATCH"
    elif shi_sim < 0.75:
        state = "DEGRADING"
    else:
        state = "CRITICAL"

    # RUL inverse to SHI
    rul_sim = max(1, int(90 * (1 - shi_sim)))

    return {
        "modified_sensors": modified,
        "shi": shi_sim,
        "state": state,
        "rul": rul_sim,
    }


def render():
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    st.markdown(f"# 🔧 What-If Simulator — {machine}")

    disclaimer_banner("SIMULATION — not real operational data. Results are approximate proxies.")

    sensor_data = _load("sensor_data.json", machine)
    health = _load("health_summary.json", machine)
    forecast = _load("failure_forecast.json", machine)

    if sensor_data is None:
        st.warning("Sensor data not found. Using demo data.")
        sensor_data = _demo_sensors()

    sensors_dict = sensor_data.get("sensors", {})
    if not sensors_dict:
        st.error("No sensors available.")
        return

    # ── Baseline values ─────────────────────────────────────────────────────
    baseline_shi = health.get("shi", 0.0) if health else 0.0
    baseline_state = health.get("state", "HEALTHY") if health else "HEALTHY"
    baseline_rul = forecast.get("rul", 45) if forecast else 45

    section_header("Original Baseline")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("SHI", f"{baseline_shi:.3f}", provenance="OBSERVED")
    with c2:
        kpi_card("State", baseline_state, provenance="OBSERVED")
    with c3:
        kpi_card("RUL", f"{baseline_rul} days", provenance="OBSERVED")

    st.markdown("---")

    # ── Sensor adjustment sliders ───────────────────────────────────────────
    section_header("Sensor Adjustments")
    st.markdown(
        f"<div style='color:#8B949E;font-size:0.82rem;margin-bottom:12px;'>"
        f"Adjust sensor offsets. Positive = increase, negative = decrease. "
        f"{provenance_badge('SIMULATED')}</div>",
        unsafe_allow_html=True,
    )

    adjustments = {}
    slider_cols = st.columns(min(len(sensors_dict), 3))
    for i, (sname, sdata) in enumerate(sorted(sensors_dict.items())):
        vals = sdata.get("values", [])
        if not vals:
            continue
        current_mean = np.mean(vals)
        std = np.std(vals) or abs(current_mean) * 0.1
        col_idx = i % len(slider_cols)
        with slider_cols[col_idx]:
            adj = st.slider(
                sname,
                min_value=-3.0 * std,
                max_value=3.0 * std,
                value=0.0,
                step=std * 0.05,
                key=f"adj_{sname}",
                help=f"Current mean: {current_mean:.3f} ± {std:.3f}",
            )
            if abs(adj) > 1e-9:
                adjustments[sname] = adj

    st.markdown("---")

    # ── Simulate ─────────────────────────────────────────────────────────────
    if adjustments:
        result = _simulate(sensors_dict, adjustments)

        section_header("Simulated Outcome")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            delta_card("SHI", baseline_shi, result["shi"])
        with c2:
            st.markdown(
                f"<div style='background:#161B22;border:1px solid #21262D;border-radius:8px;"
                f"padding:14px;text-align:center;'>"
                f"<div style='font-size:0.72rem;color:#8B949E;text-transform:uppercase;"
                f"letter-spacing:0.6px;margin-bottom:4px;'>State</div>"
                f"<div style='font-size:0.85rem;color:#C9D1D9;'>{baseline_state} → "
                f"<b>{result['state']}</b></div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            delta_card("RUL (days)", baseline_rul, result["rul"], fmt=".0f")
        with c4:
            st.plotly_chart(
                gauge_chart(result["shi"], title="Simulated SHI", max_val=1.0),
                use_container_width=True,
            )

        st.markdown("---")

        # ── Original vs Simulated curves ────────────────────────────────────
        section_header("Original vs Simulated Curves")
        for sname, sdata in sorted(sensors_dict.items()):
            vals = sdata.get("values", [])
            if not vals:
                continue
            timestamps = sdata.get("timestamps", list(range(len(vals))))
            mod_vals = result["modified_sensors"].get(sname, vals)

            traces = [
                {
                    "x": list(timestamps), "y": list(vals),
                    "name": "Original", "color": "#4C8BF5",
                },
                {
                    "x": list(timestamps), "y": list(mod_vals),
                    "name": "Simulated", "color": "#9B6BFF", "dash": "dash",
                },
            ]
            fig = multi_line_chart(
                traces,
                title=f"{sname} — Original vs Simulated",
                y_label="Value",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"whatif_{sname}")
    else:
        st.info("Adjust at least one sensor slider above to run a simulation.")

    st.markdown("---")
    disclaimer_banner("End of simulation. All outputs are approximate.")


def _demo_sensors() -> dict:
    rng = np.random.default_rng(42)
    n = 300
    t = list(range(n))
    sensors = {}
    for name in ["VIBRATION_X", "VIBRATION_Y", "TEMPERATURE", "PRESSURE", "CURRENT"]:
        base = rng.uniform(1.0, 5.0)
        sensors[name] = {"timestamps": t, "values": (base + rng.normal(0, base * 0.04, n)).tolist()}
    return {"sensors": sensors}
