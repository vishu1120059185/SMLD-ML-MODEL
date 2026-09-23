"""Page 6 — WHAT-IF SIMULATOR: sliders outside, live comparison inside."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from stattwin.dashboard.components.cards import (
    delta_card,
    disclaimer_banner,
    kpi_card,
    page_header,
    provenance_badge,
    section_header,
)
from stattwin.dashboard.components.charts import gauge_chart, multi_line_chart
from stattwin.dashboard.components.live import (
    LIVE_INTERVAL,
    clip01,
    current_tick,
    live_scalar,
    live_status,
    live_window,
    rul_countdown,
    state_from_shi,
)

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


def _simulate(sensors: dict, adjustments: dict) -> dict:
    """Heuristic what-if proxy: shift sensors, re-estimate SHI, state and RUL."""
    modified = {}
    for name, data in sensors.items():
        vals = list(data.get("values", []))
        if name in adjustments:
            delta = adjustments[name]
            vals = [v + delta for v in vals]
        modified[name] = vals

    all_stds = []
    weights = {
        "VIBRATION_X": 1.5,
        "VIBRATION_Y": 1.5,
        "TEMPERATURE": 1.0,
        "PRESSURE": 0.8,
        "CURRENT": 1.2,
    }
    for name, vals in modified.items():
        w = weights.get(name, 1.0)
        if vals:
            all_stds.append(float(np.std(vals)) * w)
    shi_sim = min(1.0, float(np.mean(all_stds)) * 0.3) if all_stds else 0.0
    state = state_from_shi(shi_sim)
    rul_sim = max(1, int(90 * (1 - shi_sim)))

    return {
        "modified_sensors": modified,
        "shi": shi_sim,
        "state": state,
        "rul": rul_sim,
    }


def render() -> None:
    """Render the live What-If Simulator page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    page_header("What-If Simulator", "counterfactual sensitivity · SIMULATION only", machine)

    disclaimer_banner(
        "SIMULATION — not real operational data. Results are approximate proxies."
    )

    sensor_data = _load("sensor_data.json", machine)
    if sensor_data is None:
        st.warning("Sensor data not found. Using demo data.")
        sensor_data = _demo_sensors()

    sensors_dict = sensor_data.get("sensors", {})
    if not sensors_dict:
        st.error("No sensors available.")
        return

    section_header("Original Baseline")

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_baseline() -> None:
        live_status(
            "whatif-baseline",
            feed="SIMULATION page · baseline stream @ 2s",
        )
        tick = current_tick()
        health = _load("health_summary.json", machine)
        forecast = _load("failure_forecast.json", machine)

        shi_base = float(health.get("shi", 0.24)) if health else 0.24
        shi_live = clip01(
            live_scalar(shi_base, tick, amp=0.02, name=f"{machine}:wi_shi")
        )
        state = state_from_shi(shi_live)
        rul_base = float(forecast.get("rul", 45)) if forecast else 45.0
        rul_live = rul_countdown(rul_base, tick)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card(
                "SHI",
                f"{shi_live:.3f}",
                delta=f"tick #{tick}",
                provenance="OBSERVED" if health else "SIMULATED",
            )
        with c2:
            kpi_card("State", state, provenance="OBSERVED" if health else "SIMULATED")
        with c3:
            kpi_card(
                "RUL",
                f"{rul_live:.1f} days",
                delta="counting down · live",
                provenance="PREDICTED",
            )

    live_baseline()

    st.markdown("---")

    section_header("Sensor Adjustments")
    st.markdown(
        f"<div style='color:#9CA3AF;font-size:0.84rem;margin-bottom:12px;"
        f"font-family:JetBrains Mono,monospace;'>"
        f"Adjust sensor offsets · + increase · − decrease "
        f"{provenance_badge('SIMULATED')}</div>",
        unsafe_allow_html=True,
    )

    adjustments: dict[str, float] = {}
    slider_cols = st.columns(min(len(sensors_dict), 3))
    for i, (sname, sdata) in enumerate(sorted(sensors_dict.items())):
        vals = sdata.get("values", [])
        if not vals:
            continue
        current_mean = float(np.mean(vals))
        std = float(np.std(vals)) or abs(current_mean) * 0.1 or 0.1
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
                adjustments[sname] = float(adj)

    st.markdown("---")

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_simulation() -> None:
        live_status(
            "whatif-sim",
            feed="SIMULATION · original vs simulated comparison @ 2s",
        )
        tick = current_tick()

        if not adjustments:
            st.info("Adjust at least one sensor slider above to run a simulation.")
            return

        health = _load("health_summary.json", machine)
        forecast = _load("failure_forecast.json", machine)
        shi_base = float(health.get("shi", 0.24)) if health else 0.24
        baseline_shi = clip01(
            live_scalar(shi_base, tick, amp=0.02, name=f"{machine}:wi_shi")
        )
        baseline_state = state_from_shi(baseline_shi)
        rul_base = float(forecast.get("rul", 45)) if forecast else 45.0
        baseline_rul = rul_countdown(rul_base, tick)

        live_sensors: dict = {}
        for sname, sdata in sensors_dict.items():
            _, vals = live_window(
                f"{machine}:{sname}",
                tick,
                n=min(200, max(50, len(sdata.get("values", []) or []))),
                base_values=sdata.get("values", []),
                timestamps=sdata.get("timestamps"),
            )
            live_sensors[sname] = {"values": vals.tolist()}

        result = _simulate(live_sensors, adjustments)

        section_header("Simulated Outcome")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            delta_card("SHI", baseline_shi, float(result["shi"]))
        with c2:
            st.markdown(
                f"<div class='st-kpi' style='text-align:center;'>"
                f"<div class='st-kpi-label' style='justify-content:center;'>State</div>"
                f"<div style='font-size:0.95rem;color:#F9FAFB;font-family:'JetBrains Mono',monospace;'>"
                f"{baseline_state} → <b>{result['state']}</b></div>"
                f"<div class='st-kpi-delta'>SIMULATED</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            delta_card(
                "RUL (days)",
                baseline_rul,
                float(result["rul"]),
                fmt=".1f",
            )
        with c4:
            st.plotly_chart(
                gauge_chart(float(result["shi"]), title="Simulated SHI", max_val=1.0),
                width="stretch",
                key=f"wi_gauge_{tick}",
            )

        st.markdown("---")

        section_header("Original vs Simulated Curves")
        for sname in sorted(sensors_dict):
            sdata = sensors_dict[sname]
            if not sdata.get("values"):
                continue
            timestamps, orig = live_window(
                f"{machine}:{sname}",
                tick,
                n=min(200, max(50, len(sdata.get("values", []) or []))),
                base_values=sdata.get("values", []),
                timestamps=sdata.get("timestamps"),
            )
            delta = adjustments.get(sname, 0.0)
            mod = orig + delta
            traces = [
                {
                    "x": list(timestamps),
                    "y": orig.tolist(),
                    "name": "Original",
                    "color": "#3B82F6",
                },
                {
                    "x": list(timestamps),
                    "y": mod.tolist(),
                    "name": "Simulated",
                    "color": "#8B5CF6",
                    "dash": "dash",
                },
            ]
            fig = multi_line_chart(
                traces,
                title=f"{sname} — Original vs Simulated",
                y_label="Value",
            )
            st.plotly_chart(fig, width="stretch", key=f"whatif_{sname}_{tick}")

    live_simulation()

    st.markdown("---")
    disclaimer_banner("End of simulation. All outputs are approximate.")


def _demo_sensors() -> dict:
    rng = np.random.default_rng(42)
    n = 300
    t = list(range(n))
    sensors = {}
    for name in ["VIBRATION_X", "VIBRATION_Y", "TEMPERATURE", "PRESSURE", "CURRENT"]:
        base = rng.uniform(1.0, 5.0)
        sensors[name] = {
            "timestamps": t,
            "values": (base + rng.normal(0, base * 0.04, n)).tolist(),
        }
    return {"sensors": sensors}
