"""Page 5 — EXPLAINABILITY: live evidence cards and attribution bars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from stattwin.dashboard.components.cards import (
    evidence_card,
    page_header,
    provenance_badge,
    section_header,
)
from stattwin.dashboard.components.charts import bar_chart, sparkline
from stattwin.dashboard.components.live import (
    LIVE_INTERVAL,
    current_tick,
    live_jitter,
    live_scalar,
    live_status,
    live_window,
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


def _demo_explanation() -> dict:
    return {
        "summary": (
            "Rolling variance on VIBRATION_X drifted above its training baseline while "
            "the TEMPERATURE EWMA climbed through the watch band. Together these "
            "features contributed to the model's risk estimate over the reporting "
            "window. (demo text)"
        ),
        "risk_delta": 0.062,
        "timeframe": "last 30 days",
    }


def _demo_contributions() -> dict:
    return {
        "sensors": [
            {"name": "VIBRATION_X", "importance": 0.31},
            {"name": "TEMPERATURE", "importance": 0.24},
            {"name": "CURRENT", "importance": 0.18},
            {"name": "VIBRATION_Y", "importance": 0.15},
            {"name": "PRESSURE", "importance": 0.12},
        ],
        "types": [
            {"name": "mean shift", "importance": 0.28},
            {"name": "variance", "importance": 0.25},
            {"name": "EWMA", "importance": 0.20},
            {"name": "z-score", "importance": 0.15},
            {"name": "autocorrelation", "importance": 0.12},
        ],
    }


def _demo_evidence() -> list[dict]:
    return [
        {
            "title": "Vibration variance drift",
            "body": "Rolling variance is 2.3σ above the training baseline.",
            "sensor": "VIBRATION_X",
            "provenance": "OBSERVED",
            "severity": "warning",
        },
        {
            "title": "Temperature EWMA rise",
            "body": "EWMA crossed the watch threshold and is still climbing.",
            "sensor": "TEMPERATURE",
            "provenance": "OBSERVED",
            "severity": "warning",
        },
        {
            "title": "Data quality nominal",
            "body": "Completeness and plausibility remain above 98%.",
            "sensor": None,
            "provenance": "OBSERVED",
            "severity": "success",
        },
    ]


def _demo_sensor_data() -> dict:
    rng = np.random.default_rng(42)
    n = 500
    t = list(range(n))
    sensors = {}
    for name in ["VIBRATION_X", "VIBRATION_Y", "TEMPERATURE", "PRESSURE", "CURRENT"]:
        base = rng.uniform(0.5, 5.0)
        sensors[name] = {
            "timestamps": t,
            "values": (base + rng.normal(0, base * 0.05, n)).tolist(),
        }
    return {"sensors": sensors}


def render() -> None:
    """Render the live Explainability page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    page_header("Explainability", "risk delta · sensor & statistic contributions · evidence", machine)

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_explain() -> None:
        live_status("explain", feed="attribution + evidence stream @ 2s")
        tick = current_tick()

        explanation = _load("explanation.json", machine)
        contributions = _load("contributions.json", machine)
        evidence = _load("evidence.json", machine)
        sensor_data = _load("sensor_data.json", machine)

        demo_notes = []
        if explanation is None:
            explanation = _demo_explanation()
            demo_notes.append("explanation")
        if contributions is None:
            contributions = _demo_contributions()
            demo_notes.append("contributions")
        if evidence is None:
            evidence = _demo_evidence()
            demo_notes.append("evidence")
        if sensor_data is None:
            sensor_data = _demo_sensor_data()
            demo_notes.append("sensor data")
        if demo_notes:
            st.info(
                "Demo fallbacks in use for: "
                + ", ".join(sorted(set(demo_notes)))
                + " — place matching JSON files in `results/` to use real artifacts."
            )

        section_header("Why did risk increase?")
        reason = explanation.get("summary", "No explanation available.")
        risk_delta_base = float(explanation.get("risk_delta", 0.0))
        risk_delta = live_scalar(
            risk_delta_base,
            tick,
            amp=0.006,
            name=f"{machine}:risk_delta",
        )
        timeframe = explanation.get("timeframe", "last 30 days")
        st.markdown(
            f"""
            <div class="st-card" style="border-left:3px solid #F59E0B;">
                <div class="st-kpi-label">
                    Risk Delta:
                    <b style="color:#EF4444;">+{risk_delta:.1%}</b>
                    over <b>{timeframe}</b> · tick #{tick}
                    &nbsp;{provenance_badge('PREDICTED')}
                </div>
                <div style="color:#F9FAFB;font-size:0.9rem;line-height:1.55;">
                    {reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        section_header(
            "Ranked Sensor Contributions", "by impact on health degradation"
        )
        sensors_list = contributions.get("sensors", [])
        types_list = contributions.get("types", [])
        if sensors_list:
            s_names = [s["name"] for s in sensors_list]
            s_vals = np.array([float(s["importance"]) for s in sensors_list])
            amp = float(np.max(np.abs(s_vals)) or 1.0) * 0.05
            s_live = np.clip(
                live_jitter(s_vals, tick, amp=amp, name=f"{machine}:contrib_s"),
                0.0,
                1.0,
            )
            fig = bar_chart(
                s_names,
                s_live.tolist(),
                title="Sensor Importance (permutation) · live",
                y_label="Importance",
                horizontal=True,
            )
            st.plotly_chart(fig, width="stretch", key=f"ex_contrib_{tick}")

        if types_list:
            st.markdown(
                f"<div style='margin-top:8px;'>"
                f"<span style='font-size:0.84rem;color:#9CA3AF;"
                f"font-family:JetBrains Mono,monospace;'>"
                f"Contribution by statistic type "
                f"{provenance_badge('PREDICTED')}</span></div>",
                unsafe_allow_html=True,
            )
            t_names = [t["name"] for t in types_list]
            t_vals = np.array([float(t["importance"]) for t in types_list])
            amp_t = float(np.max(np.abs(t_vals)) or 1.0) * 0.05
            t_live = np.clip(
                live_jitter(t_vals, tick, amp=amp_t, name=f"{machine}:contrib_t"),
                0.0,
                1.0,
            )
            fig = bar_chart(
                t_names,
                t_live.tolist(),
                title="Statistical Feature Importance · live",
                y_label="Importance",
                color="#F59E0B",
                horizontal=True,
            )
            st.plotly_chart(fig, width="stretch", key=f"ex_types_{tick}")
        if not sensors_list and not types_list:
            st.info("Contribution data not available.")

        st.markdown("---")

        section_header("Supporting Evidence")
        items = evidence if isinstance(evidence, list) else evidence.get("items", [])
        for ev in items[:8]:
            live_tag = (
                f" <span style='color:#3B82F6;font-family:"
                f"'JetBrains Mono',monospace;font-size:0.7rem;'>· live #{tick}</span>"
            )
            evidence_card(
                ev.get("title", "Evidence"),
                f"{ev.get('body', '')}{live_tag}",
                sensor=ev.get("sensor"),
                provenance=ev.get("provenance", "OBSERVED"),
                severity=ev.get("severity", "info"),
            )

        st.markdown("---")

        section_header("Sensor Sparklines", "with contribution ranking")
        sensors_dict = sensor_data.get("sensors", {})
        ranked = contributions.get("sensors", [])
        top_sensors = [s["name"] for s in ranked if s["name"] in sensors_dict][:6]
        if top_sensors:
            cols = st.columns(min(len(top_sensors), 3))
            for i, sname in enumerate(top_sensors[:3]):
                sdata = sensors_dict[sname]
                _, vals = live_window(
                    f"{machine}:{sname}",
                    tick,
                    n=150,
                    base_values=sdata.get("values", []),
                    timestamps=sdata.get("timestamps"),
                )
                imp = next(
                    (s["importance"] for s in ranked if s["name"] == sname), 0
                )
                with cols[i]:
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.78rem;color:#9CA3AF;"
                        f"font-family:JetBrains Mono,monospace;'>"
                        f"{sname} <span style='color:#F59E0B;'>imp={float(imp):.3f}"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        sparkline(vals, color="#EF4444" if imp > 0.5 else "#3B82F6"),
                        width="stretch",
                        key=f"explain_spark_{sname}_{tick}",
                    )
        else:
            st.info("Sensor data or contributions unavailable for sparklines.")

    live_explain()
