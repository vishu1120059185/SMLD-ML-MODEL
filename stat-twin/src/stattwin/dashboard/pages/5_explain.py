"""Page 5 — EXPLAINABILITY: why risk increased, contributions, evidence."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from stattwin.dashboard.components.cards import evidence_card, provenance_badge, section_header
from stattwin.dashboard.components.charts import bar_chart, sparkline

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
    st.markdown(f"# 🔍 Explainability — {machine}")

    explanation = _load("explanation.json", machine)
    contributions = _load("contributions.json", machine)
    evidence = _load("evidence.json", machine)
    sensor_data = _load("sensor_data.json", machine)

    # ── "Why did risk increase?" panel ──────────────────────────────────────
    section_header("Why did risk increase?")
    if explanation:
        reason = explanation.get("summary", "No explanation available.")
        risk_delta = explanation.get("risk_delta", 0.0)
        timeframe = explanation.get("timeframe", "last 30 days")
        st.markdown(
            f"""
            <div style="background:#161B22;border-left:3px solid #F5A623;
                        border-radius:0 6px 6px 0;padding:16px 20px;margin-bottom:14px;">
                <div style="font-size:0.72rem;color:#8B949E;text-transform:uppercase;
                            letter-spacing:0.6px;margin-bottom:6px;">
                    Risk Delta: <b style="color:#D64545;">+{risk_delta:.1%}</b>
                    over <b>{timeframe}</b>
                    &nbsp;{provenance_badge('PREDICTED')}
                </div>
                <div style="color:#C9D1D9;font-size:0.9rem;line-height:1.5;">
                    {reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Explanation data not available in results/.")

    st.markdown("---")

    # ── Ranked contributions by sensor ───────────────────────────────────────
    section_header("Ranked Sensor Contributions", "by impact on health degradation")
    if contributions:
        sensors_list = contributions.get("sensors", [])
        types_list = contributions.get("types", [])

        if sensors_list:
            s_names = [s["name"] for s in sensors_list]
            s_vals = [s["importance"] for s in sensors_list]
            fig = bar_chart(
                s_names, s_vals,
                title="Sensor Importance (permutation)",
                y_label="Importance",
                horizontal=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        if types_list:
            st.markdown(f"<div style='margin-top:8px;'>"
                        f"<span style='font-size:0.82rem;color:#8B949E;'>"
                        f"Contribution by statistic type "
                        f"{provenance_badge('PREDICTED')}</span></div>",
                        unsafe_allow_html=True)
            t_names = [t["name"] for t in types_list]
            t_vals = [t["importance"] for t in types_list]
            fig = bar_chart(
                t_names, t_vals,
                title="Statistical Feature Importance",
                y_label="Importance",
                color="#F5A623",
                horizontal=True,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Contribution data not available.")

    st.markdown("---")

    # ── Evidence cards ───────────────────────────────────────────────────────
    section_header("Supporting Evidence")
    if evidence:
        items = evidence if isinstance(evidence, list) else evidence.get("items", [])
        for ev in items[:8]:
            evidence_card(
                ev.get("title", "Evidence"),
                ev.get("body", ""),
                sensor=ev.get("sensor"),
                provenance=ev.get("provenance", "OBSERVED"),
                severity=ev.get("severity", "info"),
            )
    else:
        evidence_card(
            "Baseline",
            "System operating within normal bounds.",
            provenance="OBSERVED",
            severity="success",
        )

    st.markdown("---")

    # ── Sensor sparklines with contribution overlay ──────────────────────────
    section_header("Sensor Sparklines", "with contribution ranking")
    if sensor_data and contributions:
        sensors_dict = sensor_data.get("sensors", {})
        ranked = contributions.get("sensors", [])
        top_sensors = [s["name"] for s in ranked[:6] if s["name"] in sensors_dict]

        if top_sensors:
            cols = st.columns(min(len(top_sensors), 3))
            for i, sname in enumerate(top_sensors[:3]):
                sdata = sensors_dict[sname]
                vals = sdata.get("values", [])
                imp = next((s["importance"] for s in ranked if s["name"] == sname), 0)
                with cols[i]:
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.78rem;color:#8B949E;'>"
                        f"{sname} <span style='color:#F5A623;'>imp={imp:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        sparkline(vals, color="#D64545" if imp > 0.5 else "#4C8BF5"),
                        use_container_width=True,
                        key=f"explain_spark_{sname}",
                    )
    else:
        st.info("Sensor data or contributions unavailable for sparklines.")
