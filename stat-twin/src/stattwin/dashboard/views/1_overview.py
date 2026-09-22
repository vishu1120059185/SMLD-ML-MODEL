"""Page 1 — OVERVIEW: machine health at a glance, live 2s auto-refresh."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from stattwin.dashboard.components.cards import (
    freshness_indicator,
    kpi_card,
    progress_card,
    recommendation_card,
    section_header,
    state_badge,
)
from stattwin.dashboard.components.charts import gauge_chart, timeline_chart
from stattwin.dashboard.components.live import (
    LIVE_INTERVAL,
    clip01,
    current_tick,
    live_scalar,
    live_status,
    live_wave,
    rul_countdown,
    slide_window,
    state_from_shi,
)

RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"

# Timestamp of last artifact load for freshness tracking
_last_load_key = "_overview_last_load"


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


def render() -> None:
    """Render the live Overview page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    st.markdown(f"# ⚙ Overview — {machine}")

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_overview() -> None:
        live_status("overview", feed="SHI · probability · risk stream @ 2s")
        tick = current_tick()

        health = _load("health_summary.json", machine)
        forecast = _load("failure_forecast.json", machine)
        recommendations = _load("recommendations.json", machine)
        dq = _load("data_quality.json", machine)
        timeline_data = _load("shi_timeline.json", machine)
        risk_data = _load("risk_timeline.json", machine)

        if health is None and forecast is None and timeline_data is None:
            st.info("No machine artifacts in `results/` — showing a demo live stream.")

        shi_base = float(health.get("shi", 0.28)) if health else 0.28
        shi = clip01(live_scalar(shi_base, tick, amp=0.03, name=f"{machine}:shi"))
        state = state_from_shi(shi)
        fp_base = float(forecast.get("failure_prob_30", 0.17)) if forecast else 0.17
        fp = clip01(live_scalar(fp_base, tick, amp=0.015, name=f"{machine}:fp"))
        rul_base = float(forecast.get("rul", 45.0)) if forecast else 45.0
        rul = rul_countdown(rul_base, tick)
        rul_ci = forecast.get("rul_ci", [None, None]) if forecast else [None, None]
        prov = "OBSERVED" if health else "SIMULATED"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div style='background:#111827;border:1px solid #1F2937;"
                f"border-radius:10px;padding:14px 16px;'>"
                f"<div style='font-size:0.72rem;color:#9CA3AF;text-transform:uppercase;"
                f"letter-spacing:0.8px;margin-bottom:6px;'>State · live</div>"
                f"<div style='margin-bottom:4px;'>{state_badge(state)}</div>"
                f"<div style='font-size:0.68rem;color:#9CA3AF;font-family:"
                f"'JetBrains Mono',monospace;'>tick #{tick}</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            kpi_card(
                "SHI",
                f"{shi:.3f}",
                delta=f"live · tick #{tick}",
                provenance=prov,
                tooltip="Statistical Health Index: 0 (healthy) to 1 (degraded)",
            )
        with col3:
            kpi_card(
                "P(Failure +30d)",
                f"{fp:.1%}",
                delta="live · simulated stream",
                provenance="PREDICTED",
                tooltip="Probability of failure within the next 30 cycles",
            )
        with col4:
            ci_str = (
                f"[{rul_ci[0]:.0f} – {rul_ci[1]:.0f}]"
                if rul_ci and rul_ci[0] is not None
                else ""
            )
            kpi_card(
                "RUL (days)",
                f"{rul:.1f} {ci_str}".strip(),
                delta="counting down · live",
                provenance="PREDICTED",
                tooltip="Remaining Useful Life: predicted cycles until failure",
            )

        left, right = st.columns([1, 2])
        with left:
            section_header("SHI Gauge")
            st.plotly_chart(
                gauge_chart(shi, title="Statistical Health Index"),
                width="stretch",
                key=f"ov_gauge_{tick}",
            )
        with right:
            section_header("Risk Timeline")
            if risk_data and risk_data.get("risk"):
                ts = list(risk_data.get("timestamps", []))
                vals = list(risk_data.get("risk", []))
                win = min(150, len(vals))
                i0 = slide_window(len(vals), tick, win, step=3)
                x = ts[i0 : i0 + win]
                y = np.asarray(vals[i0 : i0 + win], dtype=float)
                bands = []
                upper = risk_data.get("upper")
                lower = risk_data.get("lower")
                if upper and lower and len(upper) == len(vals):
                    bands.append(
                        {
                            "upper": np.asarray(upper[i0 : i0 + win], dtype=float),
                            "lower": np.asarray(lower[i0 : i0 + win], dtype=float),
                            "label": "95 % CI",
                            "fill": "rgba(76,139,245,0.10)",
                        }
                    )
                title = "Failure Risk Over Time"
            else:
                x = list(range(150))
                y = live_wave(
                    f"{machine}:risk", tick, n=150, base=0.35, scale=0.06, low=0.0, high=1.0
                )
                bands = [
                    {
                        "upper": np.minimum(1.0, y + 0.05),
                        "lower": np.maximum(0.0, y - 0.05),
                        "label": "95 % CI",
                        "fill": "rgba(76,139,245,0.10)",
                    }
                ]
                title = "Failure Risk Over Time (demo stream)"
            fig = timeline_chart(x, y, title=title, y_label="Risk", bands=bands)
            st.plotly_chart(fig, width="stretch", key=f"ov_risk_{tick}")

        section_header("Data Quality")
        if dq:
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                kpi_card(
                    "Completeness",
                    f"{dq.get('completeness', 0):.1%}",
                    tooltip="Fraction of expected sensor readings present",
                )
            with col_b:
                kpi_card(
                    "Timeliness",
                    f"{dq.get('timeliness', 0):.1%}",
                    tooltip="Freshness of data relative to expected update cadence",
                )
            with col_c:
                kpi_card(
                    "Plausibility",
                    f"{dq.get('plausibility', 0):.1%}",
                    tooltip="Fraction of values within physically plausible ranges",
                )
            with col_d:
                kpi_card(
                    "Overall DQ",
                    f"{dq.get('overall', 0):.1%}",
                    tooltip="Composite data quality score across all dimensions",
                )
        else:
            st.info("Data-quality metrics not available in results/.")

        section_header("Recommendations")
        if recommendations:
            recs = (
                recommendations if isinstance(recommendations, list) else [recommendations]
            )
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

        section_header("RUL Health")
        progress_card(
            "Remaining Useful Life",
            rul,
            maximum=rul_base if rul_base > 0 else 100.0,
            suffix=" days",
        )

        section_header("SHI Over Time")
        if timeline_data and timeline_data.get("shi"):
            ts = list(timeline_data.get("timestamps", []))
            vals = list(timeline_data.get("shi", []))
            win = min(150, len(vals))
            i0 = slide_window(len(vals), tick, win, step=3)
            x = ts[i0 : i0 + win]
            y = np.asarray(vals[i0 : i0 + win], dtype=float)
            y = np.clip(y + 0.008 * np.sin(np.arange(win) * 0.6 + tick * 0.5), 0.0, 1.0)
            title = "SHI Trajectory"
        else:
            x = list(range(150))
            y = live_wave(
                f"{machine}:shi_wave",
                tick,
                n=150,
                base=shi_base,
                scale=0.05,
                low=0.0,
                high=1.0,
            )
            title = "SHI Trajectory (demo stream)"
        fig = timeline_chart(x, y, title=title, y_label="SHI")
        st.plotly_chart(fig, width="stretch", key=f"ov_shi_{tick}")

    live_overview()
