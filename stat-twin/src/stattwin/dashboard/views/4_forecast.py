"""Page 4 — FAILURE FORECAST: live probability, RUL countdown, conformal bands."""
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
from stattwin.dashboard.components.charts import gauge_chart, multi_line_chart, timeline_chart
from stattwin.dashboard.components.live import (
    LIVE_INTERVAL,
    clip01,
    current_tick,
    live_jitter,
    live_scalar,
    live_status,
    live_wave,
    rul_countdown,
    slide_window,
)


def render() -> None:
    """Render the live Failure Forecast page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    page_header("Failure Forecast", "P(fail by +h) · RUL · conformal interval · warnings", machine)

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_forecast() -> None:
        live_status("forecast", feed="probability curve · RUL countdown · bands @ 2s")
        tick = current_tick()

        forecast = _load("failure_forecast.json", machine)
        warning = _load("warning_timeline.json", machine)
        conformal = _load("conformal_bands.json", machine)

        demo_mode = forecast is None
        if demo_mode:
            forecast = _demo_forecast()
            st.info(
                "Forecast data not found — showing **demo** data for UI validation."
            )

        fp30_base = float(forecast.get("failure_prob_30", 0.0))
        fp30 = clip01(
            live_scalar(fp30_base, tick, amp=0.012, name=f"{machine}:fp30")
        )
        rul_base = float(forecast.get("rul", 42.0))
        rul = rul_countdown(rul_base, tick)
        rul_ci = forecast.get("rul_ci", [None, None])
        has_ci = bool(rul_ci) and rul_ci[0] is not None and rul_ci[1] is not None

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card(
                "P(Failure +30 cycles)",
                f"{fp30:.1%}",
                delta=f"live · tick #{tick}",
                provenance="PREDICTED" if not demo_mode else "SIMULATED",
            )
        with c2:
            ci_str = f"[{rul_ci[0]:.0f} – {rul_ci[1]:.0f}]" if has_ci else ""
            kpi_card(
                "RUL (cycles)",
                f"{rul:.1f} {ci_str}".strip(),
                delta="counting down · live",
                provenance="PREDICTED" if not demo_mode else "SIMULATED",
            )
        with c3:
            kpi_card(
                "Model",
                str(forecast.get("model", "ensemble")),
                provenance="PREDICTED" if not demo_mode else "SIMULATED",
            )
        with c4:
            st.plotly_chart(
                gauge_chart(fp30, title="Failure Risk", max_val=1.0),
                width="stretch",
                key=f"fc_gauge_{tick}",
            )

        st.markdown("---")

        section_header("Multi-Horizon Failure Probability")
        horizon_probs = forecast.get("horizon_probs", {})
        if horizon_probs:
            horizons = sorted(horizon_probs.keys(), key=lambda h: int(h))
            base_probs = np.array(
                [float(horizon_probs[h]) for h in horizons], dtype=float
            )
            probs = np.clip(
                live_jitter(
                    base_probs,
                    tick,
                    amp=0.012,
                    name=f"{machine}:horizon_probs",
                ),
                0.0,
                1.0,
            )
            traces = [
                {
                    "x": [int(h) for h in horizons],
                    "y": probs.tolist(),
                    "name": "P(failure)",
                    "color": "#EF4444",
                }
            ]
            fig = multi_line_chart(
                traces,
                title="P(Failure) by Forecast Horizon",
                y_label="Probability",
            )
            if conformal and "upper" in conformal and "lower" in conformal:
                upper = np.asarray(conformal.get("upper", []), dtype=float)
                lower = np.asarray(conformal.get("lower", []), dtype=float)
                if len(upper) == len(horizons) and len(lower) == len(horizons):
                    upper = np.clip(
                        live_jitter(
                            upper, tick, amp=0.01, name=f"{machine}:conf_up"
                        ),
                        0.0,
                        1.0,
                    )
                    lower = np.clip(
                        live_jitter(
                            lower, tick, amp=0.01, name=f"{machine}:conf_lo"
                        ),
                        0.0,
                        1.0,
                    )
                    xs = [int(h) for h in horizons]
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=upper.tolist(),
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=lower.tolist(),
                            fill="tonexty",
                            mode="lines",
                            line=dict(width=0),
                            fillcolor="rgba(239,68,68,0.12)",
                            name="Conformal 90%",
                        )
                    )
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text="Warning threshold",
            )
            st.plotly_chart(fig, width="stretch", key=f"fc_horizon_{tick}")
        else:
            st.info("Horizon probabilities not available.")

        section_header("RUL Prediction")
        col_left, col_right = st.columns([1, 1])
        with col_left:
            rul_fig = go.Figure()
            rul_fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=rul,
                    number=dict(suffix=" cycles", font=dict(size=36)),
                    delta=dict(
                        reference=forecast.get("rul_prev", rul_base),
                        valueformat=".1f",
                        suffix=" c",
                    ),
                    title=dict(text="Remaining Useful Life", font=dict(size=14)),
                )
            )
            rul_fig.update_layout(
                paper_bgcolor="#111827",
                font=dict(color="#F9FAFB"),
                height=200,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(
                rul_fig, width="stretch", key=f"fc_rul_{tick}"
            )
        with col_right:
            if has_ci:
                ci_fig = go.Figure()
                ci_fig.add_trace(
                    go.Bar(
                        x=[rul_ci[1] - rul_ci[0]],
                        y=["RUL"],
                        base=[rul_ci[0]],
                        orientation="h",
                        marker_color="#3B82F6",
                        name="90% CI",
                    )
                )
                ci_fig.add_trace(
                    go.Scatter(
                        x=[rul],
                        y=["RUL"],
                        mode="markers",
                        marker=dict(size=14, color="#EF4444", symbol="diamond"),
                        name="Point estimate",
                    )
                )
                ci_fig.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#0A0E17",
                    font=dict(color="#F9FAFB"),
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=10),
                    xaxis=dict(title="Cycles", gridcolor="#1F2937"),
                    yaxis=dict(gridcolor="#1F2937"),
                    showlegend=True,
                )
                st.plotly_chart(
                    ci_fig, width="stretch", key=f"fc_ci_{tick}"
                )
            else:
                kpi_card("RUL", "—")

        st.markdown("---")

        section_header("Warning Timeline")
        if warning and warning.get("risk"):
            timestamps_w = list(warning.get("timestamps", []))
            alert_level = list(warning.get("alert_level", []))
            risk_vals = list(warning.get("risk", []))
            wwin = min(140, len(risk_vals))
            w0 = slide_window(len(risk_vals), tick, wwin, step=3)
            x_w = timestamps_w[w0 : w0 + wwin]
            y_w = np.asarray(risk_vals[w0 : w0 + wwin], dtype=float)
            bands = []
            upper_w = warning.get("upper")
            lower_w = warning.get("lower")
            if upper_w and lower_w and len(upper_w) == len(risk_vals):
                bands.append(
                    {
                        "upper": np.asarray(upper_w[w0 : w0 + wwin], dtype=float),
                        "lower": np.asarray(lower_w[w0 : w0 + wwin], dtype=float),
                        "label": "90% CI",
                        "fill": "rgba(239,68,68,0.10)",
                    }
                )
            fig = timeline_chart(
                x_w,
                y_w,
                title="Risk Trajectory with Alert Levels",
                y_label="Risk",
                bands=bands,
            )
            alert_colors = {"GREEN": "#10B981", "YELLOW": "#F59E0B", "RED": "#EF4444"}
            window_alerts = alert_level[w0 : w0 + wwin]
            if window_alerts and x_w:
                prev_al = window_alerts[0]
                start_i = 0
                for i in range(1, len(window_alerts)):
                    if window_alerts[i] != prev_al:
                        fig.add_vrect(
                            x0=x_w[start_i],
                            x1=x_w[i],
                            fillcolor=alert_colors.get(prev_al, "rgba(0,0,0,0)"),
                            opacity=0.08,
                            layer="below",
                            line_width=0,
                        )
                        prev_al = window_alerts[i]
                        start_i = i
                fig.add_vrect(
                    x0=x_w[start_i],
                    x1=x_w[-1],
                    fillcolor=alert_colors.get(prev_al, "rgba(0,0,0,0)"),
                    opacity=0.08,
                    layer="below",
                    line_width=0,
                )
            st.plotly_chart(fig, width="stretch", key=f"fc_warn_{tick}")
        else:
            if warning is None:
                st.caption("Warning timeline data not available — demo stream shown.")
            y_demo = live_wave(
                f"{machine}:warn",
                tick,
                n=140,
                base=0.3,
                scale=0.06,
                low=0.0,
                high=1.0,
            )
            fig = timeline_chart(
                list(range(140)),
                y_demo,
                title="Risk Trajectory (demo stream)",
                y_label="Risk",
                bands=[
                    {
                        "upper": np.minimum(1.0, y_demo + 0.05),
                        "lower": np.maximum(0.0, y_demo - 0.05),
                        "label": "90% CI",
                        "fill": "rgba(239,68,68,0.10)",
                    }
                ],
            )
            st.plotly_chart(fig, width="stretch", key=f"fc_warn_{tick}")

        section_header("Forecast Methodology")
        st.markdown(
            f"""
            <div class="st-card" style="font-size:0.84rem;color:#9CA3AF;line-height:1.55;">
                <b style="color:#F9FAFB;">Ensemble approach:</b> multi-horizon heads
                (+10…+50 cycles) with isotonic probability calibration and split-conformal
                RUL intervals (α = 0.10 target · 90%). Contributions are described as
                <i>contributed to the model's risk estimate</i>, never as causal failure claims.
                {provenance_badge('PREDICTED')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    live_forecast()


def _demo_forecast() -> dict:
    rng = np.random.default_rng(42)
    horizons = list(range(1, 91))
    probs = [min(1.0, 0.02 + 0.008 * h + rng.normal(0, 0.005)) for h in horizons]
    return {
        "failure_prob_30": probs[29] if len(probs) > 29 else 0.5,
        "rul": 42,
        "rul_ci": [28, 58],
        "model": "ensemble (demo)",
        "demo": True,
        "horizon_probs": {
            str(h): p for h, p in zip(horizons, probs, strict=False)
        },
        "timestamps": list(range(len(horizons))),
    }
