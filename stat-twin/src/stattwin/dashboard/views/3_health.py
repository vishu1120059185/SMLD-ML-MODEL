"""Page 3 — STATISTICAL HEALTH: live SHI, state bands, heatmap, variance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stattwin.dashboard.components.artifacts import load_artifact as _load
from stattwin.dashboard.components.cards import (
    kpi_card,
    page_header,
    section_header,
    state_badge,
)
from stattwin.dashboard.components.charts import heatmap_chart, timeline_chart
from stattwin.dashboard.components.live import (
    LIVE_INTERVAL,
    current_tick,
    live_jitter,
    live_scalar,
    live_status,
    slide_window,
    state_from_shi,
)


def _state_changes(timestamps: list, states: list) -> list[dict]:
    changes: list[dict] = []
    if not states or not timestamps:
        return changes
    prev_state = states[0]
    start_idx = 0
    for i in range(1, len(states)):
        if states[i] != prev_state:
            changes.append(
                {
                    "start": timestamps[start_idx],
                    "end": timestamps[i - 1],
                    "state": prev_state,
                }
            )
            prev_state = states[i]
            start_idx = i
    changes.append(
        {"start": timestamps[start_idx], "end": timestamps[-1], "state": prev_state}
    )
    return changes


def _demo_shi() -> dict:
    rng = np.random.default_rng(11)
    n = 240
    shi = np.clip(np.linspace(0.18, 0.44, n) + rng.normal(0, 0.02, n), 0.0, 1.0)
    return {
        "timestamps": list(range(n)),
        "shi": shi.tolist(),
        "states": [state_from_shi(v) for v in shi],
    }


def _demo_z_heatmap() -> dict:
    rng = np.random.default_rng(13)
    sensors = ["VIBRATION_X", "VIBRATION_Y", "TEMPERATURE", "PRESSURE", "CURRENT"]
    z = rng.normal(0.0, 1.0, (len(sensors), 60))
    drift = np.linspace(0.0, 1.2, 60) * rng.choice([-0.4, 0.5], size=(len(sensors), 1))
    return {
        "z": (z + drift).tolist(),
        "time_labels": [f"t{i}" for i in range(60)],
        "sensor_labels": sensors,
    }


def render() -> None:
    """Render the live Statistical Health page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    page_header("Statistical Health", "SHI · state bands · z heatmap · shift tables", machine)

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_health() -> None:
        live_status("health", feed="SHI trajectory · state bands · heatmap @ 2s")
        tick = current_tick()

        health = _load("health_summary.json", machine)
        shi_timeline = _load("shi_timeline.json", machine)
        z_heatmap = _load("z_heatmap.json", machine)
        variance_data = _load("variance_change.json", machine)
        dist_shift = _load("distribution_shift.json", machine)
        corr_matrix = _load("correlation_matrix.json", machine)

        demo_mode = shi_timeline is None and z_heatmap is None
        if demo_mode:
            st.info("Health artifacts not found — showing a demo live stream.")
        if shi_timeline is None:
            shi_timeline = _demo_shi()
        if z_heatmap is None:
            z_heatmap = _demo_z_heatmap()

        section_header("SHI Over Time", "with live state-transition bands")
        ts = list(shi_timeline.get("timestamps", []))
        shi_vals = list(shi_timeline.get("shi", []))
        win = min(140, len(shi_vals))
        i0 = slide_window(len(shi_vals), tick, win, step=3)
        x = ts[i0 : i0 + win]
        y = np.asarray(shi_vals[i0 : i0 + win], dtype=float)
        y = np.clip(y + 0.012 * np.sin(np.arange(win) * 0.55 + tick * 0.5), 0.0, 1.0)
        states = [state_from_shi(v) for v in y]
        fig = timeline_chart(
            x,
            y.tolist(),
            title="Statistical Health Index",
            y_label="SHI",
            state_changes=_state_changes(list(x), states),
        )
        fig.add_hline(
            y=0.25,
            line_dash="dot",
            line_color="#F59E0B",
            line_width=1,
            annotation_text="WATCH",
            annotation_position="right",
        )
        fig.add_hline(
            y=0.50,
            line_dash="dot",
            line_color="#F97316",
            line_width=1,
            annotation_text="DEGRADING",
            annotation_position="right",
        )
        fig.add_hline(
            y=0.75,
            line_dash="dot",
            line_color="#EF4444",
            line_width=1,
            annotation_text="CRITICAL",
            annotation_position="right",
        )
        st.plotly_chart(fig, width="stretch", key=f"health_shi_{tick}")

        shi_base = float(health.get("shi", float(np.mean(y)))) if health else float(np.mean(y))
        shi_live = float(
            np.clip(
                live_scalar(shi_base, tick, amp=0.02, name=f"{machine}:health_shi"),
                0.0,
                1.0,
            )
        )
        state = state_from_shi(shi_live)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='st-kpi' style='text-align:center;'>"
                f"<div class='st-kpi-label' style='justify-content:center;'>"
                f"Current State · live</div>"
                f"{state_badge(state)}"
                f"<div class='st-kpi-delta'>tick #{tick}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            kpi_card(
                "SHI",
                f"{shi_live:.4f}",
                delta=f"tick #{tick}",
                provenance="OBSERVED" if health else "SIMULATED",
            )
        with c3:
            kpi_card(
                "State",
                state,
                delta="re-derived from live SHI",
                provenance="OBSERVED" if health else "SIMULATED",
            )

        section_header("Z-Score Heatmap", "sensor × time · cells shift every 2s")
        z = np.asarray(z_heatmap.get("z", [[]]), dtype=float)
        n_cols = z.shape[1] if z.ndim == 2 else 0
        x_labels = z_heatmap.get(
            "time_labels", [f"t{i}" for i in range(n_cols)]
        )
        y_labels = z_heatmap.get(
            "sensor_labels", [f"s{i}" for i in range(z.shape[0])]
        )
        if n_cols > 30:
            c0 = slide_window(n_cols, tick, 30, step=2)
            z = z[:, c0 : c0 + 30]
            x_labels = list(x_labels[c0 : c0 + 30])
        z = live_jitter(z, tick, amp=0.18, name=f"{machine}:z_heatmap")
        fig = heatmap_chart(z, list(x_labels), list(y_labels), title="Rolling Z-Score Heatmap")
        st.plotly_chart(fig, width="stretch", key=f"health_heat_{tick}")

        section_header("Variance-Change Detection")
        if variance_data and variance_data.get("sensors"):
            sensors_v = variance_data["sensors"]
            tab_names = list(sensors_v.keys())[:6]
            var_tabs = st.tabs(tab_names, key="health_var_tabs", on_change="rerun")
            for idx, sname in enumerate(tab_names):
                with var_tabs[idx]:
                    svar = sensors_v[sname]
                    timestamps_v = list(svar.get("timestamps", []))
                    var_vals = list(svar.get("variance", []))
                    changepoints = svar.get("changepoints", [])
                    if not var_vals:
                        st.info("No variance series for this sensor.")
                        continue
                    vwin = min(140, len(var_vals))
                    v0 = slide_window(len(var_vals), tick, vwin, step=3)
                    x_v = timestamps_v[v0 : v0 + vwin]
                    y_v = np.asarray(var_vals[v0 : v0 + vwin], dtype=float)
                    var_amp = float(np.std(y_v) or 0.1) * 0.08
                    y_v = live_jitter(
                        y_v,
                        tick,
                        amp=var_amp,
                        name=f"{machine}:var:{sname}",
                    )
                    fig = timeline_chart(
                        x_v,
                        y_v.tolist(),
                        title=f"{sname} — Rolling Variance",
                        y_label="Variance",
                    )
                    for cp in changepoints:
                        cp_idx = cp if isinstance(cp, (int, float)) else cp.get("index", 0)
                        cp_idx = int(cp_idx) - v0
                        if 0 <= cp_idx < len(x_v):
                            fig.add_vline(
                                x=x_v[cp_idx],
                                line_dash="dash",
                                line_color="#EF4444",
                                line_width=1.5,
                                annotation_text="Change",
                            )
                    st.plotly_chart(
                        fig, width="stretch", key=f"health_var_{sname}_{tick}"
                    )
        else:
            st.info("Variance-change data not available.")

        section_header("Distribution-Shift Table")
        if dist_shift:
            rows = (
                dist_shift
                if isinstance(dist_shift, list)
                else dist_shift.get("rows", [])
            )
            if rows:
                df = pd.DataFrame(rows)
                expected = {
                    "sensor": st.column_config.TextColumn("Sensor"),
                    "ks_stat": st.column_config.NumberColumn("KS Statistic", format="%.4f"),
                    "ks_p": st.column_config.NumberColumn("KS p-value", format="%.4f"),
                    "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                    "shifted": st.column_config.CheckboxColumn("Shifted"),
                }
                config = {k: v for k, v in expected.items() if k in df.columns}
                st.dataframe(
                    df,
                    width="stretch",
                    key=f"health_dist_{tick}",
                    column_config=config or None,
                )
            else:
                st.info("No distribution-shift rows found.")
        else:
            st.info("Distribution-shift table not available.")

        section_header("Correlation Matrix")
        if corr_matrix:
            c = np.asarray(corr_matrix.get("matrix", [[]]), dtype=float)
            labels = corr_matrix.get("labels", [f"s{i}" for i in range(c.shape[0])])
            fig = heatmap_chart(
                c,
                list(labels),
                list(labels),
                title="Sensor Correlation Matrix",
                colorscale="RdBu_r",
            )
            st.plotly_chart(fig, width="stretch", key=f"health_corr_{tick}")
        else:
            st.info("Correlation matrix not available.")

    live_health()
