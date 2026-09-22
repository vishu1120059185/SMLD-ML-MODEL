"""Page 7 — MODEL COMPARISON: Metrics, Early Warning, Ablation, Calibration, Intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from stattwin.dashboard.components.cards import kpi_card, provenance_badge, section_header
from stattwin.dashboard.components.charts import bar_chart

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


def _demo_comparison() -> dict:
    models = ["Cox PH", "RSF", "LSTM", "Survival SVM", "Ensemble"]
    return {
        "models": models,
        "metrics": {
            "C-index": [0.78, 0.82, 0.79, 0.76, 0.86],
            "IBS": [0.22, 0.19, 0.21, 0.24, 0.17],
            "Brier": [0.18, 0.15, 0.17, 0.20, 0.13],
        },
        "early_warning": {
            "models": models,
            "lead_time_mean": [12.5, 18.3, 15.1, 10.8, 22.4],
            "alert_rate": [0.85, 0.92, 0.88, 0.79, 0.95],
        },
        "ablation": {
            "variants": ["Full model", "No SHI", "No DQ", "No conformal", "No EWMA"],
            "c_index": [0.86, 0.81, 0.83, 0.84, 0.82],
        },
        "calibration": {
            "bin_edges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "predicted": [0.05, 0.12, 0.18, 0.27, 0.38, 0.48, 0.57, 0.68, 0.78, 0.88],
            "observed": [0.04, 0.10, 0.19, 0.25, 0.36, 0.50, 0.55, 0.70, 0.82, 0.91],
        },
        "intervals": {
            "coverage_90": 0.91,
            "coverage_80": 0.82,
            "avg_width": 14.2,
            "sharpness": 0.78,
        },
        "generalization": {
            "machines": ["M-001", "M-002", "M-003", "M-004", "M-005"],
            "c_index": [0.86, 0.83, 0.81, 0.84, 0.79],
            "train_size": [1200, 980, 1100, 850, 1050],
        },
    }


def render():
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    st.markdown(f"# 📊 Model Comparison — {machine}")

    comparison = _load("model_comparison.json", machine)
    if comparison is None:
        comparison = _demo_comparison()
        st.info("Showing **demo** comparison data. Place `model_comparison.json` in results/.")

    models = comparison.get("models", [])

    tabs = st.tabs([
        "📈 Metrics",
        "⏰ Early Warning",
        "✂ Ablation",
        "🎯 Calibration",
        "📏 Intervals",
        "🌍 Generalization",
    ])

    # ── Tab 1: Metrics ──────────────────────────────────────────────────────
    with tabs[0]:
        section_header("Model Metrics", f"{len(models)} models compared")
        metrics = comparison.get("metrics", {})
        if metrics:
            for metric_name, values in metrics.items():
                fig = bar_chart(
                    models, values,
                    title=metric_name,
                    y_label=metric_name,
                )
                # Highlight best
                best_idx = np.argmax(values) if metric_name != "IBS" else np.argmin(values)
                fig.data[0].marker.color = [
                    "#2E9E6B" if i == best_idx else "#4C8BF5" for i in range(len(values))
                ]
                st.plotly_chart(fig, use_container_width=True)

            # Summary KPIs
            st.markdown("---")
            cols = st.columns(min(len(models), 5))
            for i, mname in enumerate(models[:5]):
                with cols[i]:
                    vals = {k: v[i] for k, v in metrics.items()}
                    kpi_card(
                        mname,
                        f"C={vals.get('C-index', 0):.3f}",
                        delta=f"IBS={vals.get('IBS', 0):.3f}  Brier={vals.get('Brier', 0):.3f}",
                    )
        else:
            st.info("Metrics data not available.")

    # ── Tab 2: Early Warning ────────────────────────────────────────────────
    with tabs[1]:
        section_header("Early Warning Performance")
        ew = comparison.get("early_warning", {})
        if ew:
            ew_models = ew.get("models", models)
            lead_times = ew.get("lead_time_mean", [])
            alert_rates = ew.get("alert_rate", [])

            col1, col2 = st.columns(2)
            with col1:
                fig = bar_chart(
                    ew_models, lead_times,
                    title="Mean Lead Time (days)",
                    y_label="Days",
                    color="#F5A623",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = bar_chart(
                    ew_models, alert_rates,
                    title="Alert Rate",
                    y_label="Rate",
                    color="#2E9E6B",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Early warning data not available.")

    # ── Tab 3: Ablation ─────────────────────────────────────────────────────
    with tabs[2]:
        section_header("Ablation Study")
        ablation = comparison.get("ablation", {})
        if ablation:
            variants = ablation.get("variants", [])
            c_indices = ablation.get("c_index", [])
            fig = bar_chart(
                variants, c_indices,
                title="C-Index by Model Variant",
                y_label="C-Index",
            )
            # Highlight full model
            fig.data[0].marker.color = [
                "#2E9E6B" if i == 0 else "#4C8BF5" for i in range(len(variants))
            ]
            st.plotly_chart(fig, use_container_width=True)

            if len(c_indices) > 1:
                delta_full = c_indices[0] - c_indices[-1]
                st.markdown(
                    f"<div style='background:#161B22;border:1px solid #21262D;border-radius:8px;"
                    f"padding:12px 16px;font-size:0.82rem;color:#8B949E;'>"
                    f"Full model vs no-EWMA: <b style='color:#C9D1D9;'>Δ={delta_full:+.3f}</b> "
                    f"C-index &nbsp;{provenance_badge('PREDICTED')}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Ablation data not available.")

    # ── Tab 4: Calibration ──────────────────────────────────────────────────
    with tabs[3]:
        section_header("Calibration Plot")
        cal = comparison.get("calibration", {})
        if cal:
            predicted = cal.get("predicted", [])
            observed = cal.get("observed", [])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predicted, y=observed,
                mode="markers+lines",
                name="Ensemble",
                line=dict(color="#4C8BF5", width=2),
                marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines", name="Perfect",
                line=dict(color="#8B949E", width=1, dash="dash"),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22", plot_bgcolor="#0E1117",
                font=dict(color="#C9D1D9"),
                title="Predicted vs Observed Failure Rate",
                xaxis=dict(title="Predicted", gridcolor="#21262D"),
                yaxis=dict(title="Observed", gridcolor="#21262D"),
                height=400,
                margin=dict(l=50, r=30, t=45, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Brier score
            brier = comparison.get("metrics", {}).get("Brier", [])
            if brier:
                best_brier = min(brier) if brier else None
                if best_brier is not None:
                    kpi_card("Best Brier Score", f"{best_brier:.4f}")
        else:
            st.info("Calibration data not available.")

    # ── Tab 5: Intervals ────────────────────────────────────────────────────
    with tabs[4]:
        section_header("Conformal Prediction Intervals")
        intervals = comparison.get("intervals", {})
        if intervals:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("90% Coverage", f"{intervals.get('coverage_90', 0):.1%}")
            with c2:
                kpi_card("80% Coverage", f"{intervals.get('coverage_80', 0):.1%}")
            with c3:
                kpi_card("Avg Width", f"{intervals.get('avg_width', 0):.1f} days")
            with c4:
                kpi_card("Sharpness", f"{intervals.get('sharpness', 0):.3f}")

            st.markdown("---")

            # Visual: coverage bar
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["80% Target", "80% Actual", "90% Target", "90% Actual"],
                y=[0.80, intervals.get("coverage_80", 0),
                   0.90, intervals.get("coverage_90", 0)],
                marker_color=["#21262D", "#4C8BF5", "#21262D", "#2E9E6B"],
                text=[f"{v:.1%}" for v in [0.80, intervals.get("coverage_80", 0),
                                             0.90, intervals.get("coverage_90", 0)]],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#161B22", plot_bgcolor="#0E1117",
                font=dict(color="#C9D1D9"),
                title="Coverage Validation",
                yaxis=dict(title="Coverage", gridcolor="#21262D"),
                height=300,
                margin=dict(l=50, r=30, t=45, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Interval data not available.")

    # ── Tab 6: Generalization ───────────────────────────────────────────────
    with tabs[5]:
        section_header("Cross-Machine Generalization")
        gen = comparison.get("generalization", {})
        if gen:
            gen_machines = gen.get("machines", [])
            gen_cindex = gen.get("c_index", [])

            fig = bar_chart(
                gen_machines, gen_cindex,
                title="C-Index per Machine",
                y_label="C-Index",
                color="#E8862F",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Train size vs performance
            train_sizes = gen.get("train_size", [])
            if train_sizes:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=train_sizes, y=gen_cindex,
                    mode="markers+text",
                    text=gen_machines,
                    textposition="top center",
                    marker=dict(size=12, color="#4C8BF5"),
                ))
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#161B22", plot_bgcolor="#0E1117",
                    font=dict(color="#C9D1D9"),
                    title="Train Size vs C-Index",
                    xaxis=dict(title="Training Samples", gridcolor="#21262D"),
                    yaxis=dict(title="C-Index", gridcolor="#21262D"),
                    height=350,
                    margin=dict(l=50, r=30, t=45, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)

            mean_ci = np.mean(gen_cindex) if gen_cindex else 0
            kpi_card("Mean C-Index (cross-machine)", f"{mean_ci:.3f}")
        else:
            st.info("Generalization data not available.")
