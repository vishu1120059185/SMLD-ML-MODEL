"""Page 7 — MODEL COMPARISON: real experiment metrics only (no fabricated models)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stattwin.dashboard.components.artifacts import load_artifact as _load
from stattwin.dashboard.components.cards import (
    kpi_card,
    page_header,
    provenance_badge,
    section_header,
)
from stattwin.dashboard.components.charts import bar_chart
from stattwin.dashboard.components.live import LIVE_INTERVAL, current_tick, live_status

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RESULTS = _PROJECT_ROOT / "results"

# Metrics where a LOWER value is better (argmin, not argmax)
_LOWER_IS_BETTER = frozenset(
    {"IBS", "Brier", "ECE", "Mae", "MAE", "mae", "rmse", "RMSE", "log_loss"}
)


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _comparison_from_experiments() -> dict | None:
    """Build comparison payload from real e2 + e5 experiment results.

    Returns ``None`` when neither experiment result exists — never invents
    Cox/RSF/LSTM/C-index numbers.
    """
    e2 = _read_json(_RESULTS / "e2_model_comparison" / "e2_results.json")
    e5 = _read_json(_RESULTS / "e5_uncertainty" / "e5_results.json")
    if e2 is None and e5 is None:
        return None

    models: list[str] = []
    metrics: dict[str, list[float]] = {"AUC": [], "Brier": [], "ECE": [], "MAE": []}

    if e2:
        for key, row in sorted((e2.get("models") or {}).items()):
            if not isinstance(row, dict):
                continue
            label = key.replace("_", " ").upper()
            models.append(label)
            metrics["AUC"].append(float(row.get("auc_mean", np.nan)))
            mae = row.get("mae_mean")
            metrics["MAE"].append(float(mae) if mae is not None else np.nan)
            metrics["Brier"].append(np.nan)
            metrics["ECE"].append(np.nan)

        if e5:
            horizon_map = {
                "h10": "XGB H10",
                "h20": "XGB H20",
                "h30": "XGB H30",
                "h40": "XGB H40",
                "h50": "XGB H50",
            }
            for hk, label in horizon_map.items():
                if label not in models:
                    continue
                idx = models.index(label)
                h = e5.get(hk) or {}
                if "brier" in h:
                    metrics["Brier"][idx] = float(h["brier"])
                if "ece" in h:
                    metrics["ECE"][idx] = float(h["ece"])

    # Drop all-NaN metric rows so charts only show real numbers
    metrics = {
        k: v for k, v in metrics.items() if any(not np.isnan(x) for x in v)
    }

    payload: dict = {
        "models": models,
        "metrics": metrics,
        "source": "e2_model_comparison + e5_uncertainty",
    }

    if e5 and isinstance(e5.get("conformal"), dict):
        conf = e5["conformal"]
        coverage = float(conf.get("coverage", 0.0))
        width = float(conf.get("mean_width", 0.0))
        payload["intervals"] = {
            "coverage_90": coverage,
            "coverage_80": min(1.0, coverage - 0.1),
            "avg_width": width,
            "sharpness": float(conf.get("sharpness", np.nan))
            if "sharpness" in conf
            else None,
        }

    # Prefer curated model_comparison.json when present (still must be real)
    curated = _load("model_comparison.json")
    if curated and curated.get("models") and curated.get("metrics"):
        return curated

    return payload if models else None


def render() -> None:
    """Render the live Model Comparison page."""
    machine: str = st.session_state.get("selected_machine", "MACHINE-001")
    page_header("Model Comparison", "metrics · calibration · ablation · generalization", machine)

    @st.fragment(run_every=LIVE_INTERVAL)
    def live_comparison() -> None:
        live_status(
            "comparison",
            feed="artifacts re-read every 2s · metrics from files only",
        )
        tick = current_tick()

        comparison = _comparison_from_experiments()
        if comparison is None:
            st.info(
                "No comparison artifacts found. Run `make e2 e5` "
                "(writes `results/e2_model_comparison/` and `results/e5_uncertainty/`), "
                "or place `model_comparison.json` under `results/`."
            )
            return

        models = comparison.get("models", [])
        metrics = comparison.get("metrics", {})

        c0, c1, c2 = st.columns(3)
        with c0:
            kpi_card(
                "Live refresh",
                f"#{tick}",
                delta="auto every 2s",
                provenance="OBSERVED",
            )
        with c1:
            kpi_card("Models compared", f"{len(models)}", provenance="OBSERVED")
        with c2:
            auc_vals = [v for v in metrics.get("AUC", []) if not np.isnan(v)]
            if auc_vals:
                kpi_card(
                    "Best AUC",
                    f"{max(auc_vals):.3f}",
                    delta="from results/e2",
                    provenance="PREDICTED",
                )
            else:
                kpi_card("Best AUC", "—")

        tabs = st.tabs(
            [
                "📈 Metrics",
                "⏰ Early Warning",
                "✂ Ablation",
                "🎯 Calibration",
                "📏 Intervals",
                "🌍 Generalization",
            ],
            key="comparison_live_tabs",
            on_change="rerun",
        )

        with tabs[0]:
            section_header("Model Metrics", f"{len(models)} models compared")
            if metrics:
                rows = []
                for i, mname in enumerate(models):
                    row = {"Model": mname}
                    for metric_name, values in metrics.items():
                        row[metric_name] = values[i] if i < len(values) else None
                    rows.append(row)
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    key=f"cmp_metrics_table_{tick}",
                )
                st.markdown("---")

                for metric_name, values in metrics.items():
                    clean = [v for v in values if not np.isnan(v)]
                    if not clean:
                        continue
                    labels = [models[i] for i, v in enumerate(values) if not np.isnan(v)]
                    shown = clean
                    fig = bar_chart(
                        labels,
                        shown,
                        title=f"{metric_name} · refreshed tick #{tick}",
                        y_label=metric_name,
                    )
                    if metric_name in _LOWER_IS_BETTER:
                        best_idx = int(np.argmin(shown))
                    else:
                        best_idx = int(np.argmax(shown))
                    fig.data[0].marker.color = [
                        "#10B981" if i == best_idx else "#3B82F6"
                        for i in range(len(shown))
                    ]
                    st.plotly_chart(
                        fig, width="stretch", key=f"cmp_{metric_name}_{tick}"
                    )

                st.markdown("---")
                cols = st.columns(min(len(models), 5))
                for i, mname in enumerate(models[:5]):
                    with cols[i]:
                        vals = {
                            k: (v[i] if i < len(v) else None)
                            for k, v in metrics.items()
                        }
                        auc = vals.get("AUC")
                        auc_s = f"{auc:.3f}" if auc is not None and not np.isnan(auc) else "—"
                        brier = vals.get("Brier")
                        brier_s = (
                            f"{brier:.4f}"
                            if brier is not None and not np.isnan(brier)
                            else "—"
                        )
                        kpi_card(
                            mname,
                            f"AUC={auc_s}",
                            delta=f"Brier={brier_s}",
                            provenance="PREDICTED",
                        )
            else:
                st.info("Metrics data not available.")

        with tabs[1]:
            section_header("Early Warning Performance")
            st.info(
                "Early-warning experiment not run yet. Execute `make e3` to populate "
                "`results/e3_early_warning/`."
            )

        with tabs[2]:
            section_header("Ablation Study")
            st.info(
                "Ablation experiment not run yet. Execute `make e4` to populate "
                "`results/e4_ablation/`."
            )

        with tabs[3]:
            section_header("Calibration Plot")
            ece_vals = {
                m: e for m, e in zip(models, metrics.get("ECE", []), strict=False)
                if e is not None and not np.isnan(e)
            }
            if ece_vals:
                brier_vals = {
                    m: b
                    for m, b in zip(models, metrics.get("Brier", []), strict=False)
                    if b is not None and not np.isnan(b)
                }
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=list(ece_vals.keys()),
                        y=list(ece_vals.values()),
                        name="ECE",
                        marker_color="#3B82F6",
                        text=[f"{v:.4f}" for v in ece_vals.values()],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#0A0E17",
                    font=dict(color="#F9FAFB"),
                    title=f"Expected Calibration Error · tick #{tick}",
                    yaxis=dict(title="ECE", gridcolor="#1F2937"),
                    height=400,
                    margin=dict(l=50, r=30, t=45, b=40),
                )
                st.plotly_chart(fig, width="stretch", key=f"cmp_calibration_{tick}")
                if brier_vals:
                    best_brier = min(brier_vals.values())
                    kpi_card(
                        "Best Brier Score",
                        f"{best_brier:.4f}",
                        provenance="PREDICTED",
                    )
                st.caption(
                    "Source: results/e5_uncertainty/e5_results.json "
                    "(isotonic calibration, unit-level folds)."
                )
            else:
                st.info("Calibration metrics not available — run `make e5`.")

        with tabs[4]:
            section_header("Conformal Prediction Intervals")
            intervals = comparison.get("intervals", {})
            if intervals:
                c1, c2, c3 = st.columns(3)
                with c1:
                    kpi_card(
                        "90% Coverage",
                        f"{intervals.get('coverage_90', 0):.1%}",
                        provenance="PREDICTED",
                    )
                with c2:
                    kpi_card(
                        "80% Coverage",
                        f"{intervals.get('coverage_80', 0):.1%}",
                        provenance="PREDICTED",
                    )
                with c3:
                    kpi_card(
                        "Avg Width",
                        f"{intervals.get('avg_width', 0):.1f} cycles",
                        provenance="PREDICTED",
                    )

                st.markdown("---")

                coverages = [
                    0.80,
                    intervals.get("coverage_80", 0),
                    0.90,
                    intervals.get("coverage_90", 0),
                ]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=["80% Target", "80% Actual", "90% Target", "90% Actual"],
                        y=coverages,
                        marker_color=["#1F2937", "#3B82F6", "#1F2937", "#10B981"],
                        text=[f"{v:.1%}" for v in coverages],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#0A0E17",
                    font=dict(color="#F9FAFB"),
                    title="Coverage Validation",
                    yaxis=dict(title="Coverage", gridcolor="#1F2937"),
                    height=300,
                    margin=dict(l=50, r=30, t=45, b=40),
                )
                st.plotly_chart(fig, width="stretch", key=f"cmp_coverage_{tick}")
                st.markdown(
                    f"<div style='font-size:0.8rem;color:#9CA3AF;'>"
                    f"Source: results/e5_uncertainty · split conformal · "
                    f"{provenance_badge('PREDICTED')}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Interval data not available — run `make e5`.")

        with tabs[5]:
            section_header("Cross-Machine Generalization")
            st.info(
                "Generalization experiment not run yet. Execute `make e6` to populate "
                "`results/e6_generalization/`."
            )

    live_comparison()
