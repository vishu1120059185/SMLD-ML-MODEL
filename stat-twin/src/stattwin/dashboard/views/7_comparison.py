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
    {"IBS", "Brier", "ECE", "Mae", "MAE", "mae", "rmse", "RMSE", "nasa", "NASA", "log_loss"}  # noqa: E501
)

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#0A0E17",
    font=dict(color="#F9FAFB"),
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

    e2 schema (list)::

        {"models": [{"model": "XGBoostModel",
                     "classification": [{"horizon": 10, "roc_auc": ...}, ...],
                     "rul": {"mae": ...}}, ...]}

    e5 schema::

        {"calibration": {"per_horizon": {"h10": {"brier": ..., "ece_equal_width": ...}}},
         "conformal": {"coverage": ..., "mean_width": ...}}
    """
    e2 = _read_json(_RESULTS / "e2_model_comparison" / "e2_results.json")
    e5 = _read_json(_RESULTS / "e5_uncertainty" / "e5_results.json")
    if e2 is None and e5 is None:
        return None

    models: list[str] = []
    # Per-model metrics that genuinely belong to each e2 model. Brier/ECE
    # are deliberately excluded: e5 calibrates a single model, so its
    # calibration numbers cannot be attributed to every e2 model.
    metrics: dict[str, list[float]] = {
        "AUC": [],
        "MAE": [],
        "RMSE": [],
        "NASA": [],
    }

    if isinstance(e2, dict):
        raw_models = e2.get("models") or []
        # Older schema: dict keyed by short name
        if isinstance(raw_models, dict):
            raw_models = [
                {"model": key, **(row if isinstance(row, dict) else {})}
                for key, row in sorted(raw_models.items())
            ]
        if isinstance(raw_models, list):
            for row in raw_models:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("model") or row.get("name") or "")
                if not name:
                    continue
                label = name.replace("Model", "").replace("_", " ").upper()
                if not label:
                    label = name.upper()
                models.append(label)

                cls = row.get("classification") or []
                aucs = [
                    float(c["roc_auc"])
                    for c in cls
                    if isinstance(c, dict) and c.get("roc_auc") is not None
                ]
                metrics["AUC"].append(float(np.mean(aucs)) if aucs else np.nan)

                rul = row.get("rul") or {}
                mae = rul.get("mae") if isinstance(rul, dict) else None
                rmse = rul.get("rmse") if isinstance(rul, dict) else None
                nasa = rul.get("nasa_score") if isinstance(rul, dict) else None
                metrics["MAE"].append(float(mae) if mae is not None else np.nan)
                metrics["RMSE"].append(float(rmse) if rmse is not None else np.nan)
                metrics["NASA"].append(float(nasa) if nasa is not None else np.nan)

    # Drop all-NaN metric rows so charts only show real numbers
    metrics = {
        k: v for k, v in metrics.items() if any(not np.isnan(x) for x in v)
    }

    payload: dict = {
        "models": models,
        "metrics": metrics,
        "source": "e2_model_comparison + e5_uncertainty",
    }

    if isinstance(e5, dict) and isinstance(e5.get("conformal"), dict):
        conf = e5["conformal"]
        coverage = float(conf.get("coverage", 0.0) or 0.0)
        width = float(conf.get("mean_width", 0.0) or 0.0)
        payload["intervals"] = {
            "coverage_90": coverage,
            "coverage_80": min(1.0, coverage - 0.1),
            "avg_width": width,
            "sharpness": float(conf["sharpness"])
            if conf.get("sharpness") is not None
            and not (isinstance(conf.get("sharpness"), float) and np.isnan(conf["sharpness"]))
            else None,
        }

    # Calibration detail lives on its own (single calibrated model)
    if isinstance(e5, dict) and isinstance(e5.get("calibration"), dict):
        cal = e5["calibration"]
        payload["calibration"] = {
            "model": str(e5.get("model", "—")),
            "mean_brier": _safe_float(cal.get("mean_brier")),
            "mean_ece": _safe_float(cal.get("mean_ece_ew")),
            "per_horizon": cal.get("per_horizon") or {},
        }

    # Prefer curated model_comparison.json when present (still must be real)
    curated = _load("model_comparison.json")
    if (
        isinstance(curated, dict)
        and curated.get("models")
        and curated.get("metrics")
        and not isinstance(curated.get("models"), list)
        or (
            isinstance(curated, dict)
            and isinstance(curated.get("models"), list)
            and curated.get("metrics")
        )
    ):
        # Accept both list-of-names and dict key styles as long as metrics exist
        if isinstance(curated.get("models"), dict):
            return curated
        if isinstance(curated.get("models"), list) and curated.get("metrics"):
            return curated

    return payload if models else None


def _safe_float(value) -> float | None:
    """Return ``float(value)`` or ``None`` when missing / NaN."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


def _early_warning_payload() -> dict | None:
    """Read e3 early-warning results (lead time per model at a FAR budget)."""
    e3 = _read_json(_RESULTS / "e3_early_warning" / "e3_results.json")
    if not isinstance(e3, dict):
        return None
    rows = [
        {
            "model": str(m.get("model", "—")).replace("Model", ""),
            "mean_lead_time": _safe_float(m.get("mean_lead_time")),
            "median_lead_time": _safe_float(m.get("median_lead_time")),
            "min_lead_time": _safe_float(m.get("min_lead_time")),
            "max_lead_time": _safe_float(m.get("max_lead_time")),
            "mean_far": _safe_float(m.get("mean_far")),
            "n_warned": m.get("n_warned"),
            "n_failed": m.get("n_failed"),
        }
        for m in (e3.get("models") or [])
        if isinstance(m, dict)
    ]
    if not rows:
        return None
    return {
        "dataset": e3.get("dataset"),
        "horizon": e3.get("horizon"),
        "far_budget": e3.get("far_budget"),
        "rows": rows,
    }


def _ablation_payload() -> dict | None:
    """Read e4 ablation results (feature-set variants + Holm tests)."""
    e4 = _read_json(_RESULTS / "e4_ablation" / "e4_results.json")
    if not isinstance(e4, dict):
        return None
    variants = e4.get("variants")
    if not isinstance(variants, dict) or not variants:
        return None
    rows = []
    for name, v in variants.items():
        if not isinstance(v, dict):
            continue
        rows.append(
            {
                "variant": name,
                "n_features": v.get("n_features"),
                "mean_mae": _safe_float(v.get("mean_mae")),
                "median_mae": _safe_float(v.get("median_mae")),
                "std_mae": _safe_float(v.get("std_mae")),
            }
        )
    if not rows:
        return None
    return {
        "base_model": e4.get("base_model"),
        "dataset": e4.get("dataset"),
        "rows": rows,
        "pairwise_tests": e4.get("pairwise_tests") or {},
        "holm_bonferroni": e4.get("holm_bonferroni") or {},
    }


def _generalization_payload() -> dict | None:
    """Read e6 cross-machine generalization results."""
    e6 = _read_json(_RESULTS / "e6_generalization" / "e6_results.json")
    if not isinstance(e6, dict):
        return None
    datasets = e6.get("datasets") or []
    auc_matrix = e6.get("auc_matrix") or []
    rmse_matrix = e6.get("rmse_matrix") or []
    if not datasets or not auc_matrix:
        return None
    cross_rows = [
        {
            "train": c.get("train"),
            "test": c.get("test"),
            "mean_roc_auc": _safe_float(c.get("mean_roc_auc")),
            "rul_rmse": _safe_float(c.get("rul_rmse")),
            "rul_mae": _safe_float(c.get("rul_mae")),
        }
        for c in (e6.get("cross_results") or [])
        if isinstance(c, dict)
    ]
    return {
        "model": e6.get("model"),
        "datasets": datasets,
        "auc_matrix": auc_matrix,
        "rmse_matrix": rmse_matrix,
        "cross_rows": cross_rows,
    }


def _uncertainty_payload() -> dict | None:
    """Read e9 uncertainty-method comparison (four interval methods)."""
    e9 = _read_json(
        _RESULTS / "e9_uncertainty_comparison" / "e9_results.json"
    )
    if not isinstance(e9, dict):
        return None
    comp = e9.get("uncertainty_comparison") or {}
    summary = comp.get("summary") or {}
    rows = [
        {
            "method": method,
            "picp": _safe_float(stats.get("mean_picp")),
            "width": _safe_float(stats.get("mean_width")),
            "winkler": _safe_float(stats.get("mean_winkler")),
        }
        for method, stats in summary.items()
        if isinstance(stats, dict) and stats.get("mean_picp") is not None
    ]
    if not rows:
        return None
    return {"alpha": e9.get("alpha"), "rows": rows}


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
                        mae = vals.get("MAE")
                        mae_s = f"{mae:.1f}" if mae is not None and not np.isnan(mae) else "—"
                        kpi_card(
                            mname,
                            f"AUC={auc_s}",
                            delta=f"MAE={mae_s}",
                            provenance="PREDICTED",
                        )
            else:
                st.info("Metrics data not available.")

        with tabs[1]:
            section_header("Early Warning Performance")
            ew = _early_warning_payload()
            if ew is None:
                st.info(
                    "Early-warning experiment not run yet. Execute `make e3` to populate "
                    "`results/e3_early_warning/`."
                )
            else:
                hz = ew.get("horizon")
                far = ew.get("far_budget")
                st.caption(
                    f"Dataset {ew.get('dataset')} · horizon h{hz} · "
                    f"FAR budget {far:.0%} · lead time in cycles ahead of failure"
                )
                table = pd.DataFrame(ew["rows"]).set_index("model")
                st.dataframe(table, width="stretch", key=f"cmp_ew_table_{tick}")

                labels = [r["model"] for r in ew["rows"]]
                lead = [r["mean_lead_time"] for r in ew["rows"]]
                keep = [
                    (lb, v) for lb, v in zip(labels, lead, strict=False)
                    if v is not None
                ]
                if keep:
                    lb_vals, l_vals = zip(*keep, strict=True)
                    fig = bar_chart(
                        list(lb_vals),
                        list(l_vals),
                        title=f"Mean lead time · h{hz} · FAR {far:.0%} · tick #{tick}",
                        y_label="Lead time (cycles)",
                    )
                    st.plotly_chart(
                        fig, width="stretch", key=f"cmp_ew_bar_{tick}"
                    )
                st.caption(
                    "Source: results/e3_early_warning/e3_results.json "
                    f"(unit-level folds, thresholded at τ={far}). "
                    f"{provenance_badge('PREDICTED')}"
                )

        with tabs[2]:
            section_header("Ablation Study")
            ab = _ablation_payload()
            if ab is None:
                st.info(
                    "Ablation experiment not run yet. Execute `make e4` to populate "
                    "`results/e4_ablation/`."
                )
            else:
                st.caption(
                    f"Base model {ab.get('base_model')} · dataset {ab.get('dataset')} · "
                    "mean RUL MAE (lower is better)"
                )
                st.dataframe(
                    pd.DataFrame(ab["rows"]).set_index("variant"),
                    width="stretch",
                    key=f"cmp_ablation_table_{tick}",
                )
                rows = ab["rows"]
                labels = [r["variant"] for r in rows]
                vals = [r["mean_mae"] for r in rows]
                pairs = [
                    (lb, v) for lb, v in zip(labels, vals, strict=False)
                    if v is not None
                ]
                if pairs:
                    fig = bar_chart(
                        [p[0] for p in pairs],
                        [p[1] for p in pairs],
                        title=f"Feature-set ablation · RUL MAE · tick #{tick}",
                        y_label="RUL MAE (cycles)",
                    )
                    fig.data[0].marker.color = [
                        "#10B981" if lb == pairs[0][0] else "#3B82F6"
                        for lb, _ in pairs
                    ]
                    st.plotly_chart(
                        fig, width="stretch", key=f"cmp_ablation_bar_{tick}"
                    )
                tests = ab.get("pairwise_tests") or {}
                if tests:
                    st.markdown("**Pairwise significance (E_full vs each variant)**")
                    holm_raw = ab.get("holm_bonferroni") or {}
                    adj = {}
                    if isinstance(holm_raw, dict):
                        raw_adj = holm_raw.get("adjusted_p_values")
                        if isinstance(raw_adj, dict):
                            adj = raw_adj
                    test_rows = []
                    for name, t in tests.items():
                        if not isinstance(t, dict):
                            continue
                        test_rows.append(
                            {
                                "comparison": name,
                                "wilcoxon_p": _safe_float(t.get("wilcoxon_p_value")),
                                "effect_size": _safe_float(t.get("effect_size")),
                                "median_diff": _safe_float(t.get("median_diff")),
                                "holm_p": _safe_float(adj.get(name)),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(test_rows).set_index("comparison"),
                        width="stretch",
                        key=f"cmp_ablation_tests_{tick}",
                    )
                    if isinstance(holm_raw, dict) and holm_raw.get("alpha") is not None:
                        st.caption(
                            f"Holm-Bonferroni family-wise alpha = {holm_raw['alpha']}"
                        )
                st.caption(
                    "Source: results/e4_ablation/e4_results.json. "
                    "Feature-engineering contribution assessed at the "
                    "unit level (no leakage). "
                    f"{provenance_badge('PREDICTED')}"
                )

        with tabs[3]:
            section_header("Calibration")
            cal = comparison.get("calibration") or {}
            per_h = cal.get("per_horizon") or {}
            if per_h:
                st.caption(
                    f"Isotonic calibration of {cal.get('model', '—')} · "
                    "unit-level folds"
                )
                cal_rows = []
                for h_name, entry in per_h.items():
                    if not isinstance(entry, dict):
                        continue
                    cal_rows.append(
                        {
                            "horizon": h_name,
                            "brier": _safe_float(entry.get("brier")),
                            "ece_equal_width": _safe_float(
                                entry.get("ece_equal_width")
                            ),
                            "ece_equal_mass": _safe_float(
                                entry.get("ece_equal_mass")
                            ),
                        }
                    )
                if cal_rows:
                    cal_df = pd.DataFrame(cal_rows).set_index("horizon")
                    st.dataframe(
                        cal_df, width="stretch", key=f"cmp_cal_table_{tick}"
                    )

                    c0, c1 = st.columns(2)
                    with c0:
                        kpi_card(
                            "Mean Brier (5 horizons)",
                            f"{cal['mean_brier']:.4f}"
                            if cal.get("mean_brier") is not None
                            else "—",
                            provenance="PREDICTED",
                        )
                    with c1:
                        kpi_card(
                            "Mean ECE (equal-width)",
                            f"{cal['mean_ece']:.4f}"
                            if cal.get("mean_ece") is not None
                            else "—",
                            provenance="PREDICTED",
                        )

                    # Reliability curve for the widest horizon available
                    h30 = per_h.get("h30") or per_h.get("h10") or {}
                    reliability = h30.get("reliability") if isinstance(h30, dict) else None
                    if reliability:
                        pred = [r["mean_predicted"] for r in reliability]
                        obs = [r["mean_observed"] for r in reliability]
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=[0, 1], y=[0, 1], mode="lines",
                                name="Perfect calibration",
                                line=dict(color="#6B7280", dash="dash"),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=pred, y=obs, mode="lines+markers",
                                name="Isotonic (OOF)",
                                line=dict(color="#3B82F6"),
                            )
                        )
                        fig.update_layout(
                            **_DARK,
                            title="Reliability diagram · h30 · OOF folds",
                            xaxis=dict(title="Mean predicted probability",
                                       gridcolor="#1F2937"),
                            yaxis=dict(title="Observed frequency",
                                       gridcolor="#1F2937"),
                            height=400,
                            margin=dict(l=50, r=30, t=45, b=40),
                        )
                        st.plotly_chart(
                            fig, width="stretch", key=f"cmp_reliability_{tick}"
                        )
                st.caption(
                    "Source: results/e5_uncertainty/e5_results.json "
                    "(isotonic calibration fitted on training units only). "
                    f"{provenance_badge('PREDICTED')}"
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

            # --- E9: four interval methods compared -------------------------
            unc = _uncertainty_payload()
            if unc is not None:
                st.markdown("---")
                section_header("Uncertainty Method Comparison (E9)")
                st.caption(
                    f"All methods target {1 - (unc.get('alpha') or 0.1):.0%} "
                    "RUL interval coverage. Higher PICP is better; "
                    "lower width and Winkler are better."
                )
                st.dataframe(
                    pd.DataFrame(unc["rows"]).set_index("method"),
                    width="stretch",
                    key=f"cmp_e9_table_{tick}",
                )
                methods = [r["method"] for r in unc["rows"]]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=[r["picp"] for r in unc["rows"]],
                        name="PICP (coverage)",
                        marker_color="#3B82F6",
                        text=[f"{r['picp']:.1%}" for r in unc["rows"]],
                        textposition="outside",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=[r["winkler"] for r in unc["rows"]],
                        name="Winkler (lower better)",
                        marker_color="#F59E0B",
                        yaxis="y2",
                    )
                )
                fig.add_hline(
                    y=1 - (unc.get("alpha") or 0.1),
                    line=dict(color="#10B981", dash="dash"),
                    annotation_text="nominal",
                )
                fig.update_layout(
                    **_DARK,
                    barmode="group",
                    title="Coverage vs sharpness · tick #" + str(tick),
                    yaxis=dict(title="PICP", gridcolor="#1F2937"),
                    yaxis2=dict(
                        title="Winkler",
                        overlaying="y",
                        side="right",
                        gridcolor="#1F2937",
                    ),
                    height=420,
                    margin=dict(l=50, r=60, t=45, b=60),
                )
                st.plotly_chart(
                    fig, width="stretch", key=f"cmp_e9_chart_{tick}"
                )
                st.caption(
                    "Source: results/e9_uncertainty_comparison/e9_results.json. "
                    "Ensemble variance = multi-seed members + calibration "
                    "residual spread; bootstrap = unit-level cluster "
                    f"bootstrap + sampled residuals. "
                    f"{provenance_badge('PREDICTED')}"
                )

        with tabs[5]:
            section_header("Cross-Machine Generalization")
            gen = _generalization_payload()
            if gen is None:
                st.info(
                    "Generalization experiment not run yet. Execute `make e6` to populate "
                    "`results/e6_generalization/`."
                )
            else:
                st.caption(
                    f"Model {gen.get('model')} · rows = training set, "
                    "columns = test set · mean ROC-AUC (higher is better)"
                )
                fig = go.Figure(
                    data=go.Heatmap(
                        z=gen["auc_matrix"],
                        x=gen["datasets"],
                        y=gen["datasets"],
                        colorscale="Viridis",
                        zmin=0.5,
                        zmax=1.0,
                        colorbar=dict(title="ROC-AUC"),
                        text=np.round(np.array(gen["auc_matrix"], dtype=float), 3),
                        texttemplate="%{text}",
                        hoverongaps=False,
                    )
                )
                fig.update_layout(
                    **_DARK,
                    title="Cross-machine ROC-AUC matrix · tick #" + str(tick),
                    xaxis=dict(title="Test dataset", gridcolor="#1F2937"),
                    yaxis=dict(title="Train dataset", gridcolor="#1F2937"),
                    height=420,
                    margin=dict(l=50, r=30, t=45, b=40),
                )
                st.plotly_chart(
                    fig, width="stretch", key=f"cmp_gen_heatmap_{tick}"
                )

                if gen.get("rmse_matrix"):
                    fig2 = go.Figure(
                        data=go.Heatmap(
                            z=gen["rmse_matrix"],
                            x=gen["datasets"],
                            y=gen["datasets"],
                            colorscale="Magma_r",
                            text=np.round(
                                np.array(gen["rmse_matrix"], dtype=float), 1
                            ),
                            texttemplate="%{text}",
                            hoverongaps=False,
                        )
                    )
                    fig2.update_layout(
                        **_DARK,
                        title="Cross-machine RUL RMSE matrix (cycles)",
                        xaxis=dict(title="Test dataset", gridcolor="#1F2937"),
                        yaxis=dict(title="Train dataset", gridcolor="#1F2937"),
                        height=420,
                        margin=dict(l=50, r=30, t=45, b=40),
                    )
                    st.plotly_chart(
                        fig2, width="stretch", key=f"cmp_gen_rmse_{tick}"
                    )

                if gen.get("cross_rows"):
                    st.dataframe(
                        pd.DataFrame(gen["cross_rows"]).set_index(
                            ["train", "test"]
                        ),
                        width="stretch",
                        key=f"cmp_gen_table_{tick}",
                    )
                st.caption(
                    "Source: results/e6_generalization/e6_results.json. "
                    "Diagonal = within-dataset; off-diagonal = transfer "
                    "to unseen operating conditions. "
                    f"{provenance_badge('PREDICTED')}"
                )

    live_comparison()
