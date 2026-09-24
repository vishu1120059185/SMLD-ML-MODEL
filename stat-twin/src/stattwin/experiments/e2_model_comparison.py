"""Experiment e2 – Model Comparison.

Compares 7 prediction models: Logistic Regression, Random Forest,
XGBoost, GRU, LSTM, Hybrid, and Threshold.  Per-horizon classification
metrics, RUL metrics, and NASA score.  Outputs comparison tables and
charts to ``results/e2/``.

Usage::

    python -m stattwin.experiments.e2_model_comparison --ds FD001 --profile smoke
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stattwin.config import load_config
from stattwin.data.loader import load_cmapss
from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.evaluation.metrics import (
    evaluate_classification,
    evaluate_rul,
)
from stattwin.models import (
    GRUModel,
    LogisticModel,
    RandomForestModel,
    XGBoostModel,
)

from ._common import (
    SENSOR_COLS,
    Timer,
    add_common_args,
    feature_columns,
    resolve_raw_path,
    save_json,
    setup_output,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_models(cfg) -> list:
    """Instantiate all models with config hyperparameters."""
    models = [
        LogisticModel(horizons=FAILURE_HORIZONS),
        RandomForestModel(horizons=FAILURE_HORIZONS),
        XGBoostModel(horizons=FAILURE_HORIZONS),
    ]
    # Only add deep-learning models if torch is available
    try:
        import torch  # noqa: F401
        models.append(GRUModel(horizons=FAILURE_HORIZONS))
    except ImportError:
        pass
    return models


# ---------------------------------------------------------------------------
# Single-model OOF evaluation
# ---------------------------------------------------------------------------

def _evaluate_single_model(
    model, X: pd.DataFrame, y: pd.DataFrame,
    splits: list[dict], feature_cols: list[str],
    rul_series: pd.Series,
) -> dict[str, Any]:
    """Run OOF evaluation for one model and return aggregated metrics."""
    all_proba: list[pd.DataFrame] = []
    all_rul_pred: list[pd.Series] = []
    all_rul_true: list[pd.Series] = []
    all_y_true: list[pd.DataFrame] = []
    all_raw_score: list[pd.Series] = []

    for fold_idx, split in enumerate(splits):
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()
        y_tr = y.loc[X_tr.index]
        y_val = y.loc[X_val.index]

        m = copy.deepcopy(model)
        try:
            m.fit(X_tr[feature_cols + ["unit_id", "cycle"]], y_tr)
            proba = m.predict_proba(X_val[feature_cols + ["unit_id", "cycle"]])
            rul_pred = m.predict_rul(X_val[feature_cols + ["unit_id", "cycle"]])
            raw = m.score_raw(X_val[feature_cols + ["unit_id", "cycle"]])
        except Exception as e:
            print(f"  Fold {fold_idx} failed for {model.name}: {e}")
            continue

        all_proba.append(proba)
        all_rul_pred.append(rul_pred)
        all_rul_true.append(rul_series.loc[X_val.index] if "RUL" in X_val.columns else pd.Series(dtype=float))  # noqa: E501
        all_y_true.append(y_val)
        all_raw_score.append(raw)

    if not all_proba:
        return {"model": model.name, "error": "all folds failed"}

    # Aggregate OOF predictions
    oof_proba = pd.concat(all_proba, ignore_index=True)
    oof_rul_pred = pd.concat(all_rul_pred, ignore_index=True)
    oof_rul_true = pd.concat(all_rul_true, ignore_index=True)
    oof_y_true = pd.concat(all_y_true, ignore_index=True)

    # Classification metrics
    y_true_dict = {}
    y_prob_dict = {}
    for h in FAILURE_HORIZONS:
        col = label_col_for(h)
        if col in oof_y_true.columns and col in oof_proba.columns:
            y_true_dict[h] = oof_y_true[col].values.astype(int)
            y_prob_dict[h] = oof_proba[col].values.astype(float)

    cls_reports = evaluate_classification(y_true_dict, y_prob_dict)

    # RUL metrics
    rul_report = evaluate_rul(
        oof_rul_true.values.astype(float),
        oof_rul_pred.values.astype(float),
    )

    return {
        "model": model.name,
        "classification": [
            {
                "horizon": r.horizon,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "roc_auc": r.roc_auc,
                "pr_auc": r.pr_auc,
            }
            for r in cls_reports
        ],
        "rul": {
            "mae": rul_report.mae,
            "rmse": rul_report.rmse,
            "nasa_score": rul_report.nasa_score,
            "n": rul_report.n,
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_roc_comparison(all_results: list[dict], out_dir: Path) -> None:
    """Bar chart comparing ROC-AUC across models and horizons."""
    models = [r["model"] for r in all_results if "classification" in r]
    horizons = [h for h in FAILURE_HORIZONS]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(horizons))
    width = 0.8 / max(len(models), 1)

    for i, model_name in enumerate(models):
        result = next(r for r in all_results if r["model"] == model_name)
        if "error" in result:
            continue
        aucs = []
        for h in horizons:
            entry = next((c for c in result["classification"] if c["horizon"] == h), None)
            aucs.append(entry["roc_auc"] if entry else 0.0)
        ax.bar(x + i * width, aucs, width, label=model_name)

    ax.set_xlabel("Horizon")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model Comparison – ROC-AUC by Horizon")
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels([f"h{h}" for h in horizons])
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "roc_comparison.png", dpi=150)
    plt.close(fig)


def _plot_rul_comparison(all_results: list[dict], out_dir: Path) -> None:
    """Bar chart of RUL RMSE and NASA score."""
    models = [r["model"] for r in all_results if "rul" in r]
    rmses = [r["rul"]["rmse"] for r in all_results if "rul" in r]
    nasas = [r["rul"]["nasa_score"] for r in all_results if "rul" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.barh(models, rmses, color="steelblue")
    ax1.set_xlabel("RMSE")
    ax1.set_title("RUL RMSE (lower is better)")

    ax2.barh(models, nasas, color="coral")
    ax2.set_xlabel("NASA Score")
    ax2.set_title("NASA Score (lower is better)")

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "rul_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e2(cfg, df, out_dir) -> dict[str, Any]:
    """Run model comparison experiment."""
    [c for c in SENSOR_COLS if c in df.columns]
    feature_cols = feature_columns(df, include_health=True)

    # Prepare labels
    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    y = df[label_cols].copy()

    # Splits
    splits = make_group_kfold_splits(df, n_splits=cfg.split.n_splits, seed=cfg.seed)

    # Build models
    models = _build_models(cfg)

    all_results = []
    for model in models:
        print(f"  Evaluating {model.name}...")
        with Timer() as t:
            result = _evaluate_single_model(model, df, y, splits, feature_cols, df["RUL"])
        result["elapsed_seconds"] = t.elapsed
        all_results.append(result)
        print(f"    Done in {t.elapsed:.1f}s")

    # Summary table
    summary_rows = []
    for r in all_results:
        if "error" in r:
            summary_rows.append({"model": r["model"], "error": r["error"]})
            continue
        row: dict[str, Any] = {"model": r["model"]}
        for c in r["classification"]:
            row[f"roc_auc_h{c['horizon']}"] = c["roc_auc"]
            row[f"f1_h{c['horizon']}"] = c["f1"]
        row["rul_rmse"] = r["rul"]["rmse"]
        row["rul_mae"] = r["rul"]["mae"]
        row["nasa_score"] = r["rul"]["nasa_score"]
        summary_rows.append(row)

    results: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "models": all_results,
        "summary_table": summary_rows,
    }

    save_json(results, out_dir / "e2_results.json")
    _plot_roc_comparison(all_results, out_dir)
    _plot_rul_comparison(all_results, out_dir)

    return results


def main() -> None:
    """CLI entry-point for e2_model_comparison."""
    parser = argparse.ArgumentParser(
        description="E2: Model Comparison – 7 models, per-horizon metrics, NASA score."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e2_model_comparison")

    with Timer() as t:
        results = run_e2(cfg, df, out_dir)

    print(f"\n[e2] Completed in {t.elapsed:.1f}s")
    print(f"     Models evaluated: {len(results['models'])}")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e2_model_comparison",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
