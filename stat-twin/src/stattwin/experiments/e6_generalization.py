"""Experiment e6 – Cross-Dataset Generalization.

Trains on FDx and tests on FDy for all dataset pairs.  Produces
a heatmap of performance (ROC-AUC, RUL RMSE) across all train/test
combinations.  Outputs to ``results/e6/``.

Usage::

    python -m stattwin.experiments.e6_generalization --ds FD001 --profile smoke
"""

from __future__ import annotations

import argparse
import copy
import json
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
from stattwin.evaluation.metrics import evaluate_classification, evaluate_rul
from stattwin.models import RandomForestModel, XGBoostModel

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
# Datasets to try
# ---------------------------------------------------------------------------

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def _load_dataset(ds_name: str) -> pd.DataFrame | None:
    """Load a C-MAPSS dataset; return None if file missing."""
    raw_path = resolve_raw_path(ds_name)
    if not raw_path.exists():
        return None
    return load_cmapss(raw_path, add_labels=True)


# ---------------------------------------------------------------------------
# Cross-dataset evaluation
# ---------------------------------------------------------------------------

def _cross_evaluate(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ds: str,
    test_ds: str,
) -> dict[str, Any]:
    """Train on one dataset, test on another."""
    feature_cols = feature_columns(test_df, include_health=True)
    # Ensure only columns present in both
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]

    y_train = train_df[label_cols].reindex(columns=label_cols, fill_value=0)
    y_test = test_df[label_cols].reindex(columns=label_cols, fill_value=0)

    m = copy.deepcopy(model)
    try:
        m.fit(train_df[feature_cols + ["unit_id", "cycle"]], y_train)
        proba = m.predict_proba(test_df[feature_cols + ["unit_id", "cycle"]])
        rul_pred = m.predict_rul(test_df[feature_cols + ["unit_id", "cycle"]])
    except Exception as e:
        return {"train": train_ds, "test": test_ds, "error": str(e)}

    # Classification metrics
    y_true_dict = {}
    y_prob_dict = {}
    for h in FAILURE_HORIZONS:
        col = label_col_for(h)
        if col in y_test.columns and col in proba.columns:
            y_true_dict[h] = y_test[col].values.astype(int)
            y_prob_dict[h] = proba[col].values.astype(float)

    cls_reports = evaluate_classification(y_true_dict, y_prob_dict)

    # RUL metrics
    rul_report = evaluate_rul(
        test_df["RUL"].values.astype(float),
        rul_pred.values.astype(float),
    )

    # Average ROC-AUC across horizons
    aucs = [r.roc_auc for r in cls_reports if not np.isnan(r.roc_auc)]
    mean_auc = float(np.mean(aucs)) if aucs else np.nan

    return {
        "train": train_ds,
        "test": test_ds,
        "mean_roc_auc": mean_auc,
        "rul_rmse": rul_report.rmse,
        "rul_mae": rul_report.mae,
        "nasa_score": rul_report.nasa_score,
        "per_horizon": [
            {"horizon": r.horizon, "roc_auc": r.roc_auc, "f1": r.f1}
            for r in cls_reports
        ],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdYlGn",
) -> None:
    """Generic heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="black" if val > (vmin or 0) + 0.5 * ((vmax or 1) - (vmin or 0)) else "white")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e6(cfg, out_dir) -> dict[str, Any]:
    """Run cross-dataset generalization for all dataset pairs."""
    # Load all available datasets
    datasets: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
        df = _load_dataset(ds)
        if df is not None:
            datasets[ds] = df
            print(f"  Loaded {ds}: {len(df)} rows, {df['unit_id'].nunique()} units")
        else:
            print(f"  Skipped {ds}: file not found")

    if len(datasets) < 2:
        return {"error": "Need at least 2 datasets for cross-dataset evaluation"}

    ds_names = sorted(datasets.keys())
    model = XGBoostModel(horizons=FAILURE_HORIZONS)

    all_results = []
    for train_ds in ds_names:
        for test_ds in ds_names:
            print(f"  Train={train_ds}, Test={test_ds}...")
            with Timer() as t:
                result = _cross_evaluate(
                    model, datasets[train_ds], datasets[test_ds], train_ds, test_ds,
                )
            result["elapsed_seconds"] = t.elapsed
            all_results.append(result)
            if "error" not in result:
                print(f"    ROC-AUC: {result['mean_roc_auc']:.4f}  RUL RMSE: {result['rul_rmse']:.2f}")
            else:
                print(f"    Error: {result['error']}")

    # Build heatmaps
    n = len(ds_names)
    auc_matrix = np.full((n, n), np.nan)
    rmse_matrix = np.full((n, n), np.nan)

    for r in all_results:
        if "error" in r:
            continue
        i = ds_names.index(r["train"])
        j = ds_names.index(r["test"])
        auc_matrix[i, j] = r["mean_roc_auc"]
        rmse_matrix[i, j] = r["rul_rmse"]

    results: dict[str, Any] = {
        "datasets": ds_names,
        "model": model.name,
        "cross_results": all_results,
        "auc_matrix": auc_matrix.tolist(),
        "rmse_matrix": rmse_matrix.tolist(),
    }

    save_json(results, out_dir / "e6_results.json")

    # Heatmaps
    _plot_heatmap(auc_matrix, ds_names, "Cross-Dataset ROC-AUC", "Test", "Train",
                  out_dir / "figures" / "heatmap_auc.png", vmin=0.4, vmax=1.0)
    _plot_heatmap(rmse_matrix, ds_names, "Cross-Dataset RUL RMSE", "Test", "Train",
                  out_dir / "figures" / "heatmap_rmse.png", vmin=0, vmax=None, cmap="RdYlGn_r")

    return results


def main() -> None:
    """CLI entry-point for e6_generalization."""
    parser = argparse.ArgumentParser(
        description="E6: Cross-Dataset Generalization – train FDx test FDy heatmaps."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    out_dir = setup_output("e6_generalization")

    with Timer() as t:
        results = run_e6(cfg, out_dir)

    print(f"\n[e6] Completed in {t.elapsed:.1f}s")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e6_generalization",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
