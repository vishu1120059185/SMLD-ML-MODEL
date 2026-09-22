"""Experiment e7 – Operating-Condition Normalization.

Tests whether per-condition normalization matters by comparing model
performance on FD002/FD004 (multi-condition datasets) with and without
per-condition normalization.  Outputs to ``results/e7/``.

Usage::

    python -m stattwin.experiments.e7_operating_conditions --ds FD002 --profile smoke
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
from stattwin.data.schema import FAILURE_HORIZONS, OP_SETTINGS, label_col_for
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.evaluation.metrics import evaluate_classification, evaluate_rul
from stattwin.models import RandomForestModel, XGBoostModel
from stattwin.preprocessing.scaler import TrainFittedScaler

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
# Regime normalization
# ---------------------------------------------------------------------------

def _assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign operational regime clusters via rounding of op settings."""
    out = df.copy()
    settings = [s for s in OP_SETTINGS if s in df.columns]
    if not settings:
        out["regime"] = 0
        return out

    # Round to 2 decimals and create composite key
    rounded = out[settings].round(2)
    keys = rounded.apply(lambda row: tuple(row), axis=1)
    out["regime"] = pd.Categorical(keys).codes
    return out


def _per_condition_normalize(
    df: pd.DataFrame,
    sensor_cols: list[str],
    fit_units: np.ndarray | None = None,
) -> pd.DataFrame:
    """Normalize sensors within each regime to zero mean, unit variance."""
    out = df.copy()
    if "regime" not in out.columns:
        out = _assign_regimes(out)

    # Fit on training units only
    if fit_units is not None:
        fit_df = out[out["unit_id"].isin(fit_units)]
    else:
        fit_df = out

    # Compute per-regime statistics
    regime_stats: dict[int, dict[str, tuple[float, float]]] = {}
    for regime, grp in fit_df.groupby("regime"):
        stats = {}
        for col in sensor_cols:
            mean = grp[col].mean()
            std = grp[col].std()
            if std == 0 or np.isnan(std):
                std = 1.0
            stats[col] = (float(mean), float(std))
        regime_stats[regime] = stats

    # Apply normalization
    for regime, stats in regime_stats.items():
        mask = out["regime"] == regime
        for col, (mean, std) in stats.items():
            out.loc[mask, col] = (out.loc[mask, col] - mean) / std

    return out


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _compare_normalization(
    df: pd.DataFrame,
    cfg,
    sensor_cols: list[str],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Compare model performance with and without per-condition normalization."""
    splits = make_group_kfold_splits(df, n_splits=cfg.split.n_splits, seed=cfg.seed)
    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    y = df[label_cols].copy()

    model = XGBoostModel(horizons=FAILURE_HORIZONS)

    results_no_norm = _oof_eval(model, df, y, splits, feature_cols)
    df_normed = _per_condition_normalize(df, sensor_cols)
    norm_feature_cols = feature_columns(df_normed, include_health=True)
    norm_feature_cols = [c for c in norm_feature_cols if c in df_normed.columns]
    results_with_norm = _oof_eval(model, df_normed, y, splits, norm_feature_cols)

    return {
        "without_normalization": results_no_norm,
        "with_normalization": results_with_norm,
    }


def _oof_eval(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    splits: list[dict],
    feature_cols: list[str],
) -> dict[str, Any]:
    """OOF evaluation returning aggregated metrics."""
    all_auc: list[float] = []
    all_rul_rmse: list[float] = []

    for split in splits:
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()
        y_tr = y.loc[X_tr.index]
        y_val = y.loc[X_val.index]

        available_features = [c for c in feature_cols if c in X_tr.columns and c in X_val.columns]

        m = copy.deepcopy(model)
        try:
            m.fit(X_tr[available_features + ["unit_id", "cycle"]], y_tr)
            proba = m.predict_proba(X_val[available_features + ["unit_id", "cycle"]])
            rul_pred = m.predict_rul(X_val[available_features + ["unit_id", "cycle"]])
        except Exception:
            continue

        # ROC-AUC for h30
        col = label_col_for(30)
        if col in y_val.columns and col in proba.columns:
            y_true = y_val[col].values.astype(int)
            y_prob = proba[col].values.astype(float)
            if len(np.unique(y_true)) >= 2:
                from sklearn.metrics import roc_auc_score
                all_auc.append(roc_auc_score(y_true, y_prob))

        # RUL RMSE
        if "RUL" in X_val.columns:
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(X_val["RUL"].values, rul_pred.values))
            all_rul_rmse.append(rmse)

    return {
        "mean_roc_auc_h30": float(np.mean(all_auc)) if all_auc else np.nan,
        "std_roc_auc_h30": float(np.std(all_auc)) if all_auc else np.nan,
        "mean_rul_rmse": float(np.mean(all_rul_rmse)) if all_rul_rmse else np.nan,
        "std_rul_rmse": float(np.std(all_rul_rmse)) if all_rul_rmse else np.nan,
        "n_folds": len(all_auc),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_normalization_comparison(results: dict, out_dir: Path) -> None:
    """Side-by-side bar chart of metrics with/without normalization."""
    no_norm = results["without_normalization"]
    with_norm = results["with_normalization"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC-AUC
    labels = ["Without Norm", "With Norm"]
    aucs = [no_norm["mean_roc_auc_h30"], with_norm["mean_roc_auc_h30"]]
    errs = [no_norm["std_roc_auc_h30"], with_norm["std_roc_auc_h30"]]
    ax1.bar(labels, aucs, yerr=errs, capsize=10, color=["coral", "steelblue"])
    ax1.set_ylabel("ROC-AUC (h30)")
    ax1.set_title("Per-Condition Normalization – ROC-AUC")

    # RUL RMSE
    rmses = [no_norm["mean_rul_rmse"], with_norm["mean_rul_rmse"]]
    rerrs = [no_norm["std_rul_rmse"], with_norm["std_rul_rmse"]]
    ax2.bar(labels, rmses, yerr=rerrs, capsize=10, color=["coral", "steelblue"])
    ax2.set_ylabel("RUL RMSE")
    ax2.set_title("Per-Condition Normalization – RUL RMSE")

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "normalization_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e7(cfg, df, out_dir) -> dict[str, Any]:
    """Run operating-condition normalization experiment."""
    sensor_cols = [c for c in SENSOR_COLS if c in df.columns]
    feature_cols = feature_columns(df, include_health=True)

    print("  Comparing with/without per-condition normalization...")
    with Timer() as t:
        results = _compare_normalization(df, cfg, sensor_cols, feature_cols)
    results["elapsed_seconds"] = t.elapsed

    # Summary
    no = results["without_normalization"]
    wi = results["with_normalization"]
    print(f"    Without norm: ROC-AUC={no['mean_roc_auc_h30']:.4f}, RMSE={no['mean_rul_rmse']:.2f}")
    print(f"    With norm:    ROC-AUC={wi['mean_roc_auc_h30']:.4f}, RMSE={wi['mean_rul_rmse']:.2f}")

    out = {
        "dataset": cfg.dataset.name,
        "normalization_comparison": results,
    }

    save_json(out, out_dir / "e7_results.json")
    _plot_normalization_comparison(results, out_dir)

    return out


def main() -> None:
    """CLI entry-point for e7_operating_conditions."""
    parser = argparse.ArgumentParser(
        description="E7: Operating-Condition Normalization – FD002/FD004 with/without."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e7_operating_conditions")

    with Timer() as t:
        results = run_e7(cfg, df, out_dir)

    print(f"\n[e7] Completed in {t.elapsed:.1f}s")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e7_operating_conditions",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
