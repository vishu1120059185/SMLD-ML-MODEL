"""Experiment e5 – Uncertainty and Calibration.

Evaluates uncertainty quantification and probability calibration:
reliability diagrams, Brier score, ECE (equal-width and equal-mass),
and conformal prediction interval coverage.  Outputs to ``results/e5/``.

Usage::

    python -m stattwin.experiments.e5_uncertainty --ds FD001 --profile smoke
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
from stattwin.models import XGBoostModel
from stattwin.uncertainty.calibration import (
    brier_score,
    ece_equal_mass,
    ece_equal_width,
    reliability_diagram_data,
)
from stattwin.uncertainty.conformal import conformal_intervals

from ._common import (
    Timer,
    add_common_args,
    feature_columns,
    resolve_raw_path,
    save_json,
    setup_output,
)

# ---------------------------------------------------------------------------
# OOF evaluation for uncertainty
# ---------------------------------------------------------------------------

def _oof_uncertainty(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    splits: list[dict],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Run OOF and collect per-horizon probabilities for calibration analysis."""
    all_proba: list[pd.DataFrame] = []
    all_y_true: list[pd.DataFrame] = []

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
        except Exception as e:
            print(f"    Fold {fold_idx} failed: {e}")
            continue

        all_proba.append(proba)
        all_y_true.append(y_val)

    if not all_proba:
        return {"error": "all folds failed"}

    oof_proba = pd.concat(all_proba, ignore_index=True)
    oof_y = pd.concat(all_y_true, ignore_index=True)

    # Per-horizon calibration analysis
    calibration_results = {}
    for h in FAILURE_HORIZONS:
        col = label_col_for(h)
        if col not in oof_y.columns or col not in oof_proba.columns:
            continue

        y_true = oof_y[col].values.astype(int)
        y_prob = oof_proba[col].values.astype(float)

        if len(np.unique(y_true)) < 2:
            continue

        # Brier
        brier = brier_score(y_true, y_prob)

        # ECE
        ece_ew = ece_equal_width(y_true, y_prob)
        ece_em = ece_equal_mass(y_true, y_prob)

        # Reliability diagram data
        rel = reliability_diagram_data(y_true, y_prob, n_bins=10)

        calibration_results[f"h{h}"] = {
            "brier": brier,
            "ece_equal_width": ece_ew,
            "ece_equal_mass": ece_em,
            "reliability": rel.to_dict(orient="records"),
        }

    # Aggregate
    brier_vals = [v["brier"] for v in calibration_results.values()]
    ece_ew_vals = [v["ece_equal_width"] for v in calibration_results.values()]

    return {
        "per_horizon": calibration_results,
        "mean_brier": float(np.mean(brier_vals)) if brier_vals else np.nan,
        "mean_ece_ew": float(np.mean(ece_ew_vals)) if ece_ew_vals else np.nan,
        "n_folds_used": len(all_proba),
    }


def _conformal_analysis(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    splits: list[dict],
    feature_cols: list[str],
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Split-conformal analysis with calibration/test split."""
    # Use first fold for calibration, rest for test
    if not splits:
        return {"error": "no splits"}

    cal_split = splits[0]
    test_units = np.concatenate([s["val_units"] for s in splits[1:]])

    X_cal = X[X["unit_id"].isin(cal_split["val_units"])].copy()
    X_test = X[X["unit_id"].isin(test_units)].copy()

    m_cal = copy.deepcopy(model)
    copy.deepcopy(model)

    try:
        # Fit on training portion of first split
        X_tr = X[X["unit_id"].isin(cal_split["train_units"])].copy()
        y_tr = y.loc[X_tr.index]
        m_cal.fit(X_tr[feature_cols + ["unit_id", "cycle"]], y_tr)

        # Calibration predictions
        rul_cal = m_cal.predict_rul(X_cal[feature_cols + ["unit_id", "cycle"]])
        rul_test = m_cal.predict_rul(X_test[feature_cols + ["unit_id", "cycle"]])

        # Ensemble std (use raw score variance as proxy)
        sigma_cal = np.abs(rul_cal.values - rul_cal.values.mean()) + 1.0
        sigma_test = np.abs(rul_test.values - rul_test.values.mean()) + 1.0

        rul_cal_true = X_cal["RUL"].values if "RUL" in X_cal.columns else rul_cal.values
        rul_test_true = X_test["RUL"].values if "RUL" in X_test.columns else rul_test.values

        report = conformal_intervals(
            y_cal=rul_cal_true.astype(float),
            yhat_cal=rul_cal.values.astype(float),
            sigma_cal=sigma_cal.astype(float),
            yhat_test=rul_test.values.astype(float),
            sigma_test=sigma_test.astype(float),
            alpha=alpha,
            rul_test=rul_test_true.astype(float) if rul_test_true is not None else None,
        )

        return {
            "alpha": alpha,
            "quantile_q": report.quantile_q,
            "coverage": report.coverage,
            "mean_width": report.mean_width,
            "winkler": report.winkler,
            "coverage_by_rul_bucket": report.coverage_by_rul_bucket,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_reliability_diagrams(cal_results: dict, out_dir: Path) -> None:
    """Plot reliability diagrams for each horizon."""
    horizon_results = cal_results.get("per_horizon", {})
    n_horizons = len(horizon_results)
    if n_horizons == 0:
        return

    fig, axes = plt.subplots(1, min(n_horizons, 5), figsize=(4 * min(n_horizons, 5), 4))
    if n_horizons == 1:
        axes = [axes]

    for i, (h_key, h_data) in enumerate(sorted(horizon_results.items())):
        if i >= 5:
            break
        rel = h_data.get("reliability", [])
        if not rel:
            continue

        df_rel = pd.DataFrame(rel).dropna(subset=["mean_predicted", "mean_observed"])
        ax = axes[i]
        ax.plot(df_rel["mean_predicted"], df_rel["mean_observed"], "o-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean predicted")
        ax.set_ylabel("Mean observed")
        ax.set_title(f"{h_key}\nBrier={h_data['brier']:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Reliability Diagrams", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_brier_comparison(cal_results: dict, out_dir: Path) -> None:
    """Bar chart of Brier scores per horizon."""
    horizon_results = cal_results.get("per_horizon", {})
    if not horizon_results:
        return

    horizons = sorted(horizon_results.keys())
    briers = [horizon_results[h]["brier"] for h in horizons]
    eces = [horizon_results[h]["ece_equal_width"] for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(horizons, briers, color="steelblue")
    ax1.set_ylabel("Brier Score")
    ax1.set_title("Brier Score per Horizon (lower is better)")

    ax2.bar(horizons, eces, color="coral")
    ax2.set_ylabel("ECE (equal-width)")
    ax2.set_title("ECE per Horizon (lower is better)")

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "brier_ece.png", dpi=150)
    plt.close(fig)


def _plot_conformal_coverage(conformal: dict, out_dir: Path) -> None:
    """Bar chart of coverage by RUL bucket."""
    buckets = conformal.get("coverage_by_rul_bucket", {})
    if not buckets:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(buckets.keys())
    values = list(buckets.values())
    ax.bar(names, values, color="steelblue")
    ax.axhline(1 - conformal.get("alpha", 0.1), color="red", linestyle="--",
               label=f"Nominal: {1 - conformal.get('alpha', 0.1):.0%}")
    ax.set_ylabel("Coverage")
    ax.set_title(f"Conformal Coverage by RUL Bucket (PICP={conformal.get('coverage', 'N/A'):.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "conformal_coverage.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e5(cfg, df, out_dir) -> dict[str, Any]:
    """Run uncertainty and calibration evaluation."""
    feature_cols = feature_columns(df, include_health=True)
    splits = make_group_kfold_splits(df, n_splits=cfg.split.n_splits, seed=cfg.seed)
    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    y = df[label_cols].copy()

    model = XGBoostModel(horizons=FAILURE_HORIZONS)

    # Calibration analysis
    print("  Running OOF calibration analysis...")
    with Timer() as t:
        cal_results = _oof_uncertainty(model, df, y, splits, feature_cols)
    cal_results["elapsed_seconds"] = t.elapsed
    print(f"    Mean Brier: {cal_results.get('mean_brier', 'N/A')}")

    # Conformal analysis
    print("  Running conformal interval analysis...")
    with Timer() as t:
        conformal = _conformal_analysis(model, df, y, splits, feature_cols,
                                        alpha=cfg.uncertainty.alpha)
    conformal["elapsed_seconds"] = t.elapsed
    print(f"    Coverage: {conformal.get('coverage', 'N/A')}")

    results: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "model": model.name,
        "calibration": cal_results,
        "conformal": conformal,
    }

    save_json(results, out_dir / "e5_results.json")
    _plot_reliability_diagrams(cal_results, out_dir)
    _plot_brier_comparison(cal_results, out_dir)
    _plot_conformal_coverage(conformal, out_dir)

    return results


def main() -> None:
    """CLI entry-point for e5_uncertainty."""
    parser = argparse.ArgumentParser(
        description="E5: Uncertainty & Calibration – Brier, ECE, reliability, conformal."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e5_uncertainty")

    with Timer() as t:
        results = run_e5(cfg, df, out_dir)

    print(f"\n[e5] Completed in {t.elapsed:.1f}s")
    print(f"     Mean Brier: {results['calibration'].get('mean_brier', 'N/A')}")
    print(f"     Conformal coverage: {results['conformal'].get('coverage', 'N/A')}")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e5_uncertainty",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
