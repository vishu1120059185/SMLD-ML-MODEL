"""Experiment e9 – Uncertainty Method Comparison.

Compares four uncertainty quantification methods for RUL prediction
(all calibrated against a unit-disjoint inner calibration split):

  1. Split conformal prediction (normalised nonconformity scores)
  2. Ensemble variance – multi-seed XGBoost members; interval =
     ensemble mean ± z · sqrt(between-member variance + calibration
     residual variance)
  3. Quantile bootstrap – residual quantiles estimated on the
     calibration split and added to point predictions
  4. Bootstrap prediction intervals – pairs bootstrap of the training
     rows **plus** sampled calibration residuals (parameter + aleatoric
     uncertainty)

Outputs comparison tables and charts to ``results/e9_uncertainty_comparison/``.

Usage::

    python -m stattwin.experiments.e9_uncertainty_comparison --ds FD001 --profile fast
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
from stattwin.data.splitter import inner_unit_split, make_group_kfold_splits
from stattwin.evaluation.metrics import interval_metrics
from stattwin.models import XGBoostModel
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
# Uncertainty methods
# ---------------------------------------------------------------------------

def _split_conformal(
    y_cal: np.ndarray, yhat_cal: np.ndarray, sigma_cal: np.ndarray,
    yhat_test: np.ndarray, sigma_test: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Split conformal prediction intervals."""
    report = conformal_intervals(y_cal, yhat_cal, sigma_cal, yhat_test, sigma_test, alpha)
    return {
        "method": "split_conformal",
        "lower": report.intervals["lower"].values if report.intervals is not None else None,
        "upper": report.intervals["upper"].values if report.intervals is not None else None,
        "coverage": report.coverage,
        "mean_width": report.mean_width,
        "quantile_q": report.quantile_q,
    }


def _ensemble_variance(
    ensemble_preds: list[np.ndarray],
    alpha: float = 0.10,
    resid_std: float | None = None,
) -> dict[str, Any]:
    """Ensemble variance: mean ± z * sqrt(var_between + var_residual).

    Parameters
    ----------
    ensemble_preds:
        Per-member point predictions on the same rows.
    alpha:
        Two-sided error rate (0.10 -> 90 % interval).
    resid_std:
        Aleatoric residual standard deviation estimated on a held-out
        calibration split.  Combined in quadrature with the
        between-member (epistemic) spread; *None* uses member variance
        alone.
    """
    from scipy.stats import norm

    stack = np.stack(ensemble_preds, axis=0)
    mean = np.mean(stack, axis=0)
    std_between = np.std(stack, axis=0, ddof=1) if len(ensemble_preds) > 1 else np.zeros_like(mean)  # noqa: E501
    if resid_std is not None:
        total_std = np.sqrt(std_between**2 + float(resid_std) ** 2)
    else:
        total_std = std_between
    z = norm.ppf(1 - alpha / 2)
    lower = mean - z * total_std
    upper = mean + z * total_std
    return {
        "method": "ensemble_variance",
        "mean": mean,
        "std": total_std,
        "std_between": std_between,
        "resid_std": resid_std,
        "lower": lower,
        "upper": upper,
    }


def _quantile_regression(
    y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
    alpha: float = 0.10,
    n_bootstrap: int = 30,
) -> dict[str, Any]:
    """Residual-quantile intervals via bootstrap averaging.

    Residual quantiles are estimated on a **calibration** split and
    returned as offsets to be added to point predictions elsewhere;
    the caller must never use test labels here.

    Returns
    -------
    dict with ``lower_offset`` / ``upper_offset`` (scalars).
    """
    rng = np.random.default_rng(42)
    residuals = y_true_cal - y_pred_cal
    n = len(residuals)
    if n == 0:
        return {"method": "quantile_bootstrap", "lower_offset": 0.0, "upper_offset": 0.0}  # noqa: E501

    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_resid = residuals[idx]
        lower_bounds.append(float(np.percentile(boot_resid, 100 * lower_q)))
        upper_bounds.append(float(np.percentile(boot_resid, 100 * upper_q)))

    return {
        "method": "quantile_bootstrap",
        "lower_offset": float(np.mean(lower_bounds)),
        "upper_offset": float(np.mean(upper_bounds)),
    }


def _bootstrap_pi(
    model_template, X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    n_bootstraps: int = 30,
    alpha: float = 0.10,
    seed: int = 42,
    residuals: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap prediction intervals via resampled model fits.

    Each replicate refits the model on a **unit-level cluster
    bootstrap** of the training data (rows within a unit stay together,
    so temporal correlation is preserved and every replicate keeps both
    label classes) and predicts on ``X_test``.  When ``residuals``
    (calibration residuals ``y_true - y_pred``) are supplied, a residual
    draw is added to every replicate so the interval captures
    **aleatoric** noise as well as parameter uncertainty – without it
    the percentile band collapses onto the (nearly deterministic)
    ensemble spread.

    ``model_template`` may be a fitted model instance (deep-copied) or a
    model class (instantiated with default horizons).
    """
    from stattwin.data.schema import FAILURE_HORIZONS as _HZ

    rng = np.random.default_rng(seed)
    test_preds = []
    n = len(X_train)
    n_test = len(X_test)
    fit_cols = feature_cols + ["unit_id", "cycle"]
    if "RUL" in X_train.columns:
        fit_cols = fit_cols + ["RUL"]

    # Group training rows by unit once (cluster bootstrap)
    if "unit_id" in X_train.columns:
        unit_codes, unit_list = pd.factorize(X_train["unit_id"], sort=True)
        row_of_unit = [np.flatnonzero(unit_codes == i) for i in range(len(unit_list))]
        n_clusters = len(unit_list)
    else:
        row_of_unit = None
        n_clusters = n

    for _b in range(n_bootstraps):
        if row_of_unit is not None:
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            boot_idx = np.concatenate([row_of_unit[c] for c in chosen])
        else:
            boot_idx = rng.integers(0, n, size=n)
        X_boot = X_train.iloc[boot_idx].reset_index(drop=True)
        y_boot = y_train.iloc[boot_idx].reset_index(drop=True)

        if isinstance(model_template, type):
            m = model_template(horizons=list(_HZ))
        else:
            m = copy.deepcopy(model_template)
        try:
            m.fit(X_boot[fit_cols], y_boot)
            pred = np.asarray(
                m.predict_rul(X_test[feature_cols + ["unit_id", "cycle"]]),
                dtype=float,
            )
        except Exception:
            continue

        if residuals is not None and len(residuals) > 0:
            noise = rng.choice(np.asarray(residuals, dtype=float), size=n_test)
            pred = pred + noise
        test_preds.append(pred)

    if not test_preds:
        return {"method": "bootstrap", "error": "all bootstraps failed"}

    stack = np.stack(test_preds, axis=0)
    mean = np.mean(stack, axis=0)
    lower = np.percentile(stack, 100 * alpha / 2, axis=0)
    upper = np.percentile(stack, 100 * (1 - alpha / 2), axis=0)

    return {
        "method": "bootstrap",
        "lower": lower,
        "upper": upper,
        "mean": mean,
        "std": np.std(stack, axis=0),
        "n_successful_bootstraps": len(test_preds),
        "residual_augmented": residuals is not None,
        "bootstrap_unit": "cluster" if row_of_unit is not None else "row",
    }


# ---------------------------------------------------------------------------
# OOF evaluation
# ---------------------------------------------------------------------------

def _compare_uncertainty_methods(
    cfg, X: pd.DataFrame, feature_cols: list[str],
) -> dict[str, Any]:
    """Compare all four uncertainty methods via OOF evaluation."""
    splits = make_group_kfold_splits(X, n_splits=cfg.split.n_splits, seed=cfg.seed)
    alpha = cfg.uncertainty.alpha
    n_members = cfg.uncertainty.n_ensemble_members
    n_bootstraps = cfg.uncertainty.n_bootstraps

    model = XGBoostModel(
        n_estimators=cfg.model.xgb.n_estimators,
        max_depth=cfg.model.xgb.max_depth,
        learning_rate=cfg.model.xgb.learning_rate,
        horizons=FAILURE_HORIZONS,
    )

    method_results: dict[str, list[dict]] = {
        "split_conformal": [],
        "ensemble_variance": [],
        "quantile_bootstrap": [],
        "bootstrap": [],
    }

    for fold_idx, split in enumerate(splits):
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()

        # Inner split for conformal calibration
        inner_train, inner_val = inner_unit_split(train_units, val_frac=0.3, seed=42 + fold_idx)
        X_cal = X_tr[X_tr["unit_id"].isin(inner_val)].copy()
        X_fit = X_tr[X_tr["unit_id"].isin(inner_train)].copy()

        available = [c for c in feature_cols if c in X_tr.columns and c in X_val.columns]
        label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
        y_fit = X_fit[label_cols].reindex(columns=label_cols, fill_value=0)

        # Fit model (include RUL so tree models train a real RUL regressor)
        fit_cols = available + ["unit_id", "cycle"]
        if "RUL" in X_fit.columns:
            fit_cols = fit_cols + ["RUL"]
        m = copy.deepcopy(model)
        try:
            m.fit(X_fit[fit_cols], y_fit)
        except Exception:
            continue

        # Predictions
        rul_cal = m.predict_rul(X_cal[available + ["unit_id", "cycle"]]).values
        rul_val = m.predict_rul(X_val[available + ["unit_id", "cycle"]]).values

        rul_cal_true = X_cal["RUL"].values.astype(float) if "RUL" in X_cal.columns else rul_cal.astype(float)  # noqa: E501
        rul_val_true = X_val["RUL"].values.astype(float) if "RUL" in X_val.columns else rul_val.astype(float)  # noqa: E501
        cal_resid = rul_cal_true - rul_cal

        # 1. Split conformal
        sigma_cal = np.abs(rul_cal - rul_cal.mean()) + 1.0
        sigma_val = np.abs(rul_val - rul_val.mean()) + 1.0
        conf = _split_conformal(rul_cal_true, rul_cal.astype(float), sigma_cal.astype(float),
                                rul_val.astype(float), sigma_val.astype(float), alpha)
        im = interval_metrics(rul_val_true, np.array(conf["lower"]), np.array(conf["upper"]), alpha)
        method_results["split_conformal"].append({
            "picp": im.picp, "mean_width": im.mean_width, "winkler": im.winkler,
        })

        # 2. Ensemble variance: multi-seed XGB members + calibration residuals
        member_preds = []
        for b in range(n_members):
            try:
                mb = XGBoostModel(
                    n_estimators=cfg.model.xgb.n_estimators,
                    max_depth=cfg.model.xgb.max_depth,
                    learning_rate=cfg.model.xgb.learning_rate,
                    random_state=cfg.seed + b,
                    horizons=FAILURE_HORIZONS,
                )
                mb.fit(X_fit[fit_cols], y_fit)
                member_preds.append(np.asarray(
                    mb.predict_rul(X_val[available + ["unit_id", "cycle"]]),
                    dtype=float,
                ))
            except Exception as exc:  # noqa: BLE001 – skip failed members
                print(f"    ensemble member {b} failed: {exc}")
        if len(member_preds) >= 2:
            resid_std = max(float(np.std(cal_resid)), 1.0)
            ens = _ensemble_variance(member_preds, alpha, resid_std=resid_std)
            im_ens = interval_metrics(
                rul_val_true, np.asarray(ens["lower"]), np.asarray(ens["upper"]), alpha
            )
            method_results["ensemble_variance"].append({
                "picp": im_ens.picp, "mean_width": im_ens.mean_width,
                "winkler": im_ens.winkler,
            })
        else:
            print(f"    ensemble_variance fold {fold_idx}: <2 members, skipped")

        # 3. Quantile bootstrap (residual quantiles from calibration split)
        qboot = _quantile_regression(
            rul_cal_true, rul_cal.astype(float), alpha, n_bootstrap=n_bootstraps
        )
        lower_qb = rul_val.astype(float) + qboot["lower_offset"]
        upper_qb = rul_val.astype(float) + qboot["upper_offset"]
        im_qb = interval_metrics(rul_val_true, lower_qb, upper_qb, alpha)
        method_results["quantile_bootstrap"].append({
            "picp": im_qb.picp, "mean_width": im_qb.mean_width, "winkler": im_qb.winkler,
        })

        # 4. Bootstrap PI with residual augmentation
        boot = _bootstrap_pi(
            model, X_fit, y_fit,
            X_val, available, n_bootstraps=n_bootstraps, alpha=alpha, seed=42,
            residuals=cal_resid,
        )
        if "error" not in boot:
            im_boot = interval_metrics(rul_val_true, np.array(boot["lower"]), np.array(boot["upper"]), alpha)  # noqa: E501
            method_results["bootstrap"].append({
                "picp": im_boot.picp, "mean_width": im_boot.mean_width, "winkler": im_boot.winkler,
            })
        else:
            print(f"    bootstrap fold {fold_idx}: {boot['error']}")

    # Aggregate
    summary = {}
    for method, fold_results in method_results.items():
        if fold_results:
            summary[method] = {
                "mean_picp": float(np.mean([r["picp"] for r in fold_results])),
                "mean_width": float(np.mean([r["mean_width"] for r in fold_results])),
                "mean_winkler": float(np.mean([r["winkler"] for r in fold_results])),
                "std_picp": float(np.std([r["picp"] for r in fold_results])),
                "n_folds": len(fold_results),
            }
        else:
            summary[method] = {"error": "no successful folds"}

    return {"per_fold": method_results, "summary": summary}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_interval_comparison(summary: dict, out_dir: Path) -> None:
    """Bar chart comparing PICP, mean width, and Winkler across methods."""
    methods = [m for m in summary if "mean_picp" in summary[m]]
    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # PICP
    picps = [summary[m]["mean_picp"] for m in methods]
    axes[0].bar(methods, picps, color="steelblue")
    axes[0].axhline(0.9, color="red", linestyle="--", label="Nominal 90%")
    axes[0].set_ylabel("PICP")
    axes[0].set_title("Coverage (PICP)")
    axes[0].legend()

    # Mean width
    widths = [summary[m]["mean_width"] for m in methods]
    axes[1].bar(methods, widths, color="coral")
    axes[1].set_ylabel("Mean Width")
    axes[1].set_title("Interval Width (lower is better)")

    # Winkler
    winklers = [summary[m]["mean_winkler"] for m in methods]
    axes[2].bar(methods, winklers, color="mediumpurple")
    axes[2].set_ylabel("Winkler Score")
    axes[2].set_title("Winkler Score (lower is better)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Uncertainty Method Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "uncertainty_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_vs_width(summary: dict, out_dir: Path) -> None:
    """Scatter: coverage vs interval width for each method."""
    methods = [m for m in summary if "mean_picp" in summary[m]]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        ax.scatter(summary[m]["mean_width"], summary[m]["mean_picp"], s=100, label=m)
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="Nominal 90%")
    ax.set_xlabel("Mean Interval Width")
    ax.set_ylabel("PICP")
    ax.set_title("Coverage vs Width Trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "coverage_vs_width.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e9(cfg, df, out_dir) -> dict[str, Any]:
    """Run uncertainty method comparison."""
    feature_cols = feature_columns(df, include_health=True)

    print("  Comparing uncertainty methods...")
    with Timer() as t:
        results = _compare_uncertainty_methods(cfg, df, feature_cols)
    results["elapsed_seconds"] = t.elapsed

    # Print summary
    summary = results.get("summary", {})
    for method, stats in summary.items():
        if "mean_picp" in stats:
            print(f"    {method}: PICP={stats['mean_picp']:.4f}, "
                  f"Width={stats['mean_width']:.2f}, Winkler={stats['mean_winkler']:.2f}")

    out = {
        "dataset": cfg.dataset.name,
        "alpha": cfg.uncertainty.alpha,
        "uncertainty_comparison": results,
    }

    save_json(out, out_dir / "e9_results.json")
    _plot_interval_comparison(summary, out_dir)
    _plot_coverage_vs_width(summary, out_dir)

    return out


def main() -> None:
    """CLI entry-point for e9_uncertainty_comparison."""
    parser = argparse.ArgumentParser(
        description="E9: Uncertainty Method Comparison – conformal, ensemble, quantile, bootstrap."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True, rul_clip=cfg.dataset.rul_clip)
    out_dir = setup_output("e9_uncertainty_comparison")

    with Timer() as t:
        run_e9(cfg, df, out_dir)

    print(f"\n[e9] Completed in {t.elapsed:.1f}s")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e9_uncertainty_comparison",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
