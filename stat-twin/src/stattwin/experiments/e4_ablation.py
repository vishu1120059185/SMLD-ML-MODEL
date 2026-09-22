"""Experiment e4 – Ablation Study (A–E).

Ablation variants:
  A – Raw sensors only
  B – Raw + statistical features
  C – Raw + temporal features (rolling statistics)
  D – Raw + statistical + temporal (full feature set minus health)
  E – Full (raw + statistical + temporal + health / SHI)

Paired Wilcoxon tests, bootstrap CIs, and Holm–Bonferroni correction
for multiple comparisons.  Outputs to ``results/e4/``.

Usage::

    python -m stattwin.experiments.e4_ablation --ds FD001 --profile smoke
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
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.evaluation.significance import (
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_wilcoxon,
)
from stattwin.models import RandomForestModel, XGBoostModel

from ._common import (
    SENSOR_COLS,
    Timer,
    add_common_args,
    resolve_raw_path,
    save_json,
    setup_output,
)


# ---------------------------------------------------------------------------
# Ablation feature sets
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: dict[str, dict[str, bool]] = {
    "A_raw": {"statistical": False, "temporal": False, "health": False},
    "B_raw_stat": {"statistical": True, "temporal": False, "health": False},
    "C_raw_temporal": {"statistical": False, "temporal": True, "health": False},
    "D_raw_stat_temporal": {"statistical": True, "temporal": True, "health": False},
    "E_full": {"statistical": True, "temporal": True, "health": True},
}


def _select_features(
    df: pd.DataFrame,
    statistical: bool,
    temporal: bool,
    health: bool,
) -> list[str]:
    """Select feature columns based on ablation flags."""
    exclude = {"unit_id", "cycle", "RUL"} | {label_col_for(h) for h in FAILURE_HORIZONS}
    all_numeric = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    raw_sensors = [c for c in all_numeric if c.startswith("sensor_") or c.startswith("op_setting_")]
    health_cols = [c for c in all_numeric if "shi" in c.lower() or "evidence_" in c.lower() or "health" in c.lower()]

    stat_temporal = [c for c in all_numeric if c not in raw_sensors and c not in health_cols]
    stat_only = [c for c in stat_temporal if "rmean_" in c or "rstd_" in c or "zscore" in c]
    temporal_only = [c for c in stat_temporal if "slope_" in c or "pctchg_" in c or "roc_" in c or "ewma_" in c]

    cols = list(raw_sensors)
    if statistical:
        cols.extend(stat_only)
    if temporal:
        cols.extend(temporal_only)
    if health:
        cols.extend(health_cols)

    return sorted(set(cols))


# ---------------------------------------------------------------------------
# Per-unit OOF evaluation
# ---------------------------------------------------------------------------

def _oof_per_unit_mae(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    splits: list[dict],
    feature_cols: list[str],
) -> dict[str, list[float]]:
    """Run OOF and return per-unit MAE for RUL."""
    unit_mae: dict[str, list[float]] = {}

    for fold_idx, split in enumerate(splits):
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()
        y_tr = y.loc[X_tr.index]

        m = copy.deepcopy(model)
        try:
            m.fit(X_tr[feature_cols + ["unit_id", "cycle"]], y_tr)
            rul_pred = m.predict_rul(X_val[feature_cols + ["unit_id", "cycle"]])
        except Exception:
            continue

        # Per-unit MAE
        for uid, grp in X_val.groupby("unit_id"):
            idx = grp.index
            true_rul = grp["RUL"].values.astype(float) if "RUL" in grp.columns else None
            pred = rul_pred.loc[idx].values.astype(float)
            if true_rul is not None and len(true_rul) > 0:
                mae = float(np.mean(np.abs(true_rul - pred)))
                unit_mae.setdefault(str(uid), []).append(mae)

    return unit_mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e4(cfg, df, out_dir) -> dict[str, Any]:
    """Run ablation study with significance testing."""
    splits = make_group_kfold_splits(df, n_splits=cfg.split.n_splits, seed=cfg.seed)
    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    y = df[label_cols].copy()

    base_model = XGBoostModel(horizons=FAILURE_HORIZONS)

    variant_results: dict[str, Any] = {}
    variant_unit_mae: dict[str, list[float]] = {}

    for variant_name, flags in ABLATION_VARIANTS.items():
        print(f"  Ablation {variant_name}: stat={flags['statistical']}, "
              f"temp={flags['temporal']}, health={flags['health']}")
        feature_cols = _select_features(df, **flags)
        print(f"    Features: {len(feature_cols)}")

        with Timer() as t:
            unit_mae = _oof_per_unit_mae(base_model, df, y, splits, feature_cols)

        # Aggregate
        all_maes = []
        for uid, maes in unit_mae.items():
            all_maes.extend(maes)

        variant_results[variant_name] = {
            "flags": flags,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols[:20],  # first 20 for brevity
            "mean_mae": float(np.mean(all_maes)) if all_maes else np.nan,
            "std_mae": float(np.std(all_maes)) if all_maes else np.nan,
            "median_mae": float(np.median(all_maes)) if all_maes else np.nan,
            "n_units": len(unit_mae),
            "elapsed_seconds": t.elapsed,
        }
        variant_unit_mae[variant_name] = all_maes
        print(f"    Mean MAE: {variant_results[variant_name]['mean_mae']:.4f}")

    # Pairwise significance tests: E_full vs each other variant
    baseline_key = "E_full"
    pairwise_tests: dict[str, Any] = {}
    p_values: dict[str, float] = {}

    baseline_maes = np.array(variant_unit_mae.get(baseline_key, []))

    for variant_name in ABLATION_VARIANTS:
        if variant_name == baseline_key:
            continue
        other_maes = np.array(variant_unit_mae.get(variant_name, []))

        # Pad to equal length if needed
        min_len = min(len(baseline_maes), len(other_maes))
        if min_len < 5:
            continue

        a = baseline_maes[:min_len]
        b = other_maes[:min_len]

        wilcox = paired_wilcoxon(a, b)
        boot = paired_bootstrap_ci(a, b, n_resamples=5000, seed=cfg.seed)

        test_key = f"{baseline_key}_vs_{variant_name}"
        pairwise_tests[test_key] = {
            "wilcoxon_statistic": wilcox.statistic,
            "wilcoxon_p_value": wilcox.p_value,
            "effect_size": wilcox.effect_size,
            "n_pairs": wilcox.n_pairs,
            "median_diff": wilcox.median_diff,
            "bootstrap_mean_diff": boot.mean_diff,
            "bootstrap_ci_lower": boot.ci_lower,
            "bootstrap_ci_upper": boot.ci_upper,
            "bootstrap_se": boot.se,
        }
        p_values[test_key] = wilcox.p_value

    # Holm-Bonferroni correction
    holm = holm_bonferroni(p_values, alpha=0.05)

    results: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "base_model": base_model.name,
        "variants": variant_results,
        "pairwise_tests": pairwise_tests,
        "holm_bonferroni": {
            "alpha": holm.alpha,
            "adjusted_p_values": holm.adjusted_p_values,
            "rejected": holm.rejected,
        },
    }

    save_json(results, out_dir / "e4_results.json")

    # Figures
    _plot_ablation_comparison(variant_results, out_dir)
    _plot_pairwise_effects(pairwise_tests, out_dir)

    return results


def _plot_ablation_comparison(variant_results: dict, out_dir: Path) -> None:
    """Bar chart of mean MAE across ablation variants."""
    names = list(variant_results.keys())
    maes = [variant_results[n]["mean_mae"] for n in names]
    stds = [variant_results[n]["std_mae"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, maes, yerr=stds, capsize=5, color="steelblue")
    ax.set_ylabel("Mean MAE")
    ax.set_title("Ablation Study – Mean MAE (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "ablation_mae.png", dpi=150)
    plt.close(fig)


def _plot_pairwise_effects(pairwise_tests: dict, out_dir: Path) -> None:
    """Forest plot of bootstrap CIs for pairwise differences."""
    names = list(pairwise_tests.keys())
    means = [pairwise_tests[n]["bootstrap_mean_diff"] for n in names]
    lows = [pairwise_tests[n]["bootstrap_ci_lower"] for n in names]
    highs = [pairwise_tests[n]["bootstrap_ci_upper"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(names))
    ax.errorbar(means, y_pos, xerr=[[m - l for m, l in zip(means, lows)],
                                     [h - m for h, m in zip(highs, means)]],
                fmt="o", capsize=5)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Difference (E_full – other)")
    ax.set_title("Bootstrap 95% CI – Pairwise Differences")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "pairwise_effects.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry-point for e4_ablation."""
    parser = argparse.ArgumentParser(
        description="E4: Ablation Study – 5 variants with paired tests and Holm correction."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e4_ablation")

    with Timer() as t:
        results = run_e4(cfg, df, out_dir)

    print(f"\n[e4] Completed in {t.elapsed:.1f}s")
    for vname, vr in results["variants"].items():
        print(f"     {vname}: MAE={vr['mean_mae']:.4f} ({vr['n_features']} features)")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e4_ablation",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
