"""Experiment e1 – SHI Health-Index Validation.

Validates the Statistical Health Index via monotonicity, trendability,
prognosability, Spearman vs RUL.  Tests SHI weights (fixed vs calibrated)
and threshold sensitivity.  Outputs to ``results/e1/``.

Usage::

    python -m stattwin.experiments.e1_health_index --ds FD001 --profile smoke
"""

from __future__ import annotations

import argparse
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
from stattwin.health.quality import (
    compute_quality_metrics,
    monotonicity,
    prognosability,
    spearman_rul_correlation,
    trendability,
)
from stattwin.health.shi import EvidenceComponent, compute_shi
from stattwin.health.states import classify_states

from ._common import (
    SENSOR_COLS,
    Timer,
    add_common_args,
    resolve_raw_path,
    save_json,
    setup_output,
)


# ---------------------------------------------------------------------------
# Weight sensitivity sweep
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "deviation": 0.25,
    "trend": 0.25,
    "ewma": 0.20,
    "variance": 0.15,
    "corr_shift": 0.15,
}


def _sweep_weight(
    df: pd.DataFrame,
    sensor_cols: list[str],
    component_name: str,
    weight_values: list[float],
    baseline_cycles: int = 30,
) -> list[dict[str, Any]]:
    """Sweep one component's weight and compute quality metrics."""
    results = []
    for w in weight_values:
        weights = dict(_DEFAULT_WEIGHTS)
        weights[component_name] = w
        # Re-normalise
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        components = [EvidenceComponent(name=k, weight=v) for k, v in weights.items()]
        hi = compute_shi(
            df, sensor_cols=sensor_cols,
            evidence_components=components,
            baseline_cycles=baseline_cycles,
        )
        shi_df = hi.shi_values

        quality = compute_quality_metrics(shi_df, rul_df=df)
        results.append({
            "weight": w,
            "weights": weights,
            "monotonicity_mean": quality.summary.get("monotonicity_mean", np.nan),
            "trendability": quality.summary.get("trendability", np.nan),
            "prognosability": quality.summary.get("prognosability", np.nan),
            "spearman_rho_mean": quality.summary.get("spearman_rho_mean", np.nan),
        })
    return results


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def _threshold_sensitivity(
    shi_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    threshold_grids: list[list[float]],
) -> list[dict[str, Any]]:
    """Test different state-classification thresholds."""
    results = []
    for grid in threshold_grids:
        states_df = classify_states(
            shi_df,
            method="fixed_grid",
            fixed_grid=grid,
            rul_df=rul_df,
        )
        # Evaluate: what fraction of units are in CRITICAL or FAILURE_LIKELY
        # in the last 10% of their life?
        state_stats = []
        for uid, grp in states_df.groupby("unit_id"):
            n = len(grp)
            if n < 10:
                continue
            late = grp.iloc[int(n * 0.8):]
            early = grp.iloc[:int(n * 0.2)]
            late_bad = (late["health_state"] <= 1).mean()
            early_good = (early["health_state"] >= 3).mean()
            state_stats.append({
                "unit_id": uid,
                "late_critical_frac": late_bad,
                "early_healthy_frac": early_good,
            })

        if state_stats:
            avg_late = np.mean([s["late_critical_frac"] for s in state_stats])
            avg_early = np.mean([s["early_healthy_frac"] for s in state_stats])
        else:
            avg_late = np.nan
            avg_early = np.nan

        results.append({
            "thresholds": grid,
            "avg_late_critical_fraction": avg_late,
            "avg_early_healthy_fraction": avg_early,
        })
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_shi_examples(
    df: pd.DataFrame, shi_df: pd.DataFrame, out_dir: Path, n_units: int = 5
) -> None:
    """Plot SHI trajectories for a few sample units."""
    units = df["unit_id"].unique()[:n_units]
    fig, axes = plt.subplots(1, n_units, figsize=(4 * n_units, 4), sharey=True)
    if n_units == 1:
        axes = [axes]
    for ax, uid in zip(axes, units):
        u_shi = shi_df[shi_df["unit_id"] == uid]
        u_rul = df[df["unit_id"] == uid]
        ax.plot(u_shi["cycle"], u_shi["shi"], label="SHI", linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(u_rul["cycle"], u_rul["RUL"], color="red", alpha=0.4, label="RUL")
        ax.set_title(f"Unit {uid}")
        ax.set_xlabel("Cycle")
        if ax == axes[0]:
            ax.set_ylabel("SHI")
            ax2.set_ylabel("RUL")
    fig.suptitle("SHI vs RUL – Sample Units", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "shi_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_weight_sensitivity(
    sweep_results: list[dict[str, Any]], component: str, out_dir: Path
) -> None:
    """Plot metric vs weight for a single component."""
    weights = [r["weight"] for r in sweep_results]
    metrics = {
        "monotonicity": [r["monotonicity_mean"] for r in sweep_results],
        "prognosability": [r["prognosability"] for r in sweep_results],
        "spearman_rho": [r["spearman_rho_mean"] for r in sweep_results],
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, vals in metrics.items():
        ax.plot(weights, vals, marker="o", label=name)
    ax.set_xlabel(f"Weight: {component}")
    ax.set_ylabel("Metric value")
    ax.set_title(f"SHI Sensitivity – {component}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"sensitivity_{component}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e1(cfg, df, out_dir) -> dict[str, Any]:
    """Run e1 health-index validation."""
    sensor_cols = [c for c in SENSOR_COLS if c in df.columns]

    # 1. Compute SHI with default weights
    hi = compute_shi(df, sensor_cols=sensor_cols, baseline_cycles=cfg.stats.baseline_cycles)
    shi_df = hi.shi_values

    # 2. Quality metrics
    quality = compute_quality_metrics(shi_df, rul_df=df)

    # 3. Weight sensitivity (sweep each component)
    weight_sweep: dict[str, list] = {}
    for comp in _DEFAULT_WEIGHTS:
        sweep = _sweep_weight(df, sensor_cols, comp, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                              baseline_cycles=cfg.stats.baseline_cycles)
        weight_sweep[comp] = sweep

    # 4. Threshold sensitivity
    threshold_grids = [
        [80.0, 60.0, 40.0],
        [75.0, 50.0, 25.0],
        [85.0, 65.0, 45.0],
        [70.0, 45.0, 20.0],
        [90.0, 70.0, 50.0],
    ]
    threshold_results = _threshold_sensitivity(shi_df, df, threshold_grids)

    # 5. Assemble results
    results: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "quality_metrics": quality.summary,
        "per_unit_monotonicity": quality.monotonicity.to_dict(orient="records"),
        "weight_sensitivity": weight_sweep,
        "threshold_sensitivity": threshold_results,
        "shi_weights_used": hi.weights,
        "sensor_weights_used": hi.sensor_weights,
    }

    save_json(results, out_dir / "e1_results.json")

    # 6. Figures
    _plot_shi_examples(df, shi_df, out_dir, n_units=5)
    for comp, sweep in weight_sweep.items():
        _plot_weight_sensitivity(sweep, comp, out_dir)

    return results


def main() -> None:
    """CLI entry-point for e1_health_index."""
    parser = argparse.ArgumentParser(
        description="E1: SHI Health-Index Validation – monotonicity, trendability, prognosability."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e1_health_index")

    with Timer() as t:
        results = run_e1(cfg, df, out_dir)

    qm = results["quality_metrics"]
    print(f"[e1] Completed in {t.elapsed:.1f}s")
    print(f"     Monotonicity:  {qm.get('monotonicity_mean', 'N/A'):.4f}")
    print(f"     Trendability:  {qm.get('trendability', 'N/A'):.4f}")
    print(f"     Prognosability:{qm.get('prognosability', 'N/A'):.4f}")
    print(f"     Spearman ρ:    {qm.get('spearman_rho_mean', 'N/A'):.4f}")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e1_health_index",
                  extras={"elapsed_seconds": t.elapsed, "quality_metrics": qm})


if __name__ == "__main__":
    main()
