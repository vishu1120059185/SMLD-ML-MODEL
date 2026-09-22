"""Experiment e0 – Data Audit.

Loads all subsets, reports unit counts, cycle-length distribution,
constant sensors, and operational-condition clusters.  Outputs JSON
summaries and figures to ``results/e0/``.

Usage::

    python -m stattwin.experiments.e0_data_audit --ds FD001 --profile smoke
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

from stattwin.data.loader import load_cmapss
from stattwin.data.schema import COLUMN_NAMES, OP_SETTINGS, SENSOR_NAMES

from ._common import (
    SENSOR_COLS,
    Timer,
    add_common_args,
    ensure_dir,
    resolve_raw_path,
    save_json,
    setup_output,
)


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def _unit_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Per-unit cycle counts and RUL range."""
    grp = df.groupby("unit_id")
    cycles = grp["cycle"].agg(["count", "min", "max"]).rename(
        columns={"count": "n_cycles", "min": "first_cycle", "max": "last_cycle"}
    )
    rul_range = grp["RUL"].agg(["min", "max"]).rename(
        columns={"min": "rul_min", "max": "rul_max"}
    )
    merged = cycles.join(rul_range)
    return {
        "n_units": int(df["unit_id"].nunique()),
        "n_rows": int(len(df)),
        "cycles_per_unit": {
            "mean": float(merged["n_cycles"].mean()),
            "std": float(merged["n_cycles"].std()),
            "min": int(merged["n_cycles"].min()),
            "max": int(merged["n_cycles"].max()),
            "median": float(merged["n_cycles"].median()),
        },
        "rul_range": {
            "overall_min": float(df["RUL"].min()),
            "overall_max": float(df["RUL"].max()),
        },
        "per_unit": merged.reset_index().to_dict(orient="records"),
    }


def _constant_sensors(df: pd.DataFrame, sensor_cols: list[str]) -> dict[str, Any]:
    """Identify sensors with zero or near-zero variance."""
    variances = df[sensor_cols].var()
    constant = variances[variances == 0].index.tolist()
    near_zero = variances[variances < 1e-10].index.tolist()
    return {
        "constant_sensors": constant,
        "near_zero_variance_sensors": near_zero,
        "variances": variances.to_dict(),
    }


def _operating_condition_clusters(
    df: pd.DataFrame,
    settings: list[str] | None = None,
) -> dict[str, Any]:
    """Cluster operational settings to identify distinct regimes."""
    if settings is None:
        settings = [s for s in OP_SETTINGS if s in df.columns]

    if not settings:
        return {"n_clusters": 0, "cluster_sizes": {}}

    X = df[settings].values
    # Simple approach: round to 2 decimal places and find unique combos
    X_rounded = np.round(X, 2)
    unique_rows, counts = np.unique(X_rounded, axis=0, return_counts=True)

    n_clusters = len(unique_rows)
    cluster_sizes = {str(i): int(c) for i, c in enumerate(counts)}

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "unique_settings": unique_rows.tolist(),
        "settings_counts": counts.tolist(),
    }


def _cycle_length_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Histogram data for cycle lengths across units."""
    cycle_counts = df.groupby("unit_id")["cycle"].count().values
    hist, bin_edges = np.histogram(cycle_counts, bins=20)
    return {
        "mean": float(np.mean(cycle_counts)),
        "std": float(np.std(cycle_counts)),
        "min": int(np.min(cycle_counts)),
        "max": int(np.max(cycle_counts)),
        "median": float(np.median(cycle_counts)),
        "hist_counts": hist.tolist(),
        "hist_bin_edges": bin_edges.tolist(),
    }


def _sensor_statistics(df: pd.DataFrame, sensor_cols: list[str]) -> dict[str, Any]:
    """Basic descriptive statistics per sensor."""
    stats = df[sensor_cols].describe().to_dict()
    return stats


def _correlation_matrix(df: pd.DataFrame, sensor_cols: list[str]) -> dict[str, Any]:
    """Pearson correlation matrix for sensors."""
    corr = df[sensor_cols].corr()
    return {
        "columns": sensor_cols,
        "values": corr.values.tolist(),
    }


def _missing_fractions(df: pd.DataFrame, sensor_cols: list[str]) -> dict[str, Any]:
    """Fraction of NaN per sensor column."""
    miss = df[sensor_cols].isna().mean()
    return {
        col: float(v) for col, v in miss.items() if v > 0
    } if miss.any() else {"no_missing": True}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_cycle_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of cycle lengths per unit."""
    cycle_counts = df.groupby("unit_id")["cycle"].count()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cycle_counts.values, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cycles per unit")
    ax.set_ylabel("Count of units")
    ax.set_title("Cycle-Length Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "cycle_distribution.png", dpi=150)
    plt.close(fig)


def _plot_rul_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of RUL values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["RUL"].values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("RUL Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "rul_distribution.png", dpi=150)
    plt.close(fig)


def _plot_sensor_variance(
    df: pd.DataFrame, sensor_cols: list[str], out_dir: Path
) -> None:
    """Bar chart of per-sensor variance."""
    variances = df[sensor_cols].var()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(sensor_cols)), variances.values, tick_label=sensor_cols)
    ax.set_ylabel("Variance")
    ax.set_title("Per-Sensor Variance")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "sensor_variance.png", dpi=150)
    plt.close(fig)


def _plot_operating_conditions(
    df: pd.DataFrame, out_dir: Path
) -> None:
    """Scatter plot of op_setting_1 vs op_setting_2 coloured by op_setting_3."""
    settings = [s for s in OP_SETTINGS if s in df.columns]
    if len(settings) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(settings) >= 3:
        sc = ax.scatter(
            df[settings[0]], df[settings[1]],
            c=df[settings[2]], cmap="viridis", alpha=0.3, s=5,
        )
        plt.colorbar(sc, ax=ax, label=settings[2])
    else:
        ax.scatter(df[settings[0]], df[settings[1]], alpha=0.3, s=5)
    ax.set_xlabel(settings[0])
    ax.set_ylabel(settings[1])
    ax.set_title("Operating Conditions")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "operating_conditions.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e0(cfg, df, out_dir) -> dict[str, Any]:
    """Run the data audit and return results dict."""
    sensor_cols = [c for c in SENSOR_COLS if c in df.columns]

    results: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "unit_summary": _unit_summary(df),
        "cycle_length_distribution": _cycle_length_distribution(df),
        "constant_sensors": _constant_sensors(df, sensor_cols),
        "missing_fractions": _missing_fractions(df, sensor_cols),
        "operating_conditions": _operating_condition_clusters(df),
        "sensor_statistics": _sensor_statistics(df, sensor_cols),
        "correlation_matrix": _correlation_matrix(df, sensor_cols),
    }

    # Save results
    save_json(results, out_dir / "e0_results.json")

    # Generate figures
    _plot_cycle_distribution(df, out_dir)
    _plot_rul_distribution(df, out_dir)
    _plot_sensor_variance(df, sensor_cols, out_dir)
    _plot_operating_conditions(df, out_dir)

    return results


def main() -> None:
    """CLI entry-point for e0_data_audit."""
    parser = argparse.ArgumentParser(
        description="E0: Data Audit – load subsets, unit counts, cycle distributions, constant sensors."
    )
    add_common_args(parser)
    args = parser.parse_args()

    from stattwin.config import load_config

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)

    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        print(f"ERROR: Raw file not found: {raw_path}", file=sys.stderr)
        print(
            "Download from https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository\n"
            f"and place '{cfg.dataset.name}.txt' in data/raw/CMAPSS/",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e0_data_audit")

    with Timer() as t:
        results = run_e0(cfg, df, out_dir)

    print(f"[e0] Completed in {t.elapsed:.1f}s")
    print(f"     Dataset: {cfg.dataset.name}")
    print(f"     Units:   {results['unit_summary']['n_units']}")
    print(f"     Rows:    {results['unit_summary']['n_rows']}")
    print(f"     Output:  {out_dir}")

    # Write manifest
    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e0_data_audit", extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
