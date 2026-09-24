"""Experiment e3 – Early Warning Benchmark.

Benchmarks early-warning performance at a matched false-alarm budget.
Computes lead-time statistics: mean, median, min, max lead time,
and false-alarm rate.  Outputs to ``results/e3/``.

Usage::

    python -m stattwin.experiments.e3_early_warning --ds FD001 --profile smoke
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
from stattwin.evaluation.lead_time import (
    LeadTimeReport,
    lead_time_analysis,
    tune_tau_for_budget,
)
from stattwin.models import LogisticModel, RandomForestModel, XGBoostModel

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
# Helpers
# ---------------------------------------------------------------------------

def _prepare_warning_df(
    df: pd.DataFrame,
    proba: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """Build a DataFrame with failure_cycle for lead-time analysis."""
    out = df[["unit_id", "cycle"]].copy()
    out["fail_prob"] = proba.values

    # Compute failure cycle: first cycle where RUL <= horizon
    failure_cycles = {}
    for uid, grp in df.groupby("unit_id"):
        failed_rows = grp[grp["RUL"] <= horizon]
        if len(failed_rows) > 0:
            failure_cycles[uid] = int(failed_rows["cycle"].min())

    out["failure_cycle"] = out["unit_id"].map(failure_cycles)
    return out


def _run_early_warning_for_model(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    splits: list[dict],
    feature_cols: list[str],
    horizon: int,
    far_budget: float,
) -> dict[str, Any]:
    """Evaluate early-warning for one model via OOF with budget-matched tau."""
    all_lead_reports: list[LeadTimeReport] = []

    for fold_idx, split in enumerate(splits):
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()
        y_tr = y.loc[X_tr.index]
        y.loc[X_val.index]

        m = copy.deepcopy(model)
        try:
            m.fit(X_tr[feature_cols + ["unit_id", "cycle"]], y_tr)
            proba_val = m.predict_proba(X_val[feature_cols + ["unit_id", "cycle"]])
        except Exception as e:
            print(f"    Fold {fold_idx} failed: {e}")
            continue

        col = label_col_for(horizon)
        if col not in proba_val.columns:
            continue

        # Split into tune / eval (50/50 of validation units)
        val_train, val_eval = inner_unit_split(val_units, val_frac=0.5, seed=42 + fold_idx)

        # Tune tau on first half
        df_val = _prepare_warning_df(X_val, proba_val[col], horizon)
        df_tune = df_val[df_val["unit_id"].isin(val_train)]
        df_eval_set = df_val[df_val["unit_id"].isin(val_eval)]

        if len(df_tune) == 0 or len(df_eval_set) == 0:
            continue

        best_tau, tune_report = tune_tau_for_budget(
            df_tune,
            far_budget=far_budget,
            warning_horizon=horizon,
            persistence=3,
        )

        # Evaluate on second half
        eval_report = lead_time_analysis(
            df_eval_set,
            tau=best_tau,
            warning_horizon=horizon,
            persistence=3,
        )
        all_lead_reports.append(eval_report)

    if not all_lead_reports:
        return {"model": model.name, "error": "no successful folds"}

    # Aggregate
    lead_times_all = []
    for r in all_lead_reports:
        lead_times_all.extend(r.lead_times)

    return {
        "model": model.name,
        "horizon": horizon,
        "far_budget": far_budget,
        "n_folds": len(all_lead_reports),
        "mean_lead_time": float(np.mean(lead_times_all)) if lead_times_all else np.nan,
        "median_lead_time": float(np.median(lead_times_all)) if lead_times_all else np.nan,
        "min_lead_time": float(np.min(lead_times_all)) if lead_times_all else np.nan,
        "max_lead_time": float(np.max(lead_times_all)) if lead_times_all else np.nan,
        "std_lead_time": float(np.std(lead_times_all)) if lead_times_all else np.nan,
        "n_warned": sum(r.n_warned for r in all_lead_reports),
        "n_failed": sum(r.n_failed for r in all_lead_reports),
        "mean_far": float(np.mean([r.far for r in all_lead_reports if not np.isnan(r.far)])),
        "per_fold": [
            {
                "fold": i,
                "tau": r.tau,
                "far": r.far,
                "n_warned": r.n_warned,
                "mean_lead_time": r.mean_lead_time,
            }
            for i, r in enumerate(all_lead_reports)
        ],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_lead_time_comparison(all_results: list[dict], out_dir: Path) -> None:
    """Box plot of lead times across models."""
    data = []
    labels = []
    for r in all_results:
        if "error" in r or not r.get("per_fold"):
            continue
        fold_lts = [
            ft["mean_lead_time"]
            for ft in r["per_fold"]
            if ft["mean_lead_time"] is not None and not np.isnan(ft["mean_lead_time"])
        ]
        if fold_lts:
            data.append(fold_lts)
            labels.append(r["model"])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Lead Time (cycles)")
    ax.set_title("Early Warning Lead Time by Model")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "lead_time_comparison.png", dpi=150)
    plt.close(fig)


def _plot_far_vs_lead(all_results: list[dict], out_dir: Path) -> None:
    """Scatter: FAR vs mean lead time for each model."""
    models, fars, leads = [], [], []
    for r in all_results:
        if "error" in r:
            continue
        models.append(r["model"])
        fars.append(r.get("mean_far", np.nan))
        leads.append(r.get("mean_lead_time", np.nan))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fars, leads, s=100)
    for i, m in enumerate(models):
        ax.annotate(m, (fars[i], leads[i]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Mean Lead Time (cycles)")
    ax.set_title("FAR vs Lead Time Trade-off")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "far_vs_lead.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e3(cfg, df, out_dir) -> dict[str, Any]:
    """Run early-warning benchmark."""
    [c for c in SENSOR_COLS if c in df.columns]
    feature_cols = feature_columns(df, include_health=True)

    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    y = df[label_cols].copy()

    splits = make_group_kfold_splits(df, n_splits=cfg.split.n_splits, seed=cfg.seed)

    horizon = cfg.warning.horizon
    far_budget = cfg.warning.far_budget

    models = [
        LogisticModel(horizons=FAILURE_HORIZONS),
        RandomForestModel(horizons=FAILURE_HORIZONS),
        XGBoostModel(horizons=FAILURE_HORIZONS),
    ]

    all_results = []
    for model in models:
        print(f"  Early warning: {model.name} (H={horizon}, FAR≤{far_budget})...")
        with Timer() as t:
            result = _run_early_warning_for_model(
                model, df, y, splits, feature_cols, horizon, far_budget,
            )
        result["elapsed_seconds"] = t.elapsed
        all_results.append(result)
        if "error" not in result:
            print(f"    Mean lead time: {result['mean_lead_time']:.1f}  FAR: {result['mean_far']:.4f}")  # noqa: E501
        else:
            print(f"    {result['error']}")

    results: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "horizon": horizon,
        "far_budget": far_budget,
        "models": all_results,
    }

    save_json(results, out_dir / "e3_results.json")
    _plot_lead_time_comparison(all_results, out_dir)
    _plot_far_vs_lead(all_results, out_dir)

    return results


def main() -> None:
    """CLI entry-point for e3_early_warning."""
    parser = argparse.ArgumentParser(
        description="E3: Early Warning Benchmark – lead-time stats at matched FAR budget."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e3_early_warning")

    with Timer() as t:
        run_e3(cfg, df, out_dir)

    print(f"\n[e3] Completed in {t.elapsed:.1f}s")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e3_early_warning",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
