"""Experiment e2 – Model Comparison.

Compares 8 prediction models: Logistic Regression, Random Forest,
XGBoost, Threshold, soft-vote Ensemble, GRU, LSTM, and Hybrid.
Per-horizon classification metrics (aggregated out-of-fold **and per
fold**), RUL metrics, and NASA score.  Tabular models can run on raw
sensor columns or on the engineered feature library via ``--features``;
sequence models (GRU/LSTM) and the threshold model always use raw
sensor columns.  Outputs comparison tables and charts to
``results/e2_model_comparison/``.

Usage::

    python -m stattwin.experiments.e2_model_comparison --ds FD001 --profile fast
    python -m stattwin.experiments.e2_model_comparison --features raw
"""

from __future__ import annotations

import argparse
import copy
import time
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
    EnsembleModel,
    GRUModel,
    HybridModel,
    LogisticModel,
    RandomForestModel,
    ThresholdModel,
    XGBoostModel,
    build_member,
)
from stattwin.statistics import compute_features

from ._common import (
    Timer,
    add_common_args,
    feature_columns,
    resolve_raw_path,
    save_json,
    sensor_columns,
    setup_output,
)

# Models that must only ever see raw sensor / operating-setting columns
_RAW_ONLY_TYPES = (GRUModel, ThresholdModel)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_models(cfg) -> list:
    """Instantiate all models with config hyperparameters.

    Deep-learning models (GRU, LSTM, Hybrid) are appended only when
    torch is importable.
    """
    m = cfg.model
    models: list = [
        LogisticModel(
            C=m.lr.C, max_iter=m.lr.max_iter, horizons=FAILURE_HORIZONS
        ),
        RandomForestModel(
            n_estimators=m.rf.n_estimators,
            max_depth=m.rf.max_depth,
            horizons=FAILURE_HORIZONS,
        ),
        XGBoostModel(
            n_estimators=m.xgb.n_estimators,
            max_depth=m.xgb.max_depth,
            learning_rate=m.xgb.learning_rate,
            horizons=FAILURE_HORIZONS,
        ),
        ThresholdModel(horizons=FAILURE_HORIZONS),
        EnsembleModel(
            members=[
                (name, build_member(name, cfg, FAILURE_HORIZONS))
                for name in m.ensemble.members
            ],
            weight_mode=(
                "learned" if m.ensemble.weights == "learned" else "uniform"
            ),
            val_fraction=m.ensemble.val_fraction,
            horizons=FAILURE_HORIZONS,
        ),
    ]

    try:
        import torch  # noqa: F401
    except ImportError:
        return models

    train_kw = dict(
        seq_len=m.gru.seq_len,
        lr=m.gru.lr,
        batch_size=m.gru.batch_size,
        epochs=m.gru.epochs,
        patience=m.gru.patience,
        horizon_weight=m.gru.horizon_weight,
        normalize_features=m.gru.normalize_features,
        val_fraction=m.gru.val_fraction,
        weight_decay=m.gru.weight_decay,
        raw_only=True,
        horizons=FAILURE_HORIZONS,
    )
    models.append(
        GRUModel(hidden=m.gru.hidden, layers=m.gru.layers, dropout=m.gru.dropout,
                 **train_kw)
    )
    models.append(
        GRUModel(backbone="lstm", hidden=m.lstm.hidden, layers=m.lstm.layers,
                 dropout=m.lstm.dropout, **train_kw)
    )
    models.append(
        HybridModel(
            hidden=m.gru.hidden,
            layers=m.gru.layers,
            dropout=m.gru.dropout,
            seq_len=m.gru.seq_len,
            ensemble_size=m.gru.ensemble_size,
            lr=m.gru.lr,
            batch_size=m.gru.batch_size,
            epochs=m.gru.epochs,
            patience=m.gru.patience,
            horizons=FAILURE_HORIZONS,
        )
    )
    return models


# ---------------------------------------------------------------------------
# Single-model OOF evaluation
# ---------------------------------------------------------------------------

def _fold_metrics(
    fold_idx: int,
    y_val: pd.DataFrame,
    proba: pd.DataFrame,
    rul_true: pd.Series,
    rul_pred: pd.Series,
    val_units: list,
) -> dict[str, Any]:
    """Compute per-fold classification and RUL metrics."""
    entry: dict[str, Any] = {
        "fold": fold_idx,
        "n_val": int(len(y_val)),
        "val_units": [int(u) for u in val_units],
    }
    try:
        y_true_dict: dict[int, np.ndarray] = {}
        y_prob_dict: dict[int, np.ndarray] = {}
        for h in FAILURE_HORIZONS:
            col = label_col_for(h)
            if col in y_val.columns and col in proba.columns:
                y_true_dict[h] = y_val[col].values.astype(int)
                y_prob_dict[h] = proba[col].values.astype(float)
        cls_reports = evaluate_classification(y_true_dict, y_prob_dict)
        rul_report = evaluate_rul(
            rul_true.values.astype(float), rul_pred.values.astype(float)
        )
        entry["classification"] = [
            {
                "horizon": r.horizon,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "roc_auc": r.roc_auc,
                "pr_auc": r.pr_auc,
            }
            for r in cls_reports
        ]
        entry["rul"] = {
            "mae": rul_report.mae,
            "rmse": rul_report.rmse,
            "nasa_score": rul_report.nasa_score,
            "n": rul_report.n,
        }
    except Exception as e:  # noqa: BLE001 – a single-class fold must not kill e2
        entry["error"] = str(e)
    return entry


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
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, split in enumerate(splits):
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()
        y_tr = y.loc[X_tr.index]
        y_val = y.loc[X_val.index]

        m = copy.deepcopy(model)
        try:
            fit_cols = feature_cols + ["unit_id", "cycle"]
            if "RUL" in X_tr.columns:
                fit_cols = fit_cols + ["RUL"]
            m.fit(X_tr[fit_cols], y_tr)
            predict_frame = X_val[feature_cols + ["unit_id", "cycle"]]
            proba = m.predict_proba(predict_frame)
            rul_pred = m.predict_rul(predict_frame)
        except Exception as e:
            print(f"  Fold {fold_idx} failed for {model.name}: {e}")
            fold_reports.append({"fold": fold_idx, "error": str(e)})
            continue

        rul_true = rul_series.loc[X_val.index]
        fold_entry = _fold_metrics(
            fold_idx, y_val, proba, rul_true, rul_pred,
            [int(u) for u in val_units],
        )
        if isinstance(m, EnsembleModel):
            fold_entry["ensemble_weights"] = {
                "prob": {k: round(v, 4) for k, v in m.prob_weights_.items()},
                "rul": {k: round(v, 4) for k, v in m.rul_weights_.items()},
                "diagnostics": m._weight_diagnostics_,
            }
        fold_reports.append(fold_entry)

        all_proba.append(proba)
        all_rul_pred.append(rul_pred)
        all_rul_true.append(rul_true)
        all_y_true.append(y_val)

    if not all_proba:
        return {
            "model": model.name,
            "error": "all folds failed",
            "folds": fold_reports,
        }

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
        "folds": fold_reports,
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
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add the statistical feature library + SHI health columns.

    Only raw sensor columns are engineered so that label columns can
    never leak into the feature set.
    """
    cols = sensor_columns(df)
    t0 = time.time()
    out, _spec = compute_features(df, cfg, sensor_cols=cols)
    print(
        f"    compute_features: {time.time() - t0:.1f}s "
        f"({out.shape[1] - df.shape[1]} new cols)",
        flush=True,
    )
    try:
        from stattwin.health import compute_shi

        t0 = time.time()
        hi = compute_shi(
            out,
            sensor_cols=cols,
            baseline_cycles=cfg.stats.baseline_cycles,
        )
        print(f"    compute_shi: {time.time() - t0:.1f}s", flush=True)
        shi_df = hi.shi_values[["unit_id", "cycle", "shi"]].rename(
            columns={"shi": "shi_score"}
        )
        out = out.merge(shi_df, on=["unit_id", "cycle"], how="left")
        out["shi_score"] = out["shi_score"].ffill().bfill().fillna(50.0)
    except Exception as exc:  # noqa: BLE001 – health columns are optional
        print(f"    SHI skipped: {exc}", flush=True)
        out["shi_score"] = 50.0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e2(
    cfg, df, out_dir, features: str = "engineered"
) -> dict[str, Any]:
    """Run model comparison experiment.

    Parameters
    ----------
    cfg:
        Validated STAT-TWIN configuration.
    df:
        Raw labelled C-MAPSS frame.
    out_dir:
        Output directory (``results/e2_model_comparison``).
    features:
        ``"engineered"`` (default) computes the full statistical feature
        library and feeds it to the tabular models.  ``"raw"`` restricts
        every model to raw sensor / operating-setting columns.
        GRU/LSTM/threshold models always receive raw sensor columns.
    """
    raw_feature_cols = feature_columns(df, include_health=True)

    df_feat = _engineer_features(df, cfg) if features == "engineered" else df
    eng_feature_cols = feature_columns(df_feat, include_health=True)

    # Prepare labels
    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    y = df[label_cols].copy()

    # Splits
    splits = make_group_kfold_splits(df, n_splits=cfg.split.n_splits, seed=cfg.seed)

    # Build models
    models = _build_models(cfg)

    all_results = []
    for model in models:
        if isinstance(model, _RAW_ONLY_TYPES):
            cols, kind = raw_feature_cols, "raw"
        elif features == "engineered":
            cols, kind = eng_feature_cols, "engineered"
        else:
            cols, kind = raw_feature_cols, "raw"
        print(f"  Evaluating {model.name} ({kind}, {len(cols)} features)...")
        with Timer() as t:
            result = _evaluate_single_model(
                model, df_feat, y, splits, cols, df_feat["RUL"]
            )
        result["elapsed_seconds"] = t.elapsed
        result["features"] = {"kind": kind, "n_features": len(cols)}
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
        "features": features,
        "n_splits": cfg.split.n_splits,
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
        description=(
            "E2: Model Comparison – 8 models, per-horizon + per-fold "
            "metrics, NASA score."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--features",
        choices=["raw", "engineered"],
        default="engineered",
        help=(
            "Feature set for tabular models: 'engineered' (statistical "
            "feature library + SHI, default) or 'raw' sensors only."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True, rul_clip=cfg.dataset.rul_clip)
    out_dir = setup_output("e2_model_comparison")

    with Timer() as t:
        results = run_e2(cfg, df, out_dir, features=args.features)

    print(f"\n[e2] Completed in {t.elapsed:.1f}s")
    print(f"     Models evaluated: {len(results['models'])}")
    print(f"     Features: {results['features']}")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(
        out_dir, cfg, args.config, "e2_model_comparison",
        extras={"elapsed_seconds": t.elapsed, "features": args.features},
    )


if __name__ == "__main__":
    main()
