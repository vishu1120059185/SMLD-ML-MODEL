"""Shared training and evaluation harness for STAT-TWIN models.

Provides:

* **OOF generation** – ``run_oof`` trains and evaluates every model on
  ``GroupKFold(n_splits=5)`` outer folds, storing out-of-fold predictions.
* **Test prediction** – ``predict_test`` refits on full training data and
  generates test-set predictions.
* **Metrics** – ``compute_metrics`` calculates ROC-AUC, PR-AUC, RMSE,
  MAE, and Spearman correlation per horizon.
* **Evaluation report** – ``evaluate_model`` combines OOF + metrics into a
  structured ``EvalResult`` dataclass.

All transformations (scalers, isotonic calibrators, feature selectors) are
fit on training folds only and applied to validation folds to prevent
leakage.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.models.base import BaseModel, ModelResult

__all__ = [
    "EvalResult",
    "compute_metrics",
    "evaluate_model",
    "predict_test",
    "run_oof",
]


# -----------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------


@dataclass
class FoldResult:
    """Result of evaluating a single outer fold."""

    fold: int
    train_units: np.ndarray
    val_units: np.ndarray
    oof_proba: pd.DataFrame
    oof_rul: pd.Series
    oof_raw_score: pd.Series
    fit_time: float
    predict_time: float
    train_size: int
    val_size: int


@dataclass
class EvalResult:
    """Full evaluation result across all folds."""

    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    oof_proba: pd.DataFrame = field(default_factory=pd.DataFrame)
    oof_rul: pd.Series = field(default_factory=pd.Series)
    oof_raw_score: pd.Series = field(default_factory=pd.Series)
    feature_cols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------


def compute_metrics(
    y_true: pd.DataFrame | np.ndarray,
    y_pred_proba: pd.DataFrame,
    y_true_rul: pd.Series | np.ndarray | None = None,
    y_pred_rul: pd.Series | None = None,
    horizons: Sequence[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-horizon classification and RUL regression metrics.

    Parameters
    ----------
    y_true:
        Binary label DataFrame or array.
    y_pred_proba:
        Predicted probabilities DataFrame.
    y_true_rul:
        True RUL values (optional).
    y_pred_rul:
        Predicted RUL values (optional).
    horizons:
        Failure horizons (default ``FAILURE_HORIZONS``).

    Returns
    -------
    dict of dict
        ``{metric_name: {horizon_str: value}}``
    """
    if horizons is None:
        horizons = FAILURE_HORIZONS

    result: dict[str, dict[str, float]] = {}

    for h in horizons:
        col = label_col_for(h)
        h_key = f"h{h}"

        if col in y_pred_proba.columns:
            proba = y_pred_proba[col].to_numpy()
            if isinstance(y_true, pd.DataFrame):
                if col in y_true.columns:
                    true = y_true[col].to_numpy().astype(np.int32)
                else:
                    continue
            else:
                true = y_true[:, FAILURE_HORIZONS.index(h)] if y_true.ndim > 1 else y_true

            # ROC-AUC
            try:
                if len(np.unique(true)) < 2:
                    result.setdefault("roc_auc", {})[h_key] = 0.5
                else:
                    result.setdefault("roc_auc", {})[h_key] = roc_auc_score(true, proba)
            except ValueError:
                result.setdefault("roc_auc", {})[h_key] = 0.5

            # PR-AUC
            try:
                result.setdefault("pr_auc", {})[h_key] = average_precision_score(true, proba)
            except ValueError:
                result.setdefault("pr_auc", {})[h_key] = 0.0

    # RUL regression metrics
    if y_true_rul is not None and y_pred_rul is not None:
        true_rul = (
            y_true_rul.to_numpy() if isinstance(y_true_rul, pd.Series) else np.asarray(y_true_rul)
        )
        pred_rul = (
            y_pred_rul.to_numpy() if isinstance(y_pred_rul, pd.Series) else np.asarray(y_pred_rul)
        )
        mask = np.isfinite(true_rul) & np.isfinite(pred_rul)
        true_rul = true_rul[mask]
        pred_rul = pred_rul[mask]

        if len(true_rul) > 0:
            result.setdefault("rul_rmse", {})["overall"] = float(
                np.sqrt(mean_squared_error(true_rul, pred_rul))
            )
            result.setdefault("rul_mae", {})["overall"] = float(
                mean_absolute_error(true_rul, pred_rul)
            )
            rho, pval = sp_stats.spearmanr(true_rul, pred_rul)
            result.setdefault("rul_spearman", {})["overall"] = float(rho)

    return result


def _merge_metrics(
    all_fold_metrics: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Average metrics across folds."""
    if not all_fold_metrics:
        return {}

    merged: dict[str, dict[str, list[float]]] = {}
    for fm in all_fold_metrics:
        for metric, hv in fm.items():
            merged.setdefault(metric, {})
            for key, val in hv.items():
                merged[metric].setdefault(key, []).append(val)

    return {
        metric: {key: float(np.mean(vals)) for key, vals in hv.items()}
        for metric, hv in merged.items()
    }


# -----------------------------------------------------------------------
# Feature column detection
# -----------------------------------------------------------------------


def _detect_feature_cols(
    X: pd.DataFrame,
    include_statistical: bool = True,
    include_health: bool = True,
) -> list[str]:
    """Detect feature columns based on ablation variant.

    Parameters
    ----------
    X:
        Feature matrix.
    include_statistical:
        If True, include derived statistical features.
    include_health:
        If True, include health-related features.
    """
    exclude = {"unit_id", "cycle", "RUL"}
    all_numeric = [
        c for c in X.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(X[c])
    ]

    raw_sensor = [c for c in all_numeric if c.startswith("sensor_") or c.startswith("op_setting_")]

    if not include_statistical and not include_health:
        # Ablation variant A: raw sensor features only
        return raw_sensor if raw_sensor else all_numeric

    health_cols = [c for c in all_numeric if "shi" in c.lower() or "health" in c.lower() or "evidence_" in c.lower()]  # noqa: E501
    stat_cols = [c for c in all_numeric if c not in raw_sensor and c not in health_cols]

    cols = list(raw_sensor)
    if include_statistical:
        cols.extend(stat_cols)
    if include_health:
        cols.extend(health_cols)

    return cols if cols else all_numeric


# -----------------------------------------------------------------------
# OOF generation
# -----------------------------------------------------------------------


def run_oof(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    include_statistical: bool = True,
    include_health: bool = True,
    verbose: bool = True,
) -> EvalResult:
    """Run out-of-fold training and evaluation.

    Parameters
    ----------
    model:
        Model instance implementing ``BaseModel`` interface.
    X:
        Full training feature matrix.
    y:
        Full training label DataFrame.
    n_splits:
        Number of GroupKFold folds.
    seed:
        Random seed.
    include_statistical:
        Include statistical features (False → ablation A).
    include_health:
        Include health features.
    verbose:
        Print progress.

    Returns
    -------
    EvalResult
        Structured evaluation result with per-fold and aggregated metrics.
    """
    splits = make_group_kfold_splits(X, n_splits=n_splits, seed=seed)
    feature_cols = _detect_feature_cols(X, include_statistical, include_health)

    fold_results: list[FoldResult] = []
    all_fold_metrics: list[dict[str, dict[str, float]]] = []

    # Collect OOF predictions
    oof_proba_parts: list[pd.DataFrame] = []
    oof_rul_parts: list[pd.Series] = []
    oof_raw_parts: list[pd.Series] = []
    oof_idx_parts: list[np.ndarray] = []

    for fold_idx, split in enumerate(splits):
        train_units = split["train_units"]
        val_units = split["val_units"]

        X_tr = X[X["unit_id"].isin(train_units)].copy()
        X_val = X[X["unit_id"].isin(val_units)].copy()
        y_tr = y.loc[X_tr.index]
        y_val = y.loc[X_val.index]

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_splits}: train={len(X_tr)}, val={len(X_val)}")

        # Fit
        t0 = time.time()
        model_copy = _clone_model(model)
        model_copy.fit(X_tr[feature_cols + ["unit_id", "cycle"]], y_tr)
        fit_time = time.time() - t0

        # Predict
        t0 = time.time()
        proba = model_copy.predict_proba(X_val[feature_cols + ["unit_id", "cycle"]])
        rul = model_copy.predict_rul(X_val[feature_cols + ["unit_id", "cycle"]])
        raw = model_copy.score_raw(X_val[feature_cols + ["unit_id", "cycle"]])
        predict_time = time.time() - t0

        # Metrics
        metrics = compute_metrics(
            y_val, proba,
            y_true_rul=X_val["RUL"] if "RUL" in X_val.columns else None,
            y_pred_rul=rul,
        )
        all_fold_metrics.append(metrics)

        fold_results.append(FoldResult(
            fold=fold_idx,
            train_units=train_units,
            val_units=val_units,
            oof_proba=proba,
            oof_rul=rul,
            oof_raw_score=raw,
            fit_time=fit_time,
            predict_time=predict_time,
            train_size=len(X_tr),
            val_size=len(X_val),
        ))

        oof_proba_parts.append(proba)
        oof_rul_parts.append(rul)
        oof_raw_parts.append(raw)
        oof_idx_parts.append(X_val.index.to_numpy())

        if verbose:
            roc = metrics.get("roc_auc", {})
            pr = metrics.get("pr_auc", {})
            print(f"    ROC-AUC (h30): {roc.get('h30', 0.0):.4f}  "
                  f"PR-AUC (h30): {pr.get('h30', 0.0):.4f}  "
                  f"RUL RMSE: {metrics.get('rul_rmse', {}).get('overall', 0.0):.2f}")

    # Assemble OOF predictions in original row order
    all_idx = np.concatenate(oof_idx_parts)
    oof_proba = pd.concat(oof_proba_parts, ignore_index=False)
    oof_rul = pd.concat(oof_rul_parts, ignore_index=False)
    oof_raw = pd.concat(oof_raw_parts, ignore_index=False)

    # Reindex to original order
    oof_proba = oof_proba.reindex(all_idx).sort_index()
    oof_rul = oof_rul.reindex(all_idx).sort_index()
    oof_raw = oof_raw.reindex(all_idx).sort_index()

    avg_metrics = _merge_metrics(all_fold_metrics)

    return EvalResult(
        model_name=model.name,
        folds=fold_results,
        metrics=avg_metrics,
        oof_proba=oof_proba,
        oof_rul=oof_rul,
        oof_raw_score=oof_raw,
        feature_cols=feature_cols,
        metadata={
            "n_splits": n_splits,
            "seed": seed,
            "total_train_time": sum(f.fit_time for f in fold_results),
            "total_predict_time": sum(f.predict_time for f in fold_results),
        },
    )


# -----------------------------------------------------------------------
# Test prediction
# -----------------------------------------------------------------------


def predict_test(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    include_statistical: bool = True,
    include_health: bool = True,
    verbose: bool = True,
) -> ModelResult:
    """Refit on full training data and predict on test set.

    Parameters
    ----------
    model:
        Model instance.
    X_train:
        Full training feature matrix.
    y_train:
        Full training labels.
    X_test:
        Test feature matrix.
    include_statistical:
        Include statistical features.
    include_health:
        Include health features.
    verbose:
        Print progress.

    Returns
    -------
    ModelResult
        Test-set predictions.
    """
    feature_cols = _detect_feature_cols(X_train, include_statistical, include_health)

    if verbose:
        print(f"Fitting {model.name} on full training data ({len(X_train)} rows)...")

    t0 = time.time()
    model_copy = _clone_model(model)
    model_copy.fit(
        X_train[feature_cols + ["unit_id", "cycle"]],
        y_train,
    )
    fit_time = time.time() - t0

    if verbose:
        print(f"  Fit time: {fit_time:.1f}s")

    t0 = time.time()
    proba = model_copy.predict_proba(X_test[feature_cols + ["unit_id", "cycle"]])
    rul = model_copy.predict_rul(X_test[feature_cols + ["unit_id", "cycle"]])
    raw = model_copy.score_raw(X_test[feature_cols + ["unit_id", "cycle"]])
    predict_time = time.time() - t0

    if verbose:
        print(f"  Predict time: {predict_time:.1f}s")

    return ModelResult(
        proba=proba,
        rul=rul,
        raw_score=raw,
        fold=-1,
        model_name=model.name,
        metadata={
            "fit_time": fit_time,
            "predict_time": predict_time,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_cols": feature_cols,
        },
    )


# -----------------------------------------------------------------------
# Convenience: evaluate a single model with OOF
# -----------------------------------------------------------------------


def evaluate_model(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    include_statistical: bool = True,
    include_health: bool = True,
    verbose: bool = True,
) -> EvalResult:
    """Full evaluation pipeline: OOF + aggregated metrics.

    Parameters
    ----------
    model:
        Model instance.
    X:
        Training feature matrix.
    y:
        Training labels.
    n_splits:
        Number of folds.
    seed:
        Random seed.
    include_statistical:
        Include statistical features.
    include_health:
        Include health features.
    verbose:
        Print progress.

    Returns
    -------
    EvalResult
        Full evaluation result.
    """
    result = run_oof(
        model,
        X, y,
        n_splits=n_splits,
        seed=seed,
        include_statistical=include_statistical,
        include_health=include_health,
        verbose=verbose,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Model: {result.model_name}")
        print(f"{'=' * 60}")
        for metric, hv in result.metrics.items():
            print(f"  {metric}:")
            for key, val in sorted(hv.items()):
                print(f"    {key}: {val:.4f}")
        print(f"  Total fit time: {result.metadata['total_train_time']:.1f}s")
        print(f"  Total predict time: {result.metadata['total_predict_time']:.1f}s")

    return result


# -----------------------------------------------------------------------
# Model cloning helper
# -----------------------------------------------------------------------


def _clone_model(model: BaseModel) -> BaseModel:
    """Create a fresh copy of a model with the same hyperparameters."""
    import copy
    return copy.deepcopy(model)
