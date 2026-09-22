"""Comprehensive evaluation metrics for STAT-TWIN.

This module provides:

* **Classification** – per-horizon precision, recall, F1, ROC-AUC, PR-AUC,
  FPR, and FNR.
* **RUL** – MAE, RMSE, and the NASA asymmetric scoring function.
* **Probabilistic** – Brier score and ECE.
* **Interval** – PICP, mean interval width, and Winkler interval score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from stattwin.data.schema import FAILURE_HORIZONS

__all__ = [
    "evaluate_classification",
    "evaluate_rul",
    "probabilistic_metrics",
    "interval_metrics",
]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    """Per-horizon classification metrics.

    Attributes
    ----------
    horizon:
        The prediction horizon.
    precision:
        Positive predictive value.
    recall:
        True positive rate (sensitivity).
    f1:
        Harmonic mean of precision and recall.
    roc_auc:
        Area under the ROC curve.
    pr_auc:
        Area under the Precision-Recall curve.
    fpr:
        False positive rate (1 – specificity).
    fnr:
        False negative rate (1 – recall).
    """

    horizon: int
    precision: float = np.nan
    recall: float = np.nan
    f1: float = np.nan
    roc_auc: float = np.nan
    pr_auc: float = np.nan
    fpr: float = np.nan
    fnr: float = np.nan


def evaluate_classification(
    y_true_dict: Dict[int, np.ndarray],
    y_prob_dict: Dict[int, np.ndarray],
    horizons: Optional[List[int]] = None,
) -> List[ClassificationReport]:
    """Evaluate per-horizon binary classification performance.

    Parameters
    ----------
    y_true_dict:
        Mapping ``horizon → binary true labels`` (0/1).
    y_prob_dict:
        Mapping ``horizon → predicted probabilities`` in [0, 1].
    horizons:
        Subset of horizons to evaluate (defaults to ``FAILURE_HORIZONS``).

    Returns
    -------
    list[ClassificationReport]
        One report per horizon.
    """
    if horizons is None:
        horizons = FAILURE_HORIZONS

    reports: List[ClassificationReport] = []
    for h in horizons:
        if h not in y_true_dict or h not in y_prob_dict:
            reports.append(ClassificationReport(horizon=h))
            continue

        y_true = np.asarray(y_true_dict[h], dtype=int)
        y_prob = np.asarray(y_prob_dict[h], dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        # Handle edge case: all one class
        if len(np.unique(y_true)) < 2:
            reports.append(
                ClassificationReport(
                    horizon=h,
                    precision=precision_score(y_true, y_pred, zero_division=0),
                    recall=recall_score(y_true, y_pred, zero_division=0),
                    f1=f1_score(y_true, y_pred, zero_division=0),
                    roc_auc=np.nan,
                    pr_auc=np.nan,
                    fpr=0.0 if y_true.sum() == 0 else np.nan,
                    fnr=0.0 if (1 - y_true).sum() == 0 else np.nan,
                )
            )
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1_val = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        reports.append(
            ClassificationReport(
                horizon=h,
                precision=float(prec),
                recall=float(rec),
                f1=float(f1_val),
                roc_auc=float(roc),
                pr_auc=float(pr),
                fpr=float(fpr_val),
                fnr=float(fnr_val),
            )
        )
    return reports


# ---------------------------------------------------------------------------
# RUL metrics
# ---------------------------------------------------------------------------

@dataclass
class RULReport:
    """RUL prediction quality metrics.

    Attributes
    ----------
    mae:
        Mean Absolute Error.
    rmse:
        Root Mean Squared Error.
    nasa_score:
        NASA asymmetric scoring function (penalises late predictions more
        than early ones).  Lower is better.
    n:
        Number of observations.
    """

    mae: float = np.nan
    rmse: float = np.nan
    nasa_score: float = np.nan
    n: int = 0


def _nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA asymmetric scoring function.

    S(d) = exp(-d/13) - 1  for d < 0  (early prediction, d = ytrue - ypred)
    S(d) = exp(d/10) - 1   for d >= 0  (late prediction)

    Lower is better (0 when predictions are perfect).
    """
    d = y_true - y_pred  # positive → late, negative → early
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.mean(scores))


def evaluate_rul(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RULReport:
    """Compute RUL prediction quality metrics.

    Parameters
    ----------
    y_true:
        True RUL values.
    y_pred:
        Predicted RUL values (point estimates).

    Returns
    -------
    RULReport
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return RULReport()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nasa = _nasa_score(y_true, y_pred)

    return RULReport(mae=mae, rmse=rmse, nasa_score=nasa, n=n)


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

@dataclass
class ProbabilisticReport:
    """Aggregate probabilistic quality metrics.

    Attributes
    ----------
    mean_brier:
        Mean Brier score across horizons.
    mean_ece_ew:
        Mean ECE (equal-width) across horizons.
    mean_ece_em:
        Mean ECE (equal-mass) across horizons.
    per_horizon:
        Mapping horizon → Brier score.
    """

    mean_brier: float = np.nan
    mean_ece_ew: float = np.nan
    mean_ece_em: float = np.nan
    per_horizon: Dict[int, float] = field(default_factory=dict)


def probabilistic_metrics(
    y_true_dict: Dict[int, np.ndarray],
    y_prob_dict: Dict[int, np.ndarray],
    horizons: Optional[List[int]] = None,
) -> ProbabilisticReport:
    """Compute aggregate Brier and ECE scores across horizons.

    Parameters
    ----------
    y_true_dict:
        Mapping ``horizon → binary true labels``.
    y_prob_dict:
        Mapping ``horizon → predicted probabilities``.
    horizons:
        Horizons to evaluate (defaults to ``FAILURE_HORIZONS``).

    Returns
    -------
    ProbabilisticReport
    """
    from stattwin.uncertainty.calibration import ece_equal_mass, ece_equal_width

    if horizons is None:
        horizons = FAILURE_HORIZONS

    brier_dict: Dict[int, float] = {}
    ece_ew_vals: List[float] = []
    ece_em_vals: List[float] = []

    for h in horizons:
        if h not in y_true_dict or h not in y_prob_dict:
            continue
        yt = np.asarray(y_true_dict[h], dtype=float)
        yp = np.asarray(y_prob_dict[h], dtype=float)
        brier_dict[h] = brier_score_loss(yt, yp)
        ece_ew_vals.append(ece_equal_width(yt, yp))
        ece_em_vals.append(ece_equal_mass(yt, yp))

    brier_vals = list(brier_dict.values())
    return ProbabilisticReport(
        mean_brier=float(np.mean(brier_vals)) if brier_vals else np.nan,
        mean_ece_ew=float(np.mean(ece_ew_vals)) if ece_ew_vals else np.nan,
        mean_ece_em=float(np.mean(ece_em_vals)) if ece_em_vals else np.nan,
        per_horizon=brier_dict,
    )


# ---------------------------------------------------------------------------
# Interval metrics
# ---------------------------------------------------------------------------

@dataclass
class IntervalReport:
    """Prediction interval quality metrics.

    Attributes
    ----------
    picp:
        Prediction Interval Coverage Probability.
    mean_width:
        Mean interval width.
    winkler:
        Mean Winkler (interval score).
    n:
        Number of intervals.
    """

    picp: float = np.nan
    mean_width: float = np.nan
    winkler: float = np.nan
    n: int = 0


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> IntervalReport:
    """Evaluate the quality of prediction intervals.

    Parameters
    ----------
    y_true:
        Observed values.
    lower, upper:
        Lower and upper bounds of each interval.
    alpha:
        Miscoverage level (0.10 → 90 % nominal coverage).

    Returns
    -------
    IntervalReport
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = len(y_true)

    if n == 0:
        return IntervalReport()

    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered))
    mean_w = float(np.mean(upper - lower))

    # Winkler score
    width = upper - lower
    penalty_lower = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    penalty_upper = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)
    winkler = float(np.mean(width + penalty_lower + penalty_upper))

    return IntervalReport(picp=picp, mean_width=mean_w, winkler=winkler, n=n)
