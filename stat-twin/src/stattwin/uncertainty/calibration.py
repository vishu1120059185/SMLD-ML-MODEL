"""Probability calibration via isotonic regression and Platt scaling.

This module provides:

* **Isotonic regression** – non-parametric, monotone mapping from raw
  scores to calibrated probabilities.
* **Platt scaling** – logistic / sigmoid parametric calibration.
* **Calibration diagnostics** – Brier score, Expected Calibration Error
  (ECE) with both equal-width and equal-mass binning, and reliability
  diagram data (before and after calibration).

All calibrators operate per-horizon and accept out-of-fold (OOF)
predictions to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = [
    "CalibrationReport",
    "calibrate_isotonic",
    "calibrate_platt",
    "brier_score",
    "ece_equal_width",
    "ece_equal_mass",
    "reliability_diagram_data",
]


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Results of calibrating and evaluating a set of predictions.

    Attributes
    ----------
    method:
        ``"isotonic"`` or ``"platt"``.
    horizon:
        The horizon being calibrated.
    brier_before:
        Brier score before calibration.
    brier_after:
        Brier score after calibration.
    ece_before_ew:
        ECE (equal-width bins) before calibration.
    ece_after_ew:
        ECE (equal-width bins) after calibration.
    ece_before_em:
        ECE (equal-mass bins) before calibration.
    ece_after_em:
        ECE (equal-mass bins) after calibration.
    reliability_before:
        Reliability diagram data before calibration (bin centres, mean
        predicted, mean observed).
    reliability_after:
        Reliability diagram data after calibration.
    calibrator:
        The fitted calibrator object (``IsotonicRegression`` or
        ``LogisticRegression``).
    """

    method: str
    horizon: int
    brier_before: float = np.nan
    brier_after: float = np.nan
    ece_before_ew: float = np.nan
    ece_after_ew: float = np.nan
    ece_before_em: float = np.nan
    ece_after_em: float = np.nan
    reliability_before: pd.DataFrame | None = None
    reliability_after: pd.DataFrame | None = None
    calibrator: Any = None


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score (lower is better).

    Parameters
    ----------
    y_true:
        Binary true labels (0 or 1).
    y_prob:
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Mean squared prediction error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# ECE – equal-width bins
# ---------------------------------------------------------------------------

def ece_equal_width(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins.

    Parameters
    ----------
    y_true:
        Binary true labels.
    y_prob:
        Predicted probabilities.
    n_bins:
        Number of bins spanning [0, 1].

    Returns
    -------
    float
        ECE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = mask | (y_prob == hi)  # include right edge of last bin
        if mask.sum() == 0:
            continue
        frac = mask.sum() / n
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += frac * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# ECE – equal-mass (adaptive) bins
# ---------------------------------------------------------------------------

def ece_equal_mass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-mass (quantile) bins.

    Parameters
    ----------
    y_true:
        Binary true labels.
    y_prob:
        Predicted probabilities.
    n_bins:
        Number of bins (each containing roughly 1/n_bins of the data).

    Returns
    -------
    float
        ECE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n < n_bins:
        n_bins = max(1, n)

    sorted_idx = np.argsort(y_prob)
    sorted_prob = y_prob[sorted_idx]
    sorted_true = y_true[sorted_idx]

    bin_size = n // n_bins
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        if start >= n:
            break
        prob_bin = sorted_prob[start:end]
        true_bin = sorted_true[start:end]
        if len(prob_bin) == 0:
            continue
        frac = len(prob_bin) / n
        acc = true_bin.mean()
        conf = prob_bin.mean()
        ece += frac * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute bin-level statistics for a reliability diagram.

    Parameters
    ----------
    y_true:
        Binary true labels.
    y_prob:
        Predicted probabilities.
    n_bins:
        Number of equal-width bins.

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_center``, ``mean_predicted``, ``mean_observed``,
        ``count``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    records = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = mask | (y_prob == hi)
        if mask.sum() == 0:
            records.append(
                {
                    "bin_center": (lo + hi) / 2,
                    "mean_predicted": np.nan,
                    "mean_observed": np.nan,
                    "count": 0,
                }
            )
        else:
            records.append(
                {
                    "bin_center": (lo + hi) / 2,
                    "mean_predicted": float(y_prob[mask].mean()),
                    "mean_observed": float(y_true[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Isotonic regression calibrator
# ---------------------------------------------------------------------------

def calibrate_isotonic(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_test_raw: np.ndarray | None = None,
    horizon: int = 0,
) -> CalibrationReport:
    """Fit an isotonic-regression calibrator and evaluate on OOF / test data.

    Parameters
    ----------
    y_true:
        Binary labels on the calibration set.
    y_raw:
        Raw (uncalibrated) probabilities on the calibration set.
    y_test_raw:
        Optional raw probabilities on a separate test set.  If *None*,
        calibration metrics are computed on the calibration set only.
    horizon:
        Horizon identifier for labelling the report.

    Returns
    -------
    CalibrationReport
    """
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(y_raw, y_true)

    y_test_before = y_true if y_test_raw is None else y_test_raw
    y_test_after = ir.predict(y_test_before)

    report = CalibrationReport(
        method="isotonic",
        horizon=horizon,
        brier_before=brier_score(y_true, y_test_before),
        brier_after=brier_score(y_true, y_test_after),
        ece_before_ew=ece_equal_width(y_true, y_test_before),
        ece_after_ew=ece_equal_width(y_true, y_test_after),
        ece_before_em=ece_equal_mass(y_true, y_test_before),
        ece_after_em=ece_equal_mass(y_true, y_test_after),
        reliability_before=reliability_diagram_data(y_true, y_test_before),
        reliability_after=reliability_diagram_data(y_true, y_test_after),
        calibrator=ir,
    )
    return report


# ---------------------------------------------------------------------------
# Platt scaling calibrator
# ---------------------------------------------------------------------------

def calibrate_platt(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_test_raw: np.ndarray | None = None,
    horizon: int = 0,
) -> CalibrationReport:
    """Fit a Platt-scaling calibrator and evaluate on OOF / test data.

    Platt scaling fits a logistic regression on logit-transformed raw
    probabilities.

    Parameters
    ----------
    y_true:
        Binary labels on the calibration set.
    y_raw:
        Raw (uncalibrated) probabilities on the calibration set.
    y_test_raw:
        Optional raw probabilities on a separate test set.
    horizon:
        Horizon identifier for labelling the report.

    Returns
    -------
    CalibrationReport
    """
    # Transform to log-odds space with clamping to avoid ±inf
    eps = 1e-7
    y_raw_clamped = np.clip(y_raw, eps, 1 - eps)
    log_odds = np.log(y_raw_clamped / (1 - y_raw_clamped)).reshape(-1, 1)

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds, y_true)

    y_test_input = y_test_raw if y_test_raw is not None else y_raw

    def _apply_platt(arr: np.ndarray) -> np.ndarray:
        arr_clamped = np.clip(arr, eps, 1 - eps)
        lo = np.log(arr_clamped / (1 - arr_clamped)).reshape(-1, 1)
        return lr.predict_proba(lo)[:, 1]

    y_test_before = y_test_input
    y_test_after = _apply_platt(y_test_input)

    report = CalibrationReport(
        method="platt",
        horizon=horizon,
        brier_before=brier_score(y_true, y_test_before),
        brier_after=brier_score(y_true, y_test_after),
        ece_before_ew=ece_equal_width(y_true, y_test_before),
        ece_after_ew=ece_equal_width(y_true, y_test_after),
        ece_before_em=ece_equal_mass(y_true, y_test_before),
        ece_after_em=ece_equal_mass(y_true, y_test_after),
        reliability_before=reliability_diagram_data(y_true, y_test_before),
        reliability_after=reliability_diagram_data(y_true, y_test_after),
        calibrator=lr,
    )
    return report
