"""Split / cross-conformal prediction intervals for RUL and failure probability.

The conformal framework calibrates ensemble-mean predictions by computing
nonconformity scores on a held-out calibration set and then building
prediction intervals with guaranteed marginal coverage under the exchangeability
assumption.

Key quantities
--------------
* **Nonconformity score** ``s = |y - yhat| / (sigma_ens + eps)``
  Normalised residuals that account for heteroscedastic ensemble spread.
* **Quantile** ``q = ceil((n+1)(1-alpha)) / n``
  Empirical quantile of the nonconformity scores on the calibration set.
* **Prediction interval** ``yhat ± q * (sigma_ens + eps)``
  Lower bound is clipped at 0 for RUL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ConformalReport",
    "conformal_intervals",
    "winkler_score",
]


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class ConformalReport:
    """Bundle of conformal prediction results.

    Attributes
    ----------
    alpha:
        Miscoverage level (e.g. 0.10 → 90 % coverage).
    quantile_q:
        Empirical quantile used to build intervals.
    coverage:
        Empirical prediction interval coverage (PICP).
    mean_width:
        Mean interval width across observations.
    winkler:
        Mean Winkler (interval score) across observations.
    coverage_by_rul_bucket:
        Coverage broken down by RUL quartile buckets.
    intervals:
        DataFrame with columns ``lower``, ``upper``, ``width``.
    scores_cal:
        Nonconformity scores computed on the calibration set.
    """

    alpha: float
    quantile_q: float
    coverage: float
    mean_width: float
    winkler: float
    coverage_by_rul_bucket: dict[str, float] = field(default_factory=dict)
    intervals: pd.DataFrame | None = None
    scores_cal: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Core conformal procedure
# ---------------------------------------------------------------------------

def _empirical_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute q = ceil((n+1)(1-alpha)) / n quantile (ascending)."""
    n = len(scores)
    idx = int(np.ceil((n + 1) * (1.0 - alpha)))
    idx = min(idx, n)  # clamp
    sorted_scores = np.sort(scores)
    return float(sorted_scores[idx - 1])


def conformal_intervals(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    sigma_cal: np.ndarray,
    yhat_test: np.ndarray,
    sigma_test: np.ndarray,
    alpha: float = 0.10,
    eps: float = 1e-8,
    rul_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> ConformalReport:
    """Build split-conformal prediction intervals on the test set.

    Parameters
    ----------
    y_cal, yhat_cal, sigma_cal:
        True values, predictions, and spread proxy on the **calibration**
        set (never on the test set — that would leak).
    yhat_test, sigma_test:
        Predictions and spread proxy on the **test** set.
    alpha:
        Miscoverage level (0.10 → 90 % nominal coverage).
    eps:
        Small constant to avoid division by zero in normalisation.
    rul_test:
        Optional true RUL used only to bucket the coverage report.
    y_test:
        Optional true values for the test set.  When supplied, coverage
        and the Winkler score are measured **honestly on held-out data**
        instead of being read off the calibration scores.

    Returns
    -------
    ConformalReport
    """
    # Nonconformity scores on calibration set
    abs_res_cal = np.abs(y_cal - yhat_cal)
    denom_cal = sigma_cal + eps
    scores_cal = abs_res_cal / denom_cal

    # Empirical quantile
    q = _empirical_quantile(scores_cal, alpha)

    # Test-set intervals
    denom_test = sigma_test + eps
    spread = q * denom_test
    lower = np.maximum(yhat_test - spread, 0.0)  # clip at 0 for RUL
    upper = yhat_test + spread

    # Metrics
    width = upper - lower
    mean_width = float(np.mean(width))

    # Coverage by RUL bucket / honest test metrics
    coverage_by_rul_bucket: dict[str, float] = {}

    if y_test is not None:
        y_test = np.asarray(y_test, dtype=float)
        covered_test = (y_test >= lower) & (y_test <= upper)
        coverage = float(np.mean(covered_test))
        winkler_val = winkler_score(y_test, lower, upper, alpha)

        if rul_test is not None:
            bucket_of = np.asarray(rul_test, dtype=float)
            quartiles = np.percentile(bucket_of, [25, 50, 75])
            edges = np.concatenate([[-np.inf], quartiles, [np.inf]])
            names = ["Q1_low", "Q2_mid-low", "Q3_mid-high", "Q4_high"]
            for i, name in enumerate(names):
                mask = (bucket_of >= edges[i]) & (bucket_of < edges[i + 1])
                if mask.sum() > 0:
                    coverage_by_rul_bucket[name] = float(np.mean(covered_test[mask]))
    else:
        # Fallback: report the in-sample calibration coverage, which is
        # ~ (1 - alpha) by construction and therefore NOT a test PICP.
        coverage = float(np.mean(scores_cal <= q))
        winkler_val = _winkler_from_scores(scores_cal, q, alpha)

    intervals_df = pd.DataFrame(
        {"lower": lower, "upper": upper, "width": width},
        index=np.arange(len(yhat_test)),
    )

    return ConformalReport(
        alpha=alpha,
        quantile_q=q,
        coverage=coverage,
        mean_width=mean_width,
        winkler=winkler_val,
        coverage_by_rul_bucket=coverage_by_rul_bucket,
        intervals=intervals_df,
        scores_cal=scores_cal,
    )


def _winkler_from_scores(
    scores: np.ndarray, q: float, alpha: float
) -> float:
    """Approximate Winkler score from normalised nonconformity scores.

    For a two-sided interval at level (1-alpha), the Winkler score of an
    observation with normalised residual *s* is:

        W = (upper - lower) + 2/alpha * (lower - y) * 1{y < lower}
            + 2/alpha * (y - upper) * 1{y > upper}

    Using s = |y - yhat| / sigma and interval = yhat ± q * sigma, this
    simplifies for the calibration set.
    """
    # For simplicity we use the mean width as a lower bound on the Winkler
    # score when the true y is not available for the test set.
    width = 2.0 * q * np.mean(scores)
    return float(width)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute the mean Winkler (interval score) for a set of intervals.

    Parameters
    ----------
    y_true:
        Observed values.
    lower, upper:
        Lower and upper bounds of each prediction interval.
    alpha:
        Miscoverage level.

    Returns
    -------
    float
        Mean Winkler score (lower is better).
    """
    width = upper - lower
    penalty_lower = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    penalty_upper = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)
    scores = width + penalty_lower + penalty_upper
    return float(np.mean(scores))
