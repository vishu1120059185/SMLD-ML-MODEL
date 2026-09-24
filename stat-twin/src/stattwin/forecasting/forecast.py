"""Probability-curve interpolation, RUL profiling, and consistency checks.

Given a fitted model's ``ModelResult``, this module produces:

* A smooth, monotone probability curve **P(fail by t+h)** for horizons
  {10, 20, 30, 40, 50} using PCHIP (Piecewise Cubic Hermite Interpolating
  Polynomial) interpolation.
* A point RUL estimate (clipped ≥ 0) and its associated failure-time
  distribution.
* A predicted failure point (t_now + RUL_hat) with an uncertainty interval
  derived from the conformal or ensemble spread.
* A consistency check comparing the probability curve against the RUL
  distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for

__all__ = [
    "ForecastProfile",
    "ConsistencyReport",
    "build_probability_curve",
    "build_rul_profile",
    "check_consistency",
    "predicted_failure_point",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ForecastProfile:
    """Result of building a probability curve for a single observation.

    Attributes
    ----------
    horizons:
        The discrete horizon values used (e.g. [10, 20, 30, 40, 50]).
    proba_at_horizons:
        Raw predicted probabilities at each discrete horizon.
    interp_x:
        Dense x-axis (cyclical offsets) for the interpolated curve.
    interp_y:
        Smooth PCHIP-interpolated probabilities at ``interp_x``.
    rul_point:
        Point RUL estimate from the model (clipped ≥ 0).
    failure_cycle_hat:
        ``t_now + rul_point`` – predicted absolute failure cycle.
    interval:
        ``(lower, upper)`` bounds around ``failure_cycle_hat`` derived from
        the conformal or ensemble spread.  May be *None* if not provided.
    """

    horizons: np.ndarray
    proba_at_horizons: np.ndarray
    interp_x: np.ndarray
    interp_y: np.ndarray
    rul_point: float
    failure_cycle_hat: float
    interval: tuple[float, float] | None = None
    t_now: float = 0.0


@dataclass
class ConsistencyReport:
    """Diagnostic comparing the probability curve with the RUL distribution.

    Attributes
    ----------
    spearman_rho:
        Spearman rank correlation between horizon values and predicted
        probabilities (should be ≥ 0 for a monotonically increasing
        failure curve).
    kendall_tau:
        Kendall tau-b between horizons and probabilities.
    monotonicity_violations:
        Number of adjacent horizon-pairs where probability *decreases*
        as the horizon grows.
    rul_at_50_pct:
        The interpolated cycle offset at which P(fail) first reaches 0.50.
    failure_cycle_at_50_pct:
        ``t_now + rul_at_50_pct`` – the predicted median failure time.
    is_consistent:
        ``True`` if Spearman rho ≥ 0.80 and monotonicity_violations == 0.
    """

    spearman_rho: float
    kendall_tau: float
    monotonicity_violations: int
    rul_at_50_pct: float | None
    failure_cycle_at_50_pct: float | None
    is_consistent: bool


# ---------------------------------------------------------------------------
# Probability-curve builder
# ---------------------------------------------------------------------------

def build_probability_curve(
    proba_row: pd.Series,
    horizons: list[int] | None = None,
    n_interp_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth monotone probability curve from per-horizon predictions.

    Parameters
    ----------
    proba_row:
        A single row of predicted probabilities with columns ``fail_h{h}``
        for each horizon *h*.  Typically ``result.proba.iloc[i]``.
    horizons:
        Horizon values.  Defaults to ``FAILURE_HORIZONS``.
    n_interp_points:
        Number of points in the dense interpolated curve.

    Returns
    -------
    horizons_arr : np.ndarray
        The discrete horizon values (float).
    proba_arr : np.ndarray
        Raw probabilities at those horizons.
    interp_x : np.ndarray
        Dense x-axis for the PCHIP curve.
    interp_y : np.ndarray
        Interpolated probabilities (clipped to [0, 1]).
    """
    if horizons is None:
        horizons = FAILURE_HORIZONS

    cols = [label_col_for(h) for h in horizons]
    missing = [c for c in cols if c not in proba_row.index]
    if missing:
        raise ValueError(f"Missing horizon columns in proba_row: {missing}")

    horizons_arr = np.asarray(horizons, dtype=float)
    proba_arr = np.array([proba_row[c] for c in cols], dtype=float)

    # PCHIP preserves monotonicity between knots when the data is monotone
    pchip = PchipInterpolator(horizons_arr, proba_arr, extrapolate=True)
    interp_x = np.linspace(horizons_arr.min(), horizons_arr.max(), n_interp_points)
    interp_y = np.clip(pchip(interp_x), 0.0, 1.0)

    return horizons_arr, proba_arr, interp_x, interp_y


# ---------------------------------------------------------------------------
# RUL profile builder
# ---------------------------------------------------------------------------

def build_rul_profile(
    rul_value: float,
    t_now: float = 0.0,
    sigma_ens: float | None = None,
    conformal_q: float | None = None,
) -> tuple[float, float, tuple[float, float] | None]:
    """Derive RUL point estimate and predicted failure cycle with interval.

    Parameters
    ----------
    rul_value:
        Raw point RUL prediction from the model.
    t_now:
        Current cycle number.
    sigma_ens:
        Ensemble standard deviation (for a +/- interval).  Ignored when
        *conformal_q* is provided.
    conformal_q:
        Conformal quantile multiplier.  The interval becomes
        ``[rul - q * sigma, rul + q * sigma]``.

    Returns
    -------
    rul_clipped:
        Non-negative clipped RUL estimate.
    failure_hat:
        ``t_now + rul_clipped``.
    interval:
        ``(lower_cycle, upper_cycle)`` or *None*.
    """
    rul_clipped = max(float(rul_value), 0.0)
    failure_hat = t_now + rul_clipped

    interval: tuple[float, float] | None = None
    if conformal_q is not None and sigma_ens is not None:
        spread = conformal_q * (sigma_ens + 1e-12)
        lower = max(t_now + rul_clipped - spread, t_now)
        upper = t_now + rul_clipped + spread
        interval = (lower, upper)
    elif sigma_ens is not None:
        lower = max(t_now + rul_clipped - 2 * sigma_ens, t_now)
        upper = t_now + rul_clipped + 2 * sigma_ens
        interval = (lower, upper)

    return rul_clipped, failure_hat, interval


# ---------------------------------------------------------------------------
# Convenience: predicted failure point
# ---------------------------------------------------------------------------

def predicted_failure_point(
    rul_series: pd.Series,
    cycle_series: pd.Series,
    sigma_ens: pd.Series | None = None,
    conformal_q: float | None = None,
) -> pd.DataFrame:
    """Compute predicted failure point for every row in a dataset.

    Parameters
    ----------
    rul_series:
        Point RUL predictions (index aligned with *cycle_series*).
    cycle_series:
        Current cycle numbers for each observation.
    sigma_ens:
        Per-row ensemble standard deviation.
    conformal_q:
        Conformal quantile multiplier.

    Returns
    -------
    pd.DataFrame
        Columns: ``rul_hat``, ``failure_hat``, ``interval_lower``,
        ``interval_upper``.
    """
    rul_arr = np.maximum(rul_series.values.astype(float), 0.0)
    cycle_arr = cycle_series.values.astype(float)
    failure_arr = cycle_arr + rul_arr

    lower = np.full_like(failure_arr, np.nan)
    upper = np.full_like(failure_arr, np.nan)

    if sigma_ens is not None:
        sig = sigma_ens.values.astype(float)
        q = conformal_q if conformal_q is not None else 2.0
        spread = q * (sig + 1e-12)
        lower = np.maximum(cycle_arr + rul_arr - spread, cycle_arr)
        upper = cycle_arr + rul_arr + spread

    return pd.DataFrame(
        {
            "rul_hat": rul_arr,
            "failure_hat": failure_arr,
            "interval_lower": lower,
            "interval_upper": upper,
        },
        index=rul_series.index,
    )


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def check_consistency(
    proba_row: pd.Series,
    rul_point: float,
    t_now: float = 0.0,
    horizons: list[int] | None = None,
    n_interp_points: int = 200,
) -> ConsistencyReport:
    """Compare the probability curve with the RUL point estimate.

    The check verifies that:
    1. Predicted failure probabilities are **non-decreasing** in horizon.
    2. The Spearman correlation between horizons and probabilities is
       high (≥ 0.80).
    3. The 50 % crossing point of the interpolated curve is within a
       reasonable range of the RUL point estimate.

    Parameters
    ----------
    proba_row:
        A single row of per-horizon failure probabilities.
    rul_point:
        Point RUL estimate from the model.
    t_now:
        Current cycle number.
    horizons:
        Horizon values (defaults to ``FAILURE_HORIZONS``).
    n_interp_points:
        Resolution of the PCHIP interpolation.

    Returns
    -------
    ConsistencyReport
    """
    if horizons is None:
        horizons = FAILURE_HORIZONS

    horizons_arr, proba_arr, interp_x, interp_y = build_probability_curve(
        proba_row, horizons, n_interp_points
    )

    # --- Spearman & Kendall ---
    from scipy.stats import kendalltau, spearmanr

    spearman_rho, _ = spearmanr(horizons_arr, proba_arr)
    kendall_tau, _ = kendalltau(horizons_arr, proba_arr)

    # --- Monotonicity violations ---
    diffs = np.diff(proba_arr)
    n_violations = int(np.sum(diffs < -1e-12))

    # --- 50 % crossing point ---
    rul_at_50: float | None = None
    failure_at_50: float | None = None
    above_50 = interp_y >= 0.50
    if above_50.any():
        # first x where curve >= 0.50
        idx = np.argmax(above_50)
        if idx > 0:
            # linear interpolation between adjacent points for sub-sample accuracy
            x0, x1 = interp_x[idx - 1], interp_x[idx]
            y0, y1 = interp_y[idx - 1], interp_y[idx]
            rul_at_50 = float(x0 + (0.50 - y0) * (x1 - x0) / (y1 - y0 + 1e-12))
        else:
            rul_at_50 = float(interp_x[0])
        failure_at_50 = t_now + rul_at_50

    # --- Consensus ---
    is_consistent = (spearman_rho >= 0.80) and (n_violations == 0)

    return ConsistencyReport(
        spearman_rho=float(spearman_rho),
        kendall_tau=float(kendall_tau),
        monotonicity_violations=n_violations,
        rul_at_50_pct=rul_at_50,
        failure_cycle_at_50_pct=failure_at_50,
        is_consistent=is_consistent,
    )
