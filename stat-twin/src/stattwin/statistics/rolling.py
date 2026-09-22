"""Causal rolling statistics for STAT-TWIN.

Every function computes a feature per *(unit_id, cycle)* using only
observations from cycles *≤ t* within each unit.  Implementation is
fully vectorised via ``pandas.groupby().rolling()`` or NumPy.

Functions
---------
* ``rolling_mean`` – simple moving average
* ``rolling_std`` – moving standard deviation
* ``rolling_min`` / ``rolling_max`` / ``rolling_range``
* ``rolling_z_score`` (via ``z_score_vs_baseline``) – z-score of the
  current value against the unit's healthy baseline
* ``rolling_ewma`` / ``rolling_ewma_dev`` – exponentially weighted
  moving average and deviation
* ``rolling_slope`` – closed-form OLS slope over the window
  (units: sigma per 10 cycles)
* ``rolling_pct_change`` / ``rolling_rate_of_change``
* ``rolling_cv`` – coefficient of variation (std / mean)
* ``rolling_skew`` / ``rolling_kurtosis``
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_range",
    "z_score_vs_baseline",
    "rolling_ewma",
    "rolling_ewma_dev",
    "rolling_slope",
    "rolling_pct_change",
    "rolling_rate_of_change",
    "rolling_cv",
    "rolling_skew",
    "rolling_kurtosis",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _causal_groupby(
    df: pd.DataFrame, unit_col: str = _UNIT_COL
) -> pd.core.groupby.DataFrameGroupBy:
    """Return a groupby object sorted by (unit, cycle) for causal rolling."""
    return df.sort_values([unit_col, _CYCLE_COL]).groupby(unit_col)


def _ensure_numeric(series: pd.Series, name: str) -> pd.Series:
    """Coerce *series* to float; raise on failure."""
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{name}' must be numeric, got {series.dtype}")
    return series.astype(np.float64)


# ------------------------------------------------------------------
# Simple rolling statistics
# ------------------------------------------------------------------


def rolling_mean(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Compute causal rolling mean for *column* over each *window*.

    Parameters
    ----------
    df:
        Input DataFrame (must contain *unit_col*, ``cycle``, *column*).
    column:
        Column to aggregate.
    windows:
        Window sizes to compute.
    unit_col:
        Name of the unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with columns ``<col>_rmean_<w>`` added.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_rmean_{w}"] = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).mean()
        )
    return out


def rolling_std(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling standard deviation (ddof=1).

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to aggregate.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_rstd_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_rstd_{w}"] = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).std(ddof=1)
        )
    return out


def rolling_min(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling minimum.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to aggregate.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_rmin_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_rmin_{w}"] = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).min()
        )
    return out


def rolling_max(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling maximum.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to aggregate.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_rmax_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_rmax_{w}"] = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).max()
        )
    return out


def rolling_range(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling range (max − min).

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to aggregate.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_rrange_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        rmax = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).max()
        )
        rmin = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).min()
        )
        out[f"{column}_rrange_{w}"] = rmax - rmin
    return out


# ------------------------------------------------------------------
# Z-score vs healthy baseline
# ------------------------------------------------------------------


def z_score_vs_baseline(
    df: pd.DataFrame,
    column: str,
    baseline_cycles: int = 30,
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Z-score of each value against the unit's healthy baseline.

    The baseline mean and standard deviation are computed from the first
    *baseline_cycles* cycles of each unit.  For the warm-up period
    (cycles 1..baseline_cycles) an expanding estimate is used instead,
    ensuring the feature is causal and always defined.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to z-score.
    baseline_cycles:
        Number of initial cycles defining the healthy baseline.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_zscore`` column added.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)

    def _zscore_causal(s: pd.Series) -> pd.Series:
        """Vectorised causal z-score within one unit group."""
        n = len(s)
        result = np.full(n, np.nan, dtype=np.float64)

        vals = s.values.astype(np.float64)
        for i in range(n):
            # Use expanding baseline up to baseline_cycles, then fixed window
            end = i + 1
            baseline_slice = vals[:end] if end <= baseline_cycles else vals[:baseline_cycles]
            mu = np.nanmean(baseline_slice)
            sigma = np.nanstd(baseline_slice, ddof=1) if len(baseline_slice) > 1 else 1.0
            if sigma == 0 or np.isnan(sigma):
                sigma = 1.0
            result[i] = (vals[i] - mu) / sigma
        return pd.Series(result, index=s.index)

    out[f"{column}_zscore"] = grp[column].transform(_zscore_causal)
    return out


# ------------------------------------------------------------------
# EWMA
# ------------------------------------------------------------------


def rolling_ewma(
    df: pd.DataFrame,
    column: str,
    alphas: Sequence[float] = (0.1, 0.3),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal exponentially weighted moving average.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to smooth.
    alphas:
        Smoothing factors (0 < alpha ≤ 1).  Smaller = more smoothing.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_ewma_<alpha>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for alpha in alphas:
        label = str(alpha).replace(".", "")
        out[f"{column}_ewma_{label}"] = grp[column].transform(
            lambda s, a=alpha: s.ewm(alpha=a, adjust=False, min_periods=1).mean()
        )
    return out


def rolling_ewma_dev(
    df: pd.DataFrame,
    column: str,
    alphas: Sequence[float] = (0.1, 0.3),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal exponentially weighted moving deviation.

    The deviation is the square root of the EWMA of squared deviations
    from the EWMA itself (i.e. EWMA standard deviation).

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to compute deviation for.
    alphas:
        Smoothing factors.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_ewma_dev_<alpha>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for alpha in alphas:
        label = str(alpha).replace(".", "")

        def _ewma_dev(s: pd.Series, a: float = alpha) -> pd.Series:
            var = s.ewm(alpha=a, adjust=False, min_periods=1).var()
            return np.sqrt(var)

        out[f"{column}_ewma_dev_{label}"] = grp[column].transform(_ewma_dev)
    return out


# ------------------------------------------------------------------
# Linear trend / slope
# ------------------------------------------------------------------


def rolling_slope(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
    sigma_per: int = 10,
) -> pd.DataFrame:
    """Causal closed-form OLS slope over each rolling window.

    The slope is normalised by the rolling standard deviation of the
    feature and then scaled to *sigma_per* cycles, so the result is in
    units of ``sigma / {sigma_per} cycles``.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to compute slope for.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.
    sigma_per:
        Number of cycles over which to normalise the slope (default 10).

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_slope_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)

    for w in windows:

        def _slope_w(
            s: pd.Series, win: int = w, sp: int = sigma_per
        ) -> pd.Series:
            """Vectorised OLS slope over a causal rolling window."""
            vals = s.values.astype(np.float64)
            n = len(vals)
            result = np.full(n, np.nan, dtype=np.float64)
            x = np.arange(win, dtype=np.float64)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            if x_var == 0:
                return pd.Series(result, index=s.index)

            for i in range(n):
                start = max(0, i - win + 1)
                end = i + 1
                y_win = vals[start:end]
                actual_len = len(y_win)
                if actual_len < 2:
                    continue
                # Center x for numerical stability
                x_use = np.arange(actual_len, dtype=np.float64)
                x_m = x_use.mean()
                y_m = np.nanmean(y_win)
                num = np.nansum((x_use - x_m) * (y_win - y_m))
                den = np.nansum((x_use - x_m) ** 2)
                if den == 0:
                    continue
                slope_per_cycle = num / den
                # Normalize by rolling std
                roll_std = np.nanstd(y_win, ddof=1) if actual_len > 1 else 1.0
                if roll_std == 0 or np.isnan(roll_std):
                    roll_std = 1.0
                result[i] = (slope_per_cycle / roll_std) * sp
            return pd.Series(result, index=s.index)

        out[f"{column}_slope_{w}"] = grp[column].transform(_slope_w)
    return out


# ------------------------------------------------------------------
# Percentage change & rate of change
# ------------------------------------------------------------------


def rolling_pct_change(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal percentage change: (x_t − x_{t−w}) / |x_{t−w}| × 100.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to compute percentage change for.
    windows:
        Lag sizes (number of cycles back).
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_pctchg_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_pctchg_{w}"] = grp[column].transform(
            lambda s, lw=w: s.pct_change(periods=lw) * 100.0
        )
    return out


def rolling_rate_of_change(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rate of change: (x_t − x_{t−w}) / w.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to compute rate of change for.
    windows:
        Lag sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_roc_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_roc_{w}"] = grp[column].transform(
            lambda s, lw=w: s.diff(lw) / float(lw)
        )
    return out


# ------------------------------------------------------------------
# Coefficient of variation
# ------------------------------------------------------------------


def rolling_cv(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling coefficient of variation: std / |mean|.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to compute CV for.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_cv_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        def _cv(s: pd.Series, win: int = w) -> pd.Series:
            rmean = s.rolling(win, min_periods=win).mean()
            rstd = s.rolling(win, min_periods=win).std(ddof=1)
            return rstd / rmean.replace(0, np.nan)

        out[f"{column}_cv_{w}"] = grp[column].transform(_cv)
    return out


# ------------------------------------------------------------------
# Higher moments
# ------------------------------------------------------------------


def rolling_skew(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling skewness.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_skew_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_skew_{w}"] = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).skew()
        )
    return out


def rolling_kurtosis(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int] = (5, 10, 20, 30),
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Causal rolling excess kurtosis.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column.
    windows:
        Window sizes.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``<col>_kurt_<w>`` columns.
    """
    out = df.copy()
    grp = _causal_groupby(out, unit_col)
    for w in windows:
        out[f"{column}_kurt_{w}"] = grp[column].transform(
            lambda s, lw=w: s.rolling(lw, min_periods=lw).kurt()
        )
    return out
