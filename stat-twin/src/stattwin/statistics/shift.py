"""Distribution shift detection for STAT-TWIN.

Measures how much a unit's current distribution has deviated from its
healthy baseline.  Three metrics are provided:

* **KS statistic** – maximum CDF distance between baseline and current.
* **Wasserstein-1 distance** – earth-mover's distance between the two
  empirical distributions.
* **Population Stability Index (PSI)** – commonly used in model
  monitoring; here applied to sensor distributions.

All computations are causal (baseline is fixed from the first K cycles,
current window uses only cycles *≤ t*).  Each function is vectorised
over units via ``pandas.groupby().rolling()`` or batch SciPy calls.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

__all__ = [
    "DistributionShift",
    "ks_statistic",
    "wasserstein_distance",
    "population_stability_index",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ------------------------------------------------------------------
# KS statistic
# ------------------------------------------------------------------


def ks_statistic(
    df: pd.DataFrame,
    column: str,
    window: int,
    baseline_cycles: int = 30,
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Rolling KS statistic against the unit's healthy baseline.

    The baseline empirical CDF is computed from the first
    *baseline_cycles* cycles of each unit.  For every subsequent cycle
    *t*, the KS statistic is computed between the baseline sample and
    the current rolling window of size *window*.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Sensor column to test.
    window:
        Size of the rolling window for the "current" distribution.
    baseline_cycles:
        Number of initial cycles defining the healthy baseline.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``shift_ks_{col}_w{window}`` column added.
    """
    out = df.copy()
    col_name = f"shift_ks_{column}_w{window}"

    grp = out.sort_values([unit_col, _CYCLE_COL]).groupby(unit_col)

    def _ks_causal(s: pd.Series) -> pd.Series:
        vals = s.values.astype(np.float64)
        n = len(vals)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            end = i + 1
            # Baseline: first baseline_cycles values
            bl_end = min(baseline_cycles, end)
            baseline = vals[:bl_end]
            # Current window
            win_start = max(0, i - window + 1)
            current = vals[win_start:end]

            if len(baseline) < 2 or len(current) < 2:
                continue

            ks_stat, _ = sp_stats.ks_2samp(baseline, current)
            result[i] = ks_stat

        return pd.Series(result, index=s.index)

    out[col_name] = grp[column].transform(_ks_causal)
    return out


# ------------------------------------------------------------------
# Wasserstein-1 distance
# ------------------------------------------------------------------


def wasserstein_distance(
    df: pd.DataFrame,
    column: str,
    window: int,
    baseline_cycles: int = 30,
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Rolling Wasserstein-1 (earth-mover's) distance vs baseline.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Sensor column.
    window:
        Rolling window size for current distribution.
    baseline_cycles:
        Baseline period length.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``shift_wasserstein_{col}_w{window}`` column.
    """
    out = df.copy()
    col_name = f"shift_wasserstein_{column}_w{window}"

    grp = out.sort_values([unit_col, _CYCLE_COL]).groupby(unit_col)

    def _wasserstein_causal(s: pd.Series) -> pd.Series:
        vals = s.values.astype(np.float64)
        n = len(vals)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            end = i + 1
            bl_end = min(baseline_cycles, end)
            baseline = vals[:bl_end]
            win_start = max(0, i - window + 1)
            current = vals[win_start:end]

            if len(baseline) < 2 or len(current) < 2:
                continue

            result[i] = sp_stats.wasserstein_distance(baseline, current)

        return pd.Series(result, index=s.index)

    out[col_name] = grp[column].transform(_wasserstein_causal)
    return out


# ------------------------------------------------------------------
# Population Stability Index (PSI)
# ------------------------------------------------------------------


def population_stability_index(
    df: pd.DataFrame,
    column: str,
    window: int,
    baseline_cycles: int = 30,
    n_bins: int = 10,
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Rolling Population Stability Index vs healthy baseline.

    PSI = Σ (P_current − P_baseline) × ln(P_current / P_baseline)

    Bins are defined by quantiles of the baseline distribution to
    ensure robustness.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Sensor column.
    window:
        Rolling window size for current distribution.
    baseline_cycles:
        Baseline period length.
    n_bins:
        Number of bins for the discretised distributions.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``shift_psi_{col}_w{window}`` column.
    """
    out = df.copy()
    col_name = f"shift_psi_{column}_w{window}"

    grp = out.sort_values([unit_col, _CYCLE_COL]).groupby(unit_col)

    def _psi_causal(s: pd.Series) -> pd.Series:
        vals = s.values.astype(np.float64)
        n = len(vals)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            end = i + 1
            bl_end = min(baseline_cycles, end)
            baseline = vals[:bl_end]
            win_start = max(0, i - window + 1)
            current = vals[win_start:end]

            if len(baseline) < n_bins or len(current) < 2:
                continue

            # Quantile-based bins from baseline
            try:
                bin_edges = np.quantile(
                    baseline, np.linspace(0, 1, n_bins + 1)
                )
            except Exception:
                continue

            # Ensure unique edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                continue

            # Digitise both distributions
            bl_dig = np.digitize(baseline, bin_edges[1:-1], right=True)
            cur_dig = np.digitize(current, bin_edges[1:-1], right=True)

            n_bins_actual = len(bin_edges) - 1
            bl_counts = np.bincount(bl_dig, minlength=n_bins_actual).astype(np.float64)
            cur_counts = np.bincount(cur_dig, minlength=n_bins_actual).astype(np.float64)

            # Proportions with Laplace smoothing to avoid log(0)
            eps = 1e-6
            bl_prop = (bl_counts + eps) / (bl_counts.sum() + eps * n_bins_actual)
            cur_prop = (cur_counts + eps) / (cur_counts.sum() + eps * n_bins_actual)

            psi = float(np.sum((cur_prop - bl_prop) * np.log(cur_prop / bl_prop)))
            result[i] = psi

        return pd.Series(result, index=s.index)

    out[col_name] = grp[column].transform(_psi_causal)
    return out


# ------------------------------------------------------------------
# High-level wrapper class
# ------------------------------------------------------------------


class DistributionShift:
    """Compute distribution-shift features for selected sensors.

    This class wraps the individual shift functions and applies them
    to a configured set of sensors and shift methods.

    Parameters
    ----------
    methods:
        Shift methods to compute.  Subset of ``{"ks", "wasserstein", "psi"}``.
    window:
        Rolling window size for the "current" distribution.
    baseline_cycles:
        Number of initial cycles defining the healthy baseline.
    n_bins:
        Number of bins for PSI computation.
    unit_col:
        Unit identifier column.
    """

    _METHOD_MAP = {
        "ks": ks_statistic,
        "wasserstein": wasserstein_distance,
        "psi": population_stability_index,
    }

    def __init__(
        self,
        methods: Sequence[str] = ("ks", "wasserstein", "psi"),
        window: int = 10,
        baseline_cycles: int = 30,
        n_bins: int = 10,
        unit_col: str = _UNIT_COL,
    ) -> None:
        for m in methods:
            if m not in self._METHOD_MAP:
                raise ValueError(
                    f"Unknown shift method '{m}'; use one of {list(self._METHOD_MAP)}"
                )
        self.methods = list(methods)
        self.window = window
        self.baseline_cycles = baseline_cycles
        self.n_bins = n_bins
        self.unit_col = unit_col

    def compute(
        self,
        df: pd.DataFrame,
        sensor_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute distribution-shift features for all configured sensors.

        Parameters
        ----------
        df:
            Input DataFrame.
        sensor_cols:
            Sensors to compute shift for.  If *None*, all numeric
            columns except identifiers are used.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with shift features appended.
        """
        if sensor_cols is None:
            sensor_cols = _numeric_sensor_cols(df, self.unit_col)

        out = df.copy()
        for method in self.methods:
            fn = self._METHOD_MAP[method]
            for col in sensor_cols:
                out = fn(
                    out,
                    column=col,
                    window=self.window,
                    baseline_cycles=self.baseline_cycles,
                    unit_col=self.unit_col,
                )

        return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _numeric_sensor_cols(
    df: pd.DataFrame, unit_col: str = _UNIT_COL
) -> list[str]:
    """Return numeric columns excluding identifiers."""
    exclude = {unit_col, _CYCLE_COL, "RUL"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
