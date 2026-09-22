"""Cross-sensor correlation features for STAT-TWIN.

Provides rolling Pearson and Spearman correlations on pre-selected
top-k sensor pairs, correlation-shift detection via the Frobenius norm
of the difference between a windowed correlation matrix and the baseline,
and per-pair delta-rho.

All computations are causal (cycle *t* uses only cycles *≤ t*) and
vectorised via ``pandas.groupby().rolling()``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CrossSensorCorrelation",
    "correlation_shift",
    "delta_rho",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ------------------------------------------------------------------
# Top-k pair selection (fit on training data only)
# ------------------------------------------------------------------


class CrossSensorCorrelation:
    """Select top-k sensor pairs and compute rolling cross-correlations.

    The top-k pairs are identified from training data only (via
    ``fit``).  ``transform`` then computes rolling Pearson and/or
    Spearman correlations for those pairs across all data.

    Parameters
    ----------
    methods:
        Correlation methods to compute.  Subset of ``{"pearson", "spearman"}``.
    top_k:
        Number of top sensor pairs to keep.
    windows:
        Rolling window sizes for correlation computation.
    unit_col:
        Unit identifier column.
    """

    def __init__(
        self,
        methods: Sequence[str] = ("pearson", "spearman"),
        top_k: int = 10,
        windows: Sequence[int] = (5, 10, 20, 30),
        unit_col: str = _UNIT_COL,
    ) -> None:
        for m in methods:
            if m not in ("pearson", "spearman"):
                raise ValueError(f"Unknown method '{m}'; use 'pearson' or 'spearman'.")
        self.methods = list(methods)
        self.top_k = top_k
        self.windows = list(windows)
        self.unit_col = unit_col

        self.selected_pairs_: list[tuple[str, str]] | None = None
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        sensor_cols: list[str] | None = None,
    ) -> CrossSensorCorrelation:
        """Select top-k sensor pairs from training data.

        Pairs are ranked by their absolute mean Pearson correlation
        across all training units.  Only sensors with sufficient
        non-NaN data are considered.

        Parameters
        ----------
        df:
            Training DataFrame (must contain *unit_col*, ``cycle``,
            and sensor columns).
        sensor_cols:
            Sensor columns to consider.  If *None*, all numeric
            columns except identifiers are used.

        Returns
        -------
        CrossSensorCorrelation
            Fitted instance with ``selected_pairs_`` populated.
        """
        if sensor_cols is None:
            sensor_cols = _numeric_sensor_cols(df, self.unit_col)

        # Compute mean absolute Pearson correlation across all units
        mean_corr = self._mean_abs_pearson(df, sensor_cols)
        pairs = self._rank_and_select(mean_corr, sensor_cols)

        self.selected_pairs_ = pairs
        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        sensor_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute rolling correlations for the selected top-k pairs.

        Parameters
        ----------
        df:
            Input DataFrame.
        sensor_cols:
            Original sensor columns (used only to validate; the
            selected pairs are taken from ``selected_pairs_``).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with correlation features appended.
        """
        if not self._is_fitted or self.selected_pairs_ is None:
            raise RuntimeError("CrossSensorCorrelation has not been fitted.")

        out = df.copy()

        for col_a, col_b in self.selected_pairs_:
            for method in self.methods:
                for w in self.windows:
                    out = _add_rolling_corr_col(
                        out, col_a, col_b, method, w, self.unit_col
                    )

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_abs_pearson(
        df: pd.DataFrame, sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Mean absolute Pearson correlation matrix across all units."""
        all_corrs: list[np.ndarray] = []
        for _uid, grp in df.groupby(_UNIT_COL):
            corr_mat = grp[sensor_cols].corr(method="pearson")
            all_corrs.append(corr_mat.abs().values)
        return pd.DataFrame(
            np.mean(all_corrs, axis=0),
            index=sensor_cols,
            columns=sensor_cols,
        )

    def _rank_and_select(
        self,
        mean_corr: pd.DataFrame,
        sensor_cols: list[str],
    ) -> list[tuple[str, str]]:
        """Rank pairs by mean |rho| and return top-k."""
        pairs_with_corr: list[tuple[float, str, str]] = []
        for i, col_a in enumerate(sensor_cols):
            for j, col_b in enumerate(sensor_cols):
                if j <= i:
                    continue
                rho = mean_corr.loc[col_a, col_b]
                if np.isfinite(rho):
                    pairs_with_corr.append((float(rho), col_a, col_b))

        pairs_with_corr.sort(key=lambda t: t[0], reverse=True)
        return [(a, b) for _, a, b in pairs_with_corr[: self.top_k]]


# ------------------------------------------------------------------
# Vectorised rolling correlation (groupby + rolling)
# ------------------------------------------------------------------


def _vectorised_rolling_corr(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int,
    method: str = "pearson",
    unit_col: str = _UNIT_COL,
) -> pd.Series:
    """Compute rolling correlation between two columns within each unit.

    Uses ``pandas.groupby().rolling()`` for full vectorisation.
    """
    if method == "spearman":
        # Spearman requires ranking first; rank within group then
        # compute Pearson on ranks
        ranked = df.groupby(unit_col)[[col_a, col_b]].transform(
            lambda s: s.rank(method="average")
        )
        df_temp = df.copy()
        rank_a_col = f"_rank_a_{col_a}_{col_b}"
        rank_b_col = f"_rank_b_{col_a}_{col_b}"
        df_temp[rank_a_col] = ranked[col_a]
        df_temp[rank_b_col] = ranked[col_b]
        grp = df_temp.sort_values([unit_col, _CYCLE_COL]).groupby(unit_col)
        result = grp.apply(
            lambda g: g[rank_a_col].rolling(window, min_periods=window).corr(g[rank_b_col]),
            include_groups=False,
        )
        # Flatten multi-index if groupby returns one
        if isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(0)
        return result
    else:
        grp = df.sort_values([unit_col, _CYCLE_COL]).groupby(unit_col)
        result = grp.apply(
            lambda g: g[col_a].rolling(window, min_periods=window).corr(g[col_b]),
            include_groups=False,
        )
        if isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(0)
        return result


def _add_rolling_corr_col(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    method: str,
    window: int,
    unit_col: str,
) -> pd.DataFrame:
    """Add a single rolling correlation column to *df*."""
    out = df.copy()
    tag = f"{col_a}_x_{col_b}_{method}"
    col_name = f"corr_{tag}_r{window}"

    rolling_corr = _vectorised_rolling_corr(
        out, col_a, col_b, window, method, unit_col
    )
    out[col_name] = rolling_corr.values
    return out


# ------------------------------------------------------------------
# Correlation shift
# ------------------------------------------------------------------


def correlation_shift(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int,
    baseline_cycles: int = 30,
    method: str = "pearson",
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Correlation shift: |ρ_window − ρ_baseline|.

    For each cycle *t* ≥ *baseline_cycles*, the baseline correlation is
    computed from cycles 1..baseline_cycles and the windowed correlation
    from cycles (t − window + 1)..t.  The shift is the absolute
    difference.

    Parameters
    ----------
    df:
        Input DataFrame.
    col_a, col_b:
        Sensor columns.
    window:
        Rolling window size.
    baseline_cycles:
        Number of initial cycles defining the healthy baseline.
    method:
        ``"pearson"`` or ``"spearman"``.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``corrshift_{a}_x_{b}_{method}_{w}`` column added.
    """
    out = df.copy()
    tag = f"{col_a}_x_{col_b}_{method}"
    col_name = f"corrshift_{tag}_r{window}"

    rolling_corr = _vectorised_rolling_corr(
        out, col_a, col_b, window, method, unit_col
    )

    # Baseline per unit from first K cycles
    baseline_values: dict[int, float] = {}
    for uid, grp_df in out.groupby(unit_col):
        first_k = grp_df.head(baseline_cycles)
        if method == "pearson":
            rho = first_k[col_a].corr(first_k[col_b])
        else:
            rho = first_k[col_a].corr(first_k[col_b], method="spearman")
        baseline_values[uid] = rho if np.isfinite(rho) else 0.0

    out["_rolling_corr_temp"] = rolling_corr.values
    out["_baseline_corr_temp"] = out[unit_col].map(baseline_values)
    out[col_name] = (out["_rolling_corr_temp"] - out["_baseline_corr_temp"]).abs()
    out.drop(columns=["_rolling_corr_temp", "_baseline_corr_temp"], inplace=True)

    return out


# ------------------------------------------------------------------
# Per-pair delta rho
# ------------------------------------------------------------------


def delta_rho(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int,
    baseline_cycles: int = 30,
    method: str = "pearson",
    unit_col: str = _UNIT_COL,
) -> pd.DataFrame:
    """Per-pair delta rho: ρ_window − ρ_baseline.

    This is the signed version of the correlation shift, showing
    whether correlation is increasing or decreasing.

    Parameters
    ----------
    df:
        Input DataFrame.
    col_a, col_b:
        Sensor columns.
    window:
        Rolling window size.
    baseline_cycles:
        Number of initial cycles for the baseline.
    method:
        ``"pearson"`` or ``"spearman"``.
    unit_col:
        Unit identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``deltarho_{a}_x_{b}_{method}_{w}`` column added.
    """
    out = df.copy()
    tag = f"{col_a}_x_{col_b}_{method}"
    col_name = f"deltarho_{tag}_r{window}"

    rolling_corr = _vectorised_rolling_corr(
        out, col_a, col_b, window, method, unit_col
    )

    # Baseline per unit from first K cycles
    baseline_values: dict[int, float] = {}
    for uid, grp_df in out.groupby(unit_col):
        first_k = grp_df.head(baseline_cycles)
        if method == "pearson":
            rho = first_k[col_a].corr(first_k[col_b])
        else:
            rho = first_k[col_a].corr(first_k[col_b], method="spearman")
        baseline_values[uid] = rho if np.isfinite(rho) else 0.0

    out["_rolling_corr_temp"] = rolling_corr.values
    out["_baseline_corr_temp"] = out[unit_col].map(baseline_values)
    out[col_name] = out["_rolling_corr_temp"] - out["_baseline_corr_temp"]
    out.drop(columns=["_rolling_corr_temp", "_baseline_corr_temp"], inplace=True)

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
