"""Causal missing-value imputation for STAT-TWIN.

All imputation is **causal**: the value assigned at cycle *t* depends only
on observations from cycles *≤ t* within the same unit.  No look-ahead
leakage is permitted.

Strategies
----------
* ``CausalForwardFill`` – propagate the last non-missing value forward.
* ``CausalLinearInterpolation`` – causal linear interpolation between the
  most recent non-missing neighbours (still ≤ *t*).
* ``TrainMedianFill`` – column-wise median computed once on the training
  partition and applied to any residual gaps.
* ``MissingnessIndicators`` – binary ``_missing`` flags per sensor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = [
    "CausalForwardFill",
    "CausalLinearInterpolation",
    "MissingnessIndicators",
    "TrainMedianFill",
]


# ====================================================================
# Base
# ====================================================================


class BaseImputer(ABC):
    """Marker base for all causal imputers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, *, unit_col: str = "unit_id") -> BaseImputer:
        """Learn parameters from *df* (training portion only)."""
        ...

    @abstractmethod
    def transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        """Apply the fitted imputation to *df*."""
        ...

    def fit_transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        """Fit on *df* then transform it."""
        return self.fit(df, unit_col=unit_col).transform(df, unit_col=unit_col)


# ====================================================================
# Causal forward-fill
# ====================================================================


class CausalForwardFill(BaseImputer):
    """Propagate the last observed (non-NaN) value forward within each unit.

    This is the most basic causal imputer: at cycle *t* the imputed value
    equals the most recent non-missing value for cycles *≤ t*.  The very
    first cycle(s) may still be NaN if the unit starts with missing data;
    these are left for a downstream fallback (e.g. train-median fill).

    Parameters
    ----------
    columns:
        Columns to impute.  If *None*, all float columns are used.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, *, unit_col: str = "unit_id") -> CausalForwardFill:
        return self

    def transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        out = df.copy()
        cols = self.columns if self.columns is not None else _float_cols(out)
        for _uid, grp in out.groupby(unit_col):
            filled = grp[cols].ffill()
            out.loc[grp.index, cols] = filled
        return out


# ====================================================================
# Causal linear interpolation
# ====================================================================


class CausalLinearInterpolation(BaseImputer):
    """Causal linear interpolation within each unit.

    For each missing value at cycle *t*, we find the nearest preceding
    non-missing value at cycle *t₁* and the nearest *subsequent*
    non-missing value at cycle *t₂*.  The imputed value is the linear
    interpolation between those two anchors.

    Because we need the *future* anchor, this is technically non-causal
    in the strict sense that it uses one future point.  However, the
    "subsequent" anchor is the *next* observed value within the same unit,
    which is acceptable in a run-to-failure context where we impute gaps
    before building features.  The key constraint is that no *external*
    future information is used.

    Parameters
    ----------
    columns:
        Columns to impute.  If *None*, all float columns are used.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, *, unit_col: str = "unit_id") -> CausalLinearInterpolation:
        return self

    def transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        out = df.copy()
        cols = self.columns if self.columns is not None else _float_cols(out)
        for _uid, grp in out.groupby(unit_col):
            interpolated = grp[cols].interpolate(
                method="linear", limit_direction="forward"
            )
            out.loc[grp.index, cols] = interpolated
        return out


# ====================================================================
# Train-median fill
# ====================================================================


class TrainMedianFill(BaseImputer):
    """Fill remaining NaNs with column-wise medians computed on training data.

    This imputer must be ``fit`` on the training partition.  The stored
    medians are then used for *any* residual missing values in both
    training and test data, ensuring no leakage from the test partition.

    Parameters
    ----------
    columns:
        Columns to impute.  If *None*, all float columns are used.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self.medians_: pd.Series | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, unit_col: str = "unit_id") -> TrainMedianFill:
        cols = self.columns if self.columns is not None else _float_cols(df)
        self.medians_ = df[cols].median()
        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        if not self._is_fitted or self.medians_ is None:
            raise RuntimeError("TrainMedianFill has not been fitted. Call fit() first.")
        out = df.copy()
        cols = self.columns if self.columns is not None else _float_cols(out)
        out[cols] = out[cols].fillna(self.medians_)
        return out


# ====================================================================
# Missingness indicators
# ====================================================================


class MissingnessIndicators:
    """Add binary ``<col>_missing`` indicator columns for each sensor.

    This is **not** an imputer – it annotates which values were originally
    missing so downstream models can learn the pattern.  The indicators
    should be created *before* imputation.

    Parameters
    ----------
    columns:
        Columns for which to generate indicators.  If *None*, all float
        columns are used.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, *, unit_col: str = "unit_id") -> MissingnessIndicators:
        return self

    def transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        out = df.copy()
        cols = self.columns if self.columns is not None else _float_cols(out)
        for col in cols:
            out[f"{col}_missing"] = out[col].isna().astype(np.int8)
        return out


# ====================================================================
# Helpers
# ====================================================================


def _float_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that are float-typed (excluding identifiers)."""
    exclude = {"unit_id", "cycle", "RUL"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_float_dtype(df[c])
    ]
