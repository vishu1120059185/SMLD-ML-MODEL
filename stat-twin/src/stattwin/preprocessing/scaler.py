"""Fitted scalers for STAT-TWIN.

All scalers are fit on training data only and applied to both train and
test partitions.  Two modes are supported:

* ``TrainFittedScaler`` – a single set of parameters for all units.
* ``PerConditionScaler`` – separate parameters per operating-condition
  group (e.g. per ``(op_setting_1, op_setting_2)`` regime).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = ["PerConditionScaler", "TrainFittedScaler"]


# ====================================================================
# Base
# ====================================================================


class BaseScaler(ABC):
    """Abstract base for train-fitted scalers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> BaseScaler:
        """Learn scaling parameters from training data."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaling to *df*."""
        ...

    def fit_transform(
        self, df: pd.DataFrame, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        return self.fit(df, columns=columns).transform(df)


# ====================================================================
# Train-fitted scaler (single group)
# ====================================================================


class TrainFittedScaler(BaseScaler):
    """StandardScaler or RobustScaler fitted once on training data.

    Parameters
    ----------
    method:
        ``"standard"`` for zero-mean / unit-variance scaling, or
        ``"robust"`` for median / IQR scaling.
    columns:
        Columns to scale.  If *None*, all numeric columns are used.
    """

    def __init__(self, method: str = "standard", columns: list[str] | None = None) -> None:
        if method not in ("standard", "robust"):
            raise ValueError(f"method must be 'standard' or 'robust', got '{method}'")
        self.method = method
        self.columns = columns
        self.center_: dict[str, float] | None = None
        self.scale_: dict[str, float] | None = None
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> TrainFittedScaler:
        cols = columns if columns is not None else self.columns
        if cols is None:
            cols = _numeric_cols(df)
        self._columns = cols
        self.center_ = {}
        self.scale_ = {}
        for col in cols:
            vals = df[col].dropna().values
            if self.method == "standard":
                self.center_[col] = float(np.mean(vals))
                s = float(np.std(vals))
                self.scale_[col] = s if s > 0 else 1.0
            else:  # robust
                self.center_[col] = float(np.median(vals))
                q1 = float(np.percentile(vals, 25))
                q3 = float(np.percentile(vals, 75))
                iqr = q3 - q1
                self.scale_[col] = iqr if iqr > 0 else 1.0
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("TrainFittedScaler has not been fitted.")
        out = df.copy()
        cols = self._columns or _numeric_cols(out)
        for col in cols:
            out[col] = (out[col] - self.center_[col]) / self.scale_[col]
        return out


# ====================================================================
# Per-condition scaler
# ====================================================================


class PerConditionScaler(BaseScaler):
    """Separate scaling parameters per operating-condition group.

    Operating conditions are identified by a combination of operational
    setting columns.  During ``fit``, the training DataFrame is grouped by
    these columns and per-group statistics are stored.

    Parameters
    ----------
    method:
        ``"standard"`` or ``"robust"``.
    condition_cols:
        Columns that define the operating-condition groups.  Typically
        ``["op_setting_1", "op_setting_2", "op_setting_3"]``.
    columns:
        Feature columns to scale.
    """

    def __init__(
        self,
        method: str = "standard",
        condition_cols: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        if method not in ("standard", "robust"):
            raise ValueError(f"method must be 'standard' or 'robust', got '{method}'")
        self.method = method
        self.condition_cols = condition_cols or ["op_setting_1", "op_setting_2", "op_setting_3"]
        self.columns = columns
        # group_key -> {col: center}, {col: scale}
        self.center_: dict[str, dict[str, float]] | None = None
        self.scale_: dict[str, dict[str, float]] | None = None
        self.fallback_center_: dict[str, float] | None = None
        self.fallback_scale_: dict[str, float] | None = None
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> PerConditionScaler:
        cols = columns if columns is not None else self.columns
        if cols is None:
            cols = _numeric_cols(df)
        self._columns = cols

        # Global fallback for unseen groups
        self.fallback_center_ = {}
        self.fallback_scale_ = {}
        for col in cols:
            vals = df[col].dropna().values
            if self.method == "standard":
                self.fallback_center_[col] = float(np.mean(vals))
                s = float(np.std(vals))
                self.fallback_scale_[col] = s if s > 0 else 1.0
            else:
                self.fallback_center_[col] = float(np.median(vals))
                q1 = float(np.percentile(vals, 25))
                q3 = float(np.percentile(vals, 75))
                iqr = q3 - q1
                self.fallback_scale_[col] = iqr if iqr > 0 else 1.0

        # Per-group parameters
        self.center_ = {}
        self.scale_ = {}
        for group_key, grp in df.groupby(self.condition_cols):
            key = _group_key_to_str(group_key)
            self.center_[key] = {}
            self.scale_[key] = {}
            for col in cols:
                vals = grp[col].dropna().values
                if len(vals) == 0:
                    # fall back to global
                    self.center_[key][col] = self.fallback_center_[col]
                    self.scale_[key][col] = self.fallback_scale_[col]
                    continue
                if self.method == "standard":
                    self.center_[key][col] = float(np.mean(vals))
                    s = float(np.std(vals))
                    self.scale_[key][col] = s if s > 0 else 1.0
                else:
                    self.center_[key][col] = float(np.median(vals))
                    q1 = float(np.percentile(vals, 25))
                    q3 = float(np.percentile(vals, 75))
                    iqr = q3 - q1
                    self.scale_[key][col] = iqr if iqr > 0 else 1.0

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("PerConditionScaler has not been fitted.")
        out = df.copy()
        cols = self._columns or _numeric_cols(out)

        # Vectorised: build a key column then map
        key_col = _group_key_col(df, self.condition_cols)
        unique_keys = key_col.unique()

        for key in unique_keys:
            mask = key_col == key
            params = self.center_.get(key)
            if params is None:
                center = self.fallback_center_
                scale = self.fallback_scale_
            else:
                center = params
                scale = self.scale_[key]
            for col in cols:
                out.loc[mask, col] = (out.loc[mask, col] - center[col]) / scale[col]

        return out


# ====================================================================
# Helpers
# ====================================================================


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"unit_id", "cycle", "RUL"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _group_key_to_str(key) -> str:
    """Convert a pandas groupby key to a hashable string."""
    if isinstance(key, tuple):
        return "|".join(str(v) for v in key)
    return str(key)


def _group_key_col(df: pd.DataFrame, condition_cols: list[str]) -> pd.Series:
    """Build a string key column from condition columns."""
    parts = [df[c].astype(str) for c in condition_cols]
    return parts[0].str.cat(parts[1:], sep="|")
