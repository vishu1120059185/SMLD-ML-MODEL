"""Outlier detection and winsorisation for STAT-TWIN.

All detectors operate per-column and are fit on training data only.

Methods
-------
* ``IQRFencesDetector`` – inter-quartile range fences
  (Q1 − k·IQR, Q3 + k·IQR).
* ``ZScoreDetector`` – Gaussian z-score beyond *k* standard deviations.
* ``RobustZDetector`` – robust z-score using median and MAD.
* ``Winsorizer`` – clip values to training-derived fences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = [
    "IQRFencesDetector",
    "RobustZDetector",
    "Winsorizer",
    "ZScoreDetector",
]


# ====================================================================
# Base
# ====================================================================


class BaseOutlierDetector(ABC):
    """Abstract base for outlier detectors."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> BaseOutlierDetector:
        """Learn thresholds from training data."""
        ...

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a boolean DataFrame of the same shape as *df[columns]*."""
        ...

    def detect_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a per-row boolean Series (True if *any* column is an outlier)."""
        bool_df = self.detect(df)
        return bool_df.any(axis=1)


# ====================================================================
# IQR Fences
# ====================================================================


class IQRFencesDetector(BaseOutlierDetector):
    """Inter-quartile range outlier detector.

    Outlier boundaries are ``Q1 − k·IQR`` and ``Q3 + k·IQR`` where
    ``IQR = Q3 − Q1``.  Both ``Q1`` and ``Q3`` are learned from the
    training partition.

    Parameters
    ----------
    k:
        Multiplier for the IQR (default 1.5 is Tukey's convention).
    """

    def __init__(self, k: float = 1.5) -> None:
        self.k = k
        self.lower_: dict[str, float] | None = None
        self.upper_: dict[str, float] | None = None
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> IQRFencesDetector:
        cols = columns if columns is not None else _numeric_cols(df)
        self._columns = cols
        self.lower_ = {}
        self.upper_ = {}
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[col] = q1 - self.k * iqr
            self.upper_[col] = q3 + self.k * iqr
        self._is_fitted = True
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("IQRFencesDetector has not been fitted.")
        cols = self._columns or _numeric_cols(df)
        result = pd.DataFrame(index=df.index, columns=cols, dtype=bool)
        for col in cols:
            result[col] = (df[col] < self.lower_[col]) | (df[col] > self.upper_[col])
        return result


# ====================================================================
# Z-Score
# ====================================================================


class ZScoreDetector(BaseOutlierDetector):
    """Gaussian z-score outlier detector.

    Values with ``|z| > threshold`` are flagged, where
    ``z = (x − mean) / std``.  Both *mean* and *std* are learned from
    training data.

    Parameters
    ----------
    threshold:
        Number of standard deviations beyond which a point is an outlier.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self.mean_: dict[str, float] | None = None
        self.std_: dict[str, float] | None = None
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> ZScoreDetector:
        cols = columns if columns is not None else _numeric_cols(df)
        self._columns = cols
        self.mean_ = {}
        self.std_ = {}
        for col in cols:
            self.mean_[col] = float(df[col].mean())
            s = float(df[col].std())
            self.std_[col] = s if s > 0 else 1e-12
        self._is_fitted = True
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("ZScoreDetector has not been fitted.")
        cols = self._columns or _numeric_cols(df)
        result = pd.DataFrame(index=df.index, columns=cols, dtype=bool)
        for col in cols:
            z = (df[col] - self.mean_[col]) / self.std_[col]
            result[col] = z.abs() > self.threshold
        return result


# ====================================================================
# Robust Z-Score (Median / MAD)
# ====================================================================


class RobustZDetector(BaseOutlierDetector):
    """Robust z-score outlier detector using median and MAD.

    ``rz = (x − median) / (1.4826 · MAD)`` where MAD is the median
    absolute deviation.  The constant 1.4826 makes the estimator
    consistent with the standard deviation under normality.

    Parameters
    ----------
    threshold:
        Threshold on the robust z-score.
    """

    _MAD_SCALE: float = 1.4826

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold
        self.median_: dict[str, float] | None = None
        self.mad_: dict[str, float] | None = None
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> RobustZDetector:
        cols = columns if columns is not None else _numeric_cols(df)
        self._columns = cols
        self.median_ = {}
        self.mad_ = {}
        for col in cols:
            med = float(df[col].median())
            mad = float(np.median(np.abs(df[col].values - med)))
            self.median_[col] = med
            self.mad_[col] = mad if mad > 0 else 1e-12
        self._is_fitted = True
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("RobustZDetector has not been fitted.")
        cols = self._columns or _numeric_cols(df)
        result = pd.DataFrame(index=df.index, columns=cols, dtype=bool)
        for col in cols:
            rz = (df[col] - self.median_[col]) / (self._MAD_SCALE * self.mad_[col])
            result[col] = rz.abs() > self.threshold
        return result


# ====================================================================
# Winsoriser
# ====================================================================


class Winsorizer:
    """Clip values to training-derived fences.

    Lower and upper fences can be provided explicitly or learned via an
    ``IQRFencesDetector`` or quantile-based approach.

    Parameters
    ----------
    lower:
        Per-column lower bounds.  If *None* and ``fit`` is called, they
        are derived from training quantiles.
    upper:
        Per-column upper bounds.
    quantile_low:
        Lower quantile used when fences are learned (default 0.01).
    quantile_high:
        Upper quantile used when fences are learned (default 0.99).
    """

    def __init__(
        self,
        lower: dict[str, float] | None = None,
        upper: dict[str, float] | None = None,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> Winsorizer:
        cols = columns if columns is not None else _numeric_cols(df)
        self._columns = cols
        if self.lower is None:
            self.lower = {}
        if self.upper is None:
            self.upper = {}
        for col in cols:
            if col not in self.lower:
                self.lower[col] = float(df[col].quantile(self.quantile_low))
            if col not in self.upper:
                self.upper[col] = float(df[col].quantile(self.quantile_high))
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Winsorizer has not been fitted.")
        out = df.copy()
        cols = self._columns or _numeric_cols(df)
        for col in cols:
            out[col] = out[col].clip(lower=self.lower[col], upper=self.upper[col])
        return out

    def fit_transform(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns=columns).transform(df)


# ====================================================================
# Helpers
# ====================================================================


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding identifiers."""
    exclude = {"unit_id", "cycle", "RUL"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
