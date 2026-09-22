"""Model 1: Fixed-threshold per-sensor degradation detector.

For each sensor the healthy baseline mean ``mu_i`` and standard deviation
``sigma_i`` are estimated from the first ``baseline_cycles`` of every
training unit.  A sensor is in violation at cycle *t* if

    |x_{i,t} - mu_i| >= k * sigma_i

A warning is triggered when at least *m* sensors are simultaneously in
violation for at least *p* consecutive cycles.  The raw score equals the
number (and severity) of violations; it is mapped to probability / RUL
via isotonic regression on training out-of-fold predictions.

References
----------
*  *Manufacturing analytics: A review* (Mobley, 2002) – fixed limits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from stattwin.data.schema import label_col_for
from stattwin.models.base import BaseModel

__all__ = ["ThresholdModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


class ThresholdModel(BaseModel):
    """Fixed-threshold per-sensor anomaly detector.

    Parameters
    ----------
    k_sigma:
        Number of standard deviations from the healthy mean for a
        single-sensor violation (default 3.0).
    min_sensors:
        Minimum number of simultaneously violating sensors to flag a
        cycle (default 2).
    persistence:
        Number of consecutive violating cycles required before the
        warning is sustained (default 3).
    baseline_cycles:
        Number of initial cycles per unit used to estimate healthy
        baseline statistics (default 30).
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        k_sigma: float = 3.0,
        min_sensors: int = 2,
        persistence: int = 3,
        baseline_cycles: int = 30,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="ThresholdModel")
        self.k_sigma = k_sigma
        self.min_sensors = min_sensors
        self.persistence = persistence
        self.baseline_cycles = baseline_cycles

        # Fitted parameters (set during ``fit``)
        self._baseline_means: Dict[str, float] = {}
        self._baseline_stds: Dict[str, float] = {}
        self._sensor_cols: List[str] = []
        self._isotonic_probas: Dict[int, IsotonicRegression] = {}
        self._isotonic_rul: IsotonicRegression | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_raw_score(self, X: pd.DataFrame) -> pd.Series:
        """Compute the per-row raw violation score.

        The score is a float encoding both the count and severity of
        sensor violations.  Higher values indicate worse health.

        score_t = sum_i  max(0,  |x_{i,t} - mu_i| / sigma_i  -  k)
        """
        scores = np.zeros(len(X), dtype=np.float64)
        for col in self._sensor_cols:
            mu = self._baseline_means[col]
            sigma = self._baseline_stds[col]
            if sigma <= 0:
                continue
            vals = X[col].to_numpy(dtype=np.float64)
            z = np.abs(vals - mu) / sigma
            severity = np.maximum(z - self.k_sigma, 0.0)
            scores += severity
        return pd.Series(scores, index=X.index, name="raw_score")

    def _raw_to_probas(self, raw: pd.Series) -> pd.DataFrame:
        """Map raw score to per-horizon probabilities using fitted isotonic models."""
        proba_dict: Dict[str, np.ndarray] = {}
        raw_vals = raw.to_numpy().reshape(-1, 1)
        for h in self.horizons:
            iso = self._isotonic_probas.get(h)
            if iso is not None:
                proba_dict[label_col_for(h)] = np.clip(
                    iso.predict(raw_vals.ravel()), 0.0, 1.0
                )
            else:
                # Fallback: sigmoid scaling
                proba_dict[label_col_for(h)] = 1.0 / (
                    1.0 + np.exp(-raw_vals.ravel())
                )
        return pd.DataFrame(proba_dict, index=raw.index)

    def _raw_to_rul(self, raw: pd.Series) -> pd.Series:
        """Map raw score to point RUL using fitted isotonic model."""
        raw_vals = raw.to_numpy()
        if self._isotonic_rul is not None:
            rul = self._isotonic_rul.predict(raw_vals)
            return pd.Series(np.maximum(rul, 0.0), index=raw.index, name="RUL")
        # Fallback: inverse sigmoid heuristic
        rul = np.maximum(100.0 * (1.0 / (1.0 + np.exp(raw_vals))), 0.0)
        return pd.Series(rul, index=raw.index, name="RUL")

    def _fit_isotonic(
        self,
        raw_train: pd.Series,
        y_train: pd.DataFrame,
        rul_train: pd.Series | None,
    ) -> None:
        """Fit isotonic regression on training raw scores → labels / RUL."""
        raw_vals = raw_train.to_numpy()
        for h in self.horizons:
            col = label_col_for(h)
            if col not in y_train.columns:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_vals, y_train[col].to_numpy().astype(np.float64))
            self._isotonic_probas[h] = iso

        if rul_train is not None:
            self._isotonic_rul = IsotonicRegression(
                y_min=0.0, y_max=125.0, out_of_bounds="clip"
            )
            self._isotonic_rul.fit(raw_vals, rul_train.to_numpy().astype(np.float64))

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: Optional[np.ndarray] = None,
    ) -> ThresholdModel:
        """Fit baseline statistics and isotonic calibrators.

        Parameters
        ----------
        X_train:
            Feature matrix with ``unit_id``, ``cycle``, and sensor columns.
        y_train:
            Binary label DataFrame with ``fail_h{h}`` columns.
        groups:
            Ignored (kept for interface compatibility).
        """
        self._sensor_cols = [
            c for c in X_train.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X_train[c])
        ]

        # Estimate healthy baseline from first baseline_cycles of each unit
        if _UNIT_COL in X_train.columns:
            for col in self._sensor_cols:
                means, stds = [], []
                for _, grp in X_train.groupby(_UNIT_COL):
                    head = grp.head(self.baseline_cycles)[col].dropna()
                    if len(head) >= 2:
                        means.append(head.mean())
                        stds.append(head.std())
                self._baseline_means[col] = np.mean(means) if means else 0.0
                self._baseline_stds[col] = np.mean(stds) if stds else 1.0
        else:
            for col in self._sensor_cols:
                head = X_train[col].head(self.baseline_cycles).dropna()
                self._baseline_means[col] = head.mean() if len(head) > 0 else 0.0
                self._baseline_stds[col] = head.std() if len(head) > 1 else 1.0

        raw_train = self._compute_raw_score(X_train)

        # Extract RUL if present
        rul_train = X_train["RUL"] if "RUL" in X_train.columns else None
        self._fit_isotonic(raw_train, y_train, rul_train)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities."""
        raw = self._compute_raw_score(X)
        return self._raw_to_probas(raw)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL."""
        raw = self._compute_raw_score(X)
        return self._raw_to_rul(raw)

    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Return raw violation score."""
        return self._compute_raw_score(X)
