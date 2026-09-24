"""Model 3: Logistic Regression per-horizon classifier.

Trains one ``LogisticRegression`` per failure horizon on raw sensor
features only (ablation variant A).  Each classifier outputs a
probability ``P(RUL <= h | features)``.  The RUL point estimate is
derived by inverting the per-horizon probabilities via isotonic mapping.

This serves as the simplest supervised baseline.

References
----------
*  Cox (1958) – logistic regression.
*  C-MAPSS benchmark (Saxena et al., 2008).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stattwin.data.schema import label_col_for
from stattwin.models.base import BaseModel

__all__ = ["LogisticModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"
_INTERNAL_HORIZONS: list[int] = [10, 20, 30, 40, 50]


class LogisticModel(BaseModel):
    """Logistic Regression per-horizon classifier on raw sensor features.

    Parameters
    ----------
    C:
        Regularisation strength (inverse) for ``LogisticRegression`` (default 1.0).
    max_iter:
        Maximum solver iterations (default 1000).
    class_weight:
        Class weighting strategy (default ``"balanced"``).
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="LogisticModel")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

        self._classifiers: dict[int, LogisticRegression] = {}
        self._scaler: StandardScaler = StandardScaler()
        self._sensor_cols: list[str] = []
        self._isotonic_rul: IsotonicRegression | None = None
        self._iso_probas: dict[int, IsotonicRegression] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        """Return raw sensor columns only (no derived features)."""
        return [
            c for c in X.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X[c])
        ]

    def _get_scaled(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Return standardised feature matrix."""
        vals = X[self._sensor_cols].to_numpy(dtype=np.float64)
        if fit:
            return self._scaler.fit_transform(vals)
        return self._scaler.transform(vals)

    def _compute_weighted_rul(self, proba: pd.DataFrame) -> pd.Series:
        """Derive RUL from per-horizon probabilities via weighted inversion.

        RUL_est = sum_h  P(RUL > h) * delta_h
        where delta_h = h_{i+1} - h_i (with h_0 = 0).
        """
        h_arr = np.array(self.horizons, dtype=np.float64)
        delta = np.diff(np.concatenate(([0.0], h_arr)))
        # P(RUL > h) = 1 - P(fail_h)
        p_survive = 1.0 - proba.values  # shape (n, len(horizons))
        rul = p_survive @ delta
        return pd.Series(np.maximum(rul, 0.0), index=proba.index, name="RUL")

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> LogisticModel:
        """Fit per-horizon logistic classifiers and RUL isotonic mapper.

        Parameters
        ----------
        X_train:
            Feature matrix with raw sensor columns.
        y_train:
            Binary label DataFrame.
        groups:
            Ignored.
        """
        self._sensor_cols = self._select_features(X_train)
        X_scaled = self._get_scaled(X_train, fit=True)

        for h in self.horizons:
            col = label_col_for(h)
            if col not in y_train.columns:
                continue
            clf = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_scaled, y_train[col].to_numpy().astype(np.int32))
            self._classifiers[h] = clf

        # Fit isotonic mapping for RUL using predicted probabilities
        proba_train = pd.DataFrame(
            {
                label_col_for(h): self._classifiers[h].predict_proba(X_scaled)[:, 1]
                for h in self.horizons
                if h in self._classifiers
            }
        )
        if "RUL" in X_train.columns:
            rul_vals = X_train["RUL"].to_numpy().astype(np.float64)
            raw_rul_est = self._compute_weighted_rul(proba_train).to_numpy()
            self._isotonic_rul = IsotonicRegression(
                y_min=0.0, y_max=125.0, out_of_bounds="clip"
            )
            self._isotonic_rul.fit(raw_rul_est, rul_vals)

        # Fit isotonic calibrators per horizon for probability calibration
        for h in self.horizons:
            col = label_col_for(h)
            if col in y_train.columns and col in proba_train.columns:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(proba_train[col].to_numpy(), y_train[col].to_numpy().astype(np.float64))
                self._iso_probas[h] = iso

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities."""
        X_scaled = self._get_scaled(X, fit=False)
        proba_dict: dict[str, np.ndarray] = {}
        for h in self.horizons:
            clf = self._classifiers.get(h)
            if clf is None:
                proba_dict[label_col_for(h)] = np.full(len(X), 0.5)
                continue
            raw_proba = clf.predict_proba(X_scaled)[:, 1]
            iso = self._iso_probas.get(h)
            if iso is not None:
                raw_proba = np.clip(iso.predict(raw_proba), 0.0, 1.0)
            proba_dict[label_col_for(h)] = raw_proba
        return pd.DataFrame(proba_dict, index=X.index)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL from per-horizon probabilities."""
        proba = self.predict_proba(X)
        raw_rul = self._compute_weighted_rul(proba)
        if self._isotonic_rul is not None:
            rul = self._isotonic_rul.predict(raw_rul.to_numpy())
            return pd.Series(np.maximum(rul, 0.0), index=X.index, name="RUL")
        return raw_rul

    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Return the max per-horizon predicted probability as the raw score."""
        X_scaled = self._get_scaled(X, fit=False)
        scores = np.zeros(len(X), dtype=np.float64)
        for h in self.horizons:
            clf = self._classifiers.get(h)
            if clf is not None:
                proba = clf.predict_proba(X_scaled)[:, 1]
                scores = np.maximum(scores, proba)
        return pd.Series(scores, index=X.index, name="raw_score")
