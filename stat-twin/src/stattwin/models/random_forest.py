"""Model 4: Random Forest per-horizon classifier + RUL regressor.

Trains one ``RandomForestClassifier`` per failure horizon and a single
``RandomForestRegressor`` for RUL, all on raw sensor features only
(ablation variant A).  Handles class imbalance with ``class_weight="balanced_subsample"``.
Reports PR-AUC alongside ROC-AUC.

References
----------
*  Breiman (2001) – Random Forests.
*  C-MAPSS benchmark (Saxena et al., 2008).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

from stattwin.data.schema import label_col_for
from stattwin.models.base import BaseModel

__all__ = ["RandomForestModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


class RandomForestModel(BaseModel):
    """Random Forest per-horizon classifier + RUL regressor.

    Parameters
    ----------
    n_estimators:
        Number of trees (default 200).
    max_depth:
        Maximum tree depth (default 10).
    min_samples_leaf:
        Minimum samples in a leaf (default 5).
    class_weight:
        Class weighting for classifiers (default ``"balanced_subsample"``).
    random_state:
        Random seed (default 42).
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced_subsample",
        random_state: int = 42,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="RandomForestModel")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

        self._classifiers: Dict[int, RandomForestClassifier] = {}
        self._regressor: RandomForestRegressor | None = None
        self._sensor_cols: List[str] = []
        self._isotonic_probas: Dict[int, IsotonicRegression] = {}
        self._isotonic_rul: IsotonicRegression | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> List[str]:
        """Return raw sensor columns only."""
        return [
            c for c in X.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X[c])
        ]

    def _get_features(self, X: pd.DataFrame) -> np.ndarray:
        """Return feature array."""
        return X[self._sensor_cols].to_numpy(dtype=np.float64)

    def _compute_weighted_rul(self, proba: pd.DataFrame) -> pd.Series:
        """Derive RUL from per-horizon probabilities."""
        h_arr = np.array(self.horizons, dtype=np.float64)
        delta = np.diff(np.concatenate(([0.0], h_arr)))
        p_survive = 1.0 - proba.values
        rul = p_survive @ delta
        return pd.Series(np.maximum(rul, 0.0), index=proba.index, name="RUL")

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: Optional[np.ndarray] = None,
    ) -> RandomForestModel:
        """Fit per-horizon classifiers, RUL regressor, and isotonic calibrators.

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
        X_arr = self._get_features(X_train)

        # Per-horizon classifiers
        for h in self.horizons:
            col = label_col_for(h)
            if col not in y_train.columns:
                continue
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X_arr, y_train[col].to_numpy().astype(np.int32))
            self._classifiers[h] = clf

        # RUL regressor
        if "RUL" in X_train.columns:
            self._regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._regressor.fit(X_arr, X_train["RUL"].to_numpy().astype(np.float64))

        # Fit isotonic calibrators on training predictions (OOF style)
        proba_train = self.predict_proba(X_train)
        for h in self.horizons:
            col = label_col_for(h)
            if col in y_train.columns and col in proba_train.columns:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(
                    proba_train[col].to_numpy(),
                    y_train[col].to_numpy().astype(np.float64),
                )
                self._isotonic_probas[h] = iso

        if "RUL" in X_train.columns:
            rul_pred = self.predict_rul(X_train)
            self._isotonic_rul = IsotonicRegression(
                y_min=0.0, y_max=125.0, out_of_bounds="clip"
            )
            self._isotonic_rul.fit(
                rul_pred.to_numpy(),
                X_train["RUL"].to_numpy().astype(np.float64),
            )

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities."""
        X_arr = self._get_features(X)
        proba_dict: Dict[str, np.ndarray] = {}
        for h in self.horizons:
            clf = self._classifiers.get(h)
            if clf is None:
                proba_dict[label_col_for(h)] = np.full(len(X), 0.5)
                continue
            raw_proba = clf.predict_proba(X_arr)[:, 1]
            iso = self._isotonic_probas.get(h)
            if iso is not None:
                raw_proba = np.clip(iso.predict(raw_proba), 0.0, 1.0)
            proba_dict[label_col_for(h)] = raw_proba
        return pd.DataFrame(proba_dict, index=X.index)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL."""
        if self._regressor is not None:
            X_arr = self._get_features(X)
            rul = self._regressor.predict(X_arr)
            if self._isotonic_rul is not None:
                rul = self._isotonic_rul.predict(rul)
            return pd.Series(np.maximum(rul, 0.0), index=X.index, name="RUL")
        # Fallback: derive from probabilities
        proba = self.predict_proba(X)
        return self._compute_weighted_rul(proba)

    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Return the max per-horizon probability as the raw score."""
        X_arr = self._get_features(X)
        scores = np.zeros(len(X), dtype=np.float64)
        for h in self.horizons:
            clf = self._classifiers.get(h)
            if clf is not None:
                proba = clf.predict_proba(X_arr)[:, 1]
                scores = np.maximum(scores, proba)
        return pd.Series(scores, index=X.index, name="raw_score")
