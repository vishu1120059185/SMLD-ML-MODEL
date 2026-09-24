"""Model 5: XGBoost per-horizon classifier + RUL regressor.

Primary fast learner – gradient-boosted trees on raw sensor features
(ablation variant A).  Trains one ``XGBClassifier`` per failure horizon
and one ``XGBRegressor`` for RUL.  Handles class imbalance via
``scale_pos_weight`` computed from label frequencies.

XGBoost serves as the strongest tabular baseline and forms the core
of the hybrid ensemble's tabular fallback path.

References
----------
*  Chen & Guestrin (2016) – XGBoost.
*  C-MAPSS benchmark (Saxena et al., 2008).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from stattwin.data.schema import label_col_for
from stattwin.models.base import BaseModel

__all__ = ["XGBoostModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


class XGBoostModel(BaseModel):
    """XGBoost per-horizon classifier + RUL regressor.

    Parameters
    ----------
    n_estimators:
        Boosting rounds (default 400).
    max_depth:
        Maximum tree depth (default 5).
    learning_rate:
        Boosting learning rate (default 0.05).
    subsample:
        Row subsampling ratio (default 0.8).
    colsample_bytree:
        Column subsampling ratio (default 0.8).
    min_child_weight:
        Minimum sum of instance weight in a child (default 5).
    random_state:
        Random seed (default 42).
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        random_state: int = 42,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="XGBoostModel")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state

        self._classifiers: dict[int, Any] = {}
        self._regressor: Any = None
        self._sensor_cols: list[str] = []
        self._isotonic_probas: dict[int, IsotonicRegression] = {}
        self._isotonic_rul: IsotonicRegression | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        """Return raw sensor columns only."""
        return [
            c for c in X.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X[c])
        ]

    def _get_features(self, X: pd.DataFrame) -> np.ndarray:
        return X[self._sensor_cols].to_numpy(dtype=np.float64)

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """Compute scale_pos_weight for class imbalance handling."""
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0:
            return float(n_neg / n_pos)
        return 1.0

    def _compute_weighted_rul(self, proba: pd.DataFrame) -> pd.Series:
        """Derive RUL from per-horizon probabilities."""
        h_arr = np.array(self.horizons, dtype=np.float64)
        delta = np.diff(np.concatenate(([0.0], h_arr)))
        p_survive = 1.0 - proba.values
        rul = p_survive @ delta
        return pd.Series(np.maximum(rul, 0.0), index=proba.index, name="RUL")

    def _make_classifier(self, scale_pos_weight: float = 1.0) -> Any:
        """Create an XGBClassifier with shared hyperparameters."""
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="aucpr",
            n_jobs=-1,
        )

    def _make_regressor(self) -> Any:
        """Create an XGBRegressor with shared hyperparameters."""
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> XGBoostModel:
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

        # Per-horizon classifiers with class-imbalance weighting
        for h in self.horizons:
            col = label_col_for(h)
            if col not in y_train.columns:
                continue
            y_arr = y_train[col].to_numpy().astype(np.int32)
            spw = self._compute_scale_pos_weight(y_arr)
            clf = self._make_classifier(scale_pos_weight=spw)
            clf.fit(X_arr, y_arr)
            self._classifiers[h] = clf

        # RUL regressor
        if "RUL" in X_train.columns:
            self._regressor = self._make_regressor()
            self._regressor.fit(X_arr, X_train["RUL"].to_numpy().astype(np.float64))

        # Isotonic calibrators on training predictions
        proba_train = self._predict_proba_raw(X_arr)
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
            rul_pred = self._predict_rul_raw(X_arr)
            self._isotonic_rul = IsotonicRegression(
                y_min=0.0, y_max=125.0, out_of_bounds="clip"
            )
            self._isotonic_rul.fit(
                rul_pred.to_numpy(),
                X_train["RUL"].to_numpy().astype(np.float64),
            )

        self.is_fitted = True
        return self

    def _predict_proba_raw(self, X_arr: np.ndarray) -> pd.DataFrame:
        """Raw probability prediction before isotonic calibration."""
        proba_dict: dict[str, np.ndarray] = {}
        for h in self.horizons:
            clf = self._classifiers.get(h)
            if clf is not None:
                proba_dict[label_col_for(h)] = clf.predict_proba(X_arr)[:, 1]
            else:
                proba_dict[label_col_for(h)] = np.full(X_arr.shape[0], 0.5)
        return pd.DataFrame(proba_dict)

    def _predict_rul_raw(self, X_arr: np.ndarray) -> pd.Series:
        """Raw RUL prediction before isotonic calibration."""
        if self._regressor is not None:
            rul = self._regressor.predict(X_arr)
            return pd.Series(np.maximum(rul, 0.0), name="RUL")
        return pd.Series(np.full(X_arr.shape[0], 60.0), name="RUL")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities."""
        X_arr = self._get_features(X)
        raw = self._predict_proba_raw(X_arr)
        calibrated = {}
        for h in self.horizons:
            col = label_col_for(h)
            iso = self._isotonic_probas.get(h)
            if iso is not None:
                calibrated[col] = np.clip(iso.predict(raw[col].to_numpy()), 0.0, 1.0)
            else:
                calibrated[col] = raw[col].to_numpy()
        return pd.DataFrame(calibrated, index=X.index)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL."""
        X_arr = self._get_features(X)
        raw_rul = self._predict_rul_raw(X_arr)
        if self._isotonic_rul is not None:
            rul = self._isotonic_rul.predict(raw_rul.to_numpy())
            return pd.Series(np.maximum(rul, 0.0), index=X.index, name="RUL")
        return pd.Series(raw_rul.to_numpy(), index=X.index, name="RUL")

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

    @property
    def feature_importances_(self) -> dict[int, np.ndarray]:
        """Return per-horizon feature importances from fitted classifiers."""
        result: dict[int, np.ndarray] = {}
        for h, clf in self._classifiers.items():
            result[h] = clf.feature_importances_
        return result
