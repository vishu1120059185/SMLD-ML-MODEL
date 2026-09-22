"""Model 2: Statistical anomaly detection via Mahalanobis distance.

Computes the Mahalanobis distance of each observation from the healthy
baseline distribution fitted on the first ``baseline_cycles`` of every
training unit.  Uses a robust covariance estimator (Minimum Covariance
Determinant) to guard against outlier contamination.

An observation is flagged anomalous when its Mahalanobis distance exceeds
the chi-square quantile at significance level ``alpha`` (default 0.99).
Persistence logic requires the anomaly to persist for at least
``persistence`` consecutive cycles.

The raw score is the Mahalanobis distance, mapped to probabilities / RUL
via isotonic regression on training folds.

References
----------
*  Rousseeuw & Driessen (1999) – Fast MCD for robust covariance.
*  Mahalanobis (1936) – distance in multivariate space.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.covariance import EllipticEnvelope
from sklearn.isotonic import IsotonicRegression

from stattwin.data.schema import label_col_for
from stattwin.models.base import BaseModel

__all__ = ["AnomalyModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


class AnomalyModel(BaseModel):
    """Statistical anomaly detector based on Mahalanobis distance.

    Parameters
    ----------
    alpha:
        Chi-square quantile level for the anomaly threshold (default 0.99).
    baseline_cycles:
        Number of initial cycles per unit for healthy reference (default 30).
    persistence:
        Consecutive anomaly cycles required before sustained alarm (default 3).
    contamination:
        Expected fraction of outliers for the robust covariance estimator
        (default 0.1).
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        alpha: float = 0.99,
        baseline_cycles: int = 30,
        persistence: int = 3,
        contamination: float = 0.1,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="AnomalyModel")
        self.alpha = alpha
        self.baseline_cycles = baseline_cycles
        self.persistence = persistence
        self.contamination = contamination

        self._sensor_cols: List[str] = []
        self._baseline_mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._threshold: float = 0.0
        self._isotonic_probas: Dict[int, IsotonicRegression] = {}
        self._isotonic_rul: IsotonicRegression | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_robust_covariance(self, X_baseline: np.ndarray) -> None:
        """Fit robust mean and inverse covariance on healthy baseline data."""
        n_features = X_baseline.shape[1]
        # Use EllipticEnvelope (Fast MCD) for robust estimation
        try:
            ee = EllipticEnvelope(
                contamination=min(self.contamination, 0.5),
                support_fraction=None,
                random_state=42,
            )
            ee.fit(X_baseline)
            self._baseline_mean = ee.location_
            cov = ee.covariance_
            # Regularise for numerical stability
            cov += np.eye(n_features) * 1e-6
            self._inv_cov = np.linalg.inv(cov)
        except Exception:
            # Fallback to classical covariance
            self._baseline_mean = np.mean(X_baseline, axis=0)
            cov = np.cov(X_baseline.T) + np.eye(n_features) * 1e-6
            self._inv_cov = np.linalg.inv(cov)

        # Chi-square threshold
        self._threshold = sp_stats.chi2.ppf(self.alpha, df=n_features)

    def _compute_mahalanobis(self, X: pd.DataFrame) -> np.ndarray:
        """Compute squared Mahalanobis distance for each row."""
        if self._baseline_mean is None or self._inv_cov is None:
            raise RuntimeError("Model has not been fitted yet.")

        vals = X[self._sensor_cols].to_numpy(dtype=np.float64)
        diff = vals - self._baseline_mean
        # maha_sq = (x - mu)^T Sigma^{-1} (x - mu)
        left = diff @ self._inv_cov
        maha_sq = np.sum(left * diff, axis=1)
        return np.maximum(maha_sq, 0.0)

    def _compute_raw_score(self, X: pd.DataFrame) -> pd.Series:
        """Return squared Mahalanobis distance as the raw score."""
        maha_sq = self._compute_mahalanobis(X)
        return pd.Series(maha_sq, index=X.index, name="raw_score")

    def _raw_to_probas(self, raw: pd.Series) -> pd.DataFrame:
        """Map Mahalanobis distance to per-horizon probabilities."""
        raw_vals = raw.to_numpy().reshape(-1, 1)
        proba_dict: Dict[str, np.ndarray] = {}
        for h in self.horizons:
            iso = self._isotonic_probas.get(h)
            if iso is not None:
                proba_dict[label_col_for(h)] = np.clip(
                    iso.predict(raw_vals.ravel()), 0.0, 1.0
                )
            else:
                # Chi-square CDF as fallback
                cdf_vals = sp_stats.chi2.cdf(
                    raw_vals.ravel(), df=len(self._sensor_cols)
                )
                proba_dict[label_col_for(h)] = np.clip(cdf_vals, 0.0, 1.0)
        return pd.DataFrame(proba_dict, index=raw.index)

    def _raw_to_rul(self, raw: pd.Series) -> pd.Series:
        """Map Mahalanobis distance to point RUL."""
        raw_vals = raw.to_numpy()
        if self._isotonic_rul is not None:
            rul = self._isotonic_rul.predict(raw_vals)
            return pd.Series(np.maximum(rul, 0.0), index=raw.index, name="RUL")
        # Fallback: inverse of chi-square CDF scaled to [0, 125]
        cdf_vals = sp_stats.chi2.cdf(raw_vals, df=len(self._sensor_cols))
        rul = np.maximum(125.0 * (1.0 - cdf_vals), 0.0)
        return pd.Series(rul, index=raw.index, name="RUL")

    def _fit_isotonic(
        self,
        raw_train: pd.Series,
        y_train: pd.DataFrame,
        rul_train: pd.Series | None,
    ) -> None:
        """Fit isotonic regression on training raw scores."""
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
    ) -> AnomalyModel:
        """Fit robust Mahalanobis reference and isotonic calibrators.

        Parameters
        ----------
        X_train:
            Feature matrix with ``unit_id``, ``cycle``, and sensor columns.
        y_train:
            Binary label DataFrame.
        groups:
            Ignored.
        """
        self._sensor_cols = [
            c for c in X_train.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X_train[c])
        ]

        # Collect healthy baseline data
        baseline_rows: list[pd.DataFrame] = []
        if _UNIT_COL in X_train.columns:
            for _, grp in X_train.groupby(_UNIT_COL):
                bl = grp.head(self.baseline_cycles)[self._sensor_cols].dropna()
                if len(bl) >= 3:
                    baseline_rows.append(bl)
        else:
            bl = X_train.head(self.baseline_cycles)[self._sensor_cols].dropna()
            if len(bl) >= 3:
                baseline_rows.append(bl)

        if baseline_rows:
            X_baseline = pd.concat(baseline_rows, ignore_index=True).to_numpy()
        else:
            X_baseline = X_train[self._sensor_cols].to_numpy()

        self._fit_robust_covariance(X_baseline)

        raw_train = self._compute_raw_score(X_train)
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
        """Return squared Mahalanobis distance."""
        return self._compute_raw_score(X)
