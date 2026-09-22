"""Abstract base class for all STAT-TWIN prediction models.

Every model must implement four methods:

* ``fit(X_train, y_train, groups=None)`` – learn from training data.
* ``predict_proba(X)`` – return per-horizon failure probabilities.
* ``predict_rul(X)`` – return point RUL estimate.
* ``score_raw(X)`` – return raw anomaly/health score (unmapped).

``ModelResult`` bundles predictions with metadata for downstream
evaluation and calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for

__all__ = ["BaseModel", "ModelResult"]


@dataclass
class ModelResult:
    """Container for model predictions and associated metadata.

    Attributes
    ----------
    proba:
        DataFrame with columns ``fail_h{h}`` for each horizon, values in [0, 1].
    rul:
        Series of point RUL estimates (non-negative floats).
    raw_score:
        Optional Series of raw model scores before isotonic / threshold mapping.
    fold:
        Outer fold index that produced this result (-1 for full-data refit).
    model_name:
        Human-readable model identifier.
    metadata:
        Arbitrary extra info (e.g. training time, feature importances).
    """

    proba: pd.DataFrame
    rul: pd.Series
    raw_score: Optional[pd.Series] = None
    fold: int = -1
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for STAT-TWIN prediction models.

    Subclasses must implement ``fit``, ``predict_proba``, ``predict_rul``,
    and ``score_raw``.  The base class provides convenience helpers for
    column bookkeeping and validation.

    Parameters
    ----------
    horizons:
        Failure horizons to predict (default ``[10, 20, 30, 40, 50]``).
    name:
        Human-readable model name.
    """

    def __init__(
        self,
        horizons: Sequence[int] | None = None,
        name: str = "BaseModel",
    ) -> None:
        self.horizons: List[int] = list(horizons or FAILURE_HORIZONS)
        self.name = name
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Fit the model on training data.

        Parameters
        ----------
        X_train:
            Feature matrix (rows = cycles, columns = features).
        y_train:
            Binary label DataFrame with columns ``fail_h{h}`` for each
            horizon in ``self.horizons``.
        groups:
            Optional group array (e.g. ``unit_id``) for group-aware splits.

        Returns
        -------
        BaseModel
            ``self`` (for method chaining).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            Columns ``fail_h{h}`` for each horizon, values clipped to [0, 1].
        """
        ...

    @abstractmethod
    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL for each row.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.Series
            Non-negative RUL estimates.
        """
        ...

    @abstractmethod
    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Return the raw anomaly / health score (unmapped).

        This is the model's native scoring signal before any isotonic
        regression or threshold calibration.  Higher values should indicate
        *worsening* health (closer to failure).

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.Series
            Raw score per row.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_result(
        self,
        X: pd.DataFrame,
        fold: int = -1,
        metadata: Dict[str, Any] | None = None,
    ) -> ModelResult:
        """Build a ``ModelResult`` from predictions.

        Parameters
        ----------
        X:
            Feature matrix.
        fold:
            Outer fold index.
        metadata:
            Arbitrary extra info.

        Returns
        -------
        ModelResult
        """
        proba = self.predict_proba(X)
        rul = self.predict_rul(X)
        raw = self.score_raw(X)
        return ModelResult(
            proba=proba,
            rul=rul,
            raw_score=raw,
            fold=fold,
            model_name=self.name,
            metadata=metadata or {},
        )

    def _validate_columns(self, X: pd.DataFrame, required: list[str] | None = None) -> None:
        """Raise if required columns are missing from *X*."""
        if required is not None:
            missing = set(required) - set(X.columns)
            if missing:
                raise ValueError(f"Missing columns in X: {missing}")

    @property
    def label_cols(self) -> list[str]:
        """Return the ``fail_h{h}`` column names for this model's horizons."""
        return [label_col_for(h) for h in self.horizons]

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(horizons={self.horizons}, {fitted})"
