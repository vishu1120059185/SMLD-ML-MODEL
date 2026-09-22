"""Train-only feature selection for STAT-TWIN.

Provides two complementary strategies executed strictly on the training
partition to prevent leakage:

1. **Near-duplicate removal** – drop features whose absolute Pearson
   correlation with an already-selected feature exceeds a threshold
   (default |ρ| > 0.98).
2. **Top-N by importance** – rank remaining features by either mutual
   information or tree-based (XGBoost / RandomForest) feature importance
   and keep the top N.

Both steps are fit on training data only; ``transform`` applies the
same column mask to any DataFrame.

Design rules (from masterplan):
* Fit selectors on training portions only.
* No information from validation/test partitions leaks into the
  selection process.
* Deterministic with seed support.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

__all__ = ["select_features", "FeatureSelector"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


class FeatureSelector:
    """Train-only feature selection: deduplicate then rank by importance.

    Parameters
    ----------
    corr_threshold:
        Absolute Pearson correlation threshold for near-duplicate
        removal.  Pairs with |ρ| > *corr_threshold* have the
        less-important feature dropped.
    top_n:
        Number of top features to keep after deduplication.  If *None*,
        all non-duplicate features are kept.
    importance_method:
        ``"mi"`` for mutual-information ranking, or ``"tree"`` for
        tree-based importance (requires ``y`` during ``fit``).
    max_features:
        Hard cap on the number of output features.  Applied after
        all other selection steps.  If *None*, no cap.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        corr_threshold: float = 0.98,
        top_n: int | None = None,
        importance_method: Literal["mi", "tree"] = "mi",
        max_features: int | None = None,
        seed: int = 42,
    ) -> None:
        if importance_method not in ("mi", "tree"):
            raise ValueError(
                f"importance_method must be 'mi' or 'tree', got '{importance_method}'"
            )
        self.corr_threshold = corr_threshold
        self.top_n = top_n
        self.importance_method = importance_method
        self.max_features = max_features
        self.seed = seed

        self.selected_features_: list[str] | None = None
        self.feature_importances_: pd.Series | None = None
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        feature_cols: list[str] | None = None,
    ) -> FeatureSelector:
        """Fit the selector on training data only.

        Parameters
        ----------
        X:
            Training feature matrix (may include non-feature columns;
            only *feature_cols* are used).
        y:
            Target variable (required when ``importance_method="tree"``).
        feature_cols:
            Columns to consider for selection.  If *None*, all numeric
            columns except identifiers are used.

        Returns
        -------
        FeatureSelector
            Fitted instance with ``selected_features_`` populated.
        """
        if feature_cols is None:
            feature_cols = _numeric_feature_cols(X)

        if len(feature_cols) == 0:
            self.selected_features_ = []
            self.feature_importances_ = pd.Series(dtype=np.float64)
            self._is_fitted = True
            return self

        # Step 1: near-duplicate removal
        deduped = self._remove_near_duplicates(X, feature_cols)

        # Step 2: rank by importance
        ranked = self._rank_by_importance(X, y, deduped)

        # Step 3: top-N
        if self.top_n is not None and self.top_n < len(ranked):
            ranked = ranked[: self.top_n]

        # Step 4: hard cap
        if self.max_features is not None and self.max_features < len(ranked):
            ranked = ranked[: self.max_features]

        self.selected_features_ = ranked
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* with only the selected feature columns (plus identifiers).

        Parameters
        ----------
        df:
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with identifier columns and selected features only.
        """
        if not self._is_fitted or self.selected_features_ is None:
            raise RuntimeError("FeatureSelector has not been fitted.")

        id_cols = [_UNIT_COL, _CYCLE_COL]
        keep = [c for c in id_cols if c in df.columns] + [
            c for c in self.selected_features_ if c in df.columns
        ]
        return df[keep].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit on training data then return selected columns.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Target variable.
        feature_cols:
            Columns to consider.

        Returns
        -------
        pd.DataFrame
            DataFrame with only the selected columns.
        """
        self.fit(X, y=y, feature_cols=feature_cols)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Internal: near-duplicate removal
    # ------------------------------------------------------------------

    def _remove_near_duplicates(
        self, X: pd.DataFrame, feature_cols: list[str]
    ) -> list[str]:
        """Remove features with |ρ| > threshold vs an already-kept feature.

        Features are considered in the order they appear.  If feature B
        has |ρ| > threshold with an already-kept feature A, B is dropped.
        """
        if len(feature_cols) <= 1:
            return list(feature_cols)

        corr_matrix = X[feature_cols].corr(method="pearson").abs()
        n = len(feature_cols)
        keep_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n):
                if not keep_mask[j]:
                    continue
                if corr_matrix.iloc[i, j] > self.corr_threshold:
                    keep_mask[j] = False

        return [feature_cols[i] for i in range(n) if keep_mask[i]]

    # ------------------------------------------------------------------
    # Internal: importance ranking
    # ------------------------------------------------------------------

    def _rank_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        feature_cols: list[str],
    ) -> list[str]:
        """Rank features by importance and return sorted list (best first)."""
        if len(feature_cols) == 0:
            return []

        if self.importance_method == "mi":
            importances = self._mutual_info_importance(X, feature_cols, y)
        else:
            importances = self._tree_importance(X, feature_cols, y)

        self.feature_importances_ = importances
        return importances.sort_values(ascending=False).index.tolist()

    def _mutual_info_importance(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        y: pd.Series | None,
    ) -> pd.Series:
        """Compute mutual information between each feature and the target.

        If *y* is not provided, an entropy-based self-relevance score is
        computed instead (MI of each feature with itself after binning).
        """
        valid_cols = [
            c for c in feature_cols
            if X[c].notna().sum() > 1 and X[c].nunique() > 1
        ]

        if len(valid_cols) == 0:
            return pd.Series(0.0, index=feature_cols)

        X_clean = X[valid_cols].fillna(0).values

        if y is not None and y.notna().sum() > 1:
            y_clean = y.fillna(0).values
            mi_scores = mutual_info_regression(
                X_clean, y_clean, random_state=self.seed, n_neighbors=5
            )
        else:
            # Self-relevance: MI of each feature discretised with itself
            mi_scores = np.zeros(len(valid_cols), dtype=np.float64)
            for idx in range(len(valid_cols)):
                col_vals = X_clean[:, idx]
                # Discretise into quantile bins
                n_bins = min(10, len(np.unique(col_vals)))
                if n_bins < 2:
                    continue
                try:
                    bins = np.quantile(
                        col_vals, np.linspace(0, 1, n_bins + 1)
                    )
                    bins = np.unique(bins)
                    if len(bins) < 2:
                        continue
                    digitised = np.digitize(col_vals, bins[1:-1], right=True)
                    mi_scores[idx] = float(np.std(digitised))
                except Exception:
                    continue

        return pd.Series(mi_scores, index=valid_cols).reindex(
            feature_cols, fill_value=0.0
        )

    def _tree_importance(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        y: pd.Series | None,
    ) -> pd.Series:
        """Compute feature importance via a lightweight tree model."""
        if y is None:
            warnings.warn(
                "y not provided for tree importance; falling back to MI.",
                UserWarning,
                stacklevel=2,
            )
            return self._mutual_info_importance(X, feature_cols, None)

        valid_cols = [
            c for c in feature_cols
            if X[c].notna().sum() > 1 and X[c].nunique() > 1
        ]

        if len(valid_cols) == 0:
            return pd.Series(0.0, index=feature_cols)

        X_clean = X[valid_cols].fillna(0).values
        y_clean = y.fillna(0).values

        try:
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=self.seed,
                n_jobs=-1,
            )
            rf.fit(X_clean, y_clean)
            importances = rf.feature_importances_
        except Exception:
            # Fallback: correlation-based importance
            importances = np.array(
                [abs(X[c].corr(y)) if X[c].notna().sum() > 1 else 0.0
                 for c in valid_cols]
            )

        return pd.Series(importances, index=valid_cols).reindex(
            feature_cols, fill_value=0.0
        )


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------


def select_features(
    df: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    corr_threshold: float = 0.98,
    top_n: int | None = None,
    importance_method: Literal["mi", "tree"] = "mi",
    max_features: int | None = None,
    feature_cols: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, FeatureSelector]:
    """Fit a feature selector on training data and return selected columns.

    Parameters
    ----------
    df:
        Training DataFrame.
    y:
        Target variable.  Required for ``importance_method="tree"``.
    corr_threshold:
        Near-duplicate correlation threshold.
    top_n:
        Number of top features to keep.
    importance_method:
        ``"mi"`` or ``"tree"``.
    max_features:
        Hard cap on number of output features.
    feature_cols:
        Columns to consider.  If *None*, all numeric columns except
        identifiers are used.
    seed:
        Random seed.

    Returns
    -------
    (selected_df, selector)
        *selected_df* contains only identifier + selected feature columns.
        *selector* is the fitted ``FeatureSelector`` for transforming
        future data.

    Examples
    --------
    >>> train_selected, sel = select_features(train_df, y=train_rul, top_n=50)
    >>> test_selected = sel.transform(test_df)
    """
    selector = FeatureSelector(
        corr_threshold=corr_threshold,
        top_n=top_n,
        importance_method=importance_method,
        max_features=max_features,
        seed=seed,
    )
    selected_df = selector.fit_transform(df, y=y, feature_cols=feature_cols)
    return selected_df, selector


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding identifiers."""
    exclude = {_UNIT_COL, _CYCLE_COL, "RUL"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
