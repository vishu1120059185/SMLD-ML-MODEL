"""Feature attribution methods for STAT-TWIN.

Three complementary methods are provided:

1. **Group occlusion** (primary) – replaces a feature group (sensor +
   derived statistics) with the healthy baseline and measures the drop in
   predicted risk.  This is model-agnostic and produces intuitive
   contribution scores.

2. **Risk-change decomposition** – decomposes the difference
   ``P(now) − P(reference)`` across sensors and statistic types, showing
   which signals drove the change in risk.

3. **SHAP TreeExplainer** (optional cross-check) – for tree-based models
   (XGBoost, Random Forest) the SHAP TreeExplainer provides exact Shapley
   values.  Requires the ``shap`` package.

All methods use the framing "contributed to the model's risk estimate"
rather than causal claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

__all__ = [
    "AttributionResult",
    "group_occlusion_attribution",
    "risk_change_decomposition",
    "shap_attribution",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"

_FAILURE_HORIZONS = [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Protocol for model-like objects
# ---------------------------------------------------------------------------

class ProbPredictor(Protocol):
    """Minimal protocol for objects with a ``predict_proba`` method."""

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AttributionResult:
    """Container for feature attribution results.

    Attributes
    ----------
    method:
        Attribution method used (``"group_occlusion"``, ``"risk_change"``,
        or ``"shap"``).
    sensor_contributions:
        ``{sensor: contribution_value}`` mapping.  For group occlusion the
        value is the probability drop (positive = higher risk).  For SHAP
        it is the mean absolute Shapley value.
    statistic_contributions:
        ``{stat_type: contribution_value}`` for risk-change decomposition.
        ``None`` for other methods.
    baseline_prediction:
        Reference (healthy) prediction.
    current_prediction:
        Prediction at the observed cycle.
    horizon:
        The failure horizon that was explained.  ``None`` if all horizons.
    metadata:
        Method-specific extra information.
    """

    method: str
    sensor_contributions: dict[str, float]
    statistic_contributions: dict[str, float] | None = None
    baseline_prediction: float = 0.0
    current_prediction: float = 0.0
    horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_sensors(self) -> list[str]:
        """Return sensor names sorted by absolute contribution (descending)."""
        return sorted(
            self.sensor_contributions,
            key=lambda s: abs(self.sensor_contributions[s]),
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "method": self.method,
            "sensor_contributions": self.sensor_contributions,
            "statistic_contributions": self.statistic_contributions,
            "baseline_prediction": self.baseline_prediction,
            "current_prediction": self.current_prediction,
            "horizon": self.horizon,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Group occlusion (primary method)
# ---------------------------------------------------------------------------

def group_occlusion_attribution(
    model: ProbPredictor,
    X_current: pd.DataFrame,
    X_baseline: pd.DataFrame,
    sensor_cols: list[str],
    feature_cols: list[str],
    horizons: list[int] | None = None,
    horizon: int | None = 30,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> AttributionResult:
    """Compute sensor-level attribution via group occlusion.

    For each sensor *s*:
    1. Clone ``X_current``.
    2. Replace all features derived from *s* with their healthy-baseline
       values from ``X_baseline``.
    3. Re-predict: ``P_occluded = model.predict_proba(X_occluded)``.
    4. Contribution of *s* = ``P_current − P_occluded``.

    A positive contribution means the sensor's current values *increase*
    the predicted risk relative to the healthy baseline.

    Parameters
    ----------
    model:
        Fitted model with a ``predict_proba`` method returning a DataFrame
        with ``fail_h{h}`` columns.
    X_current:
        Single-row DataFrame with the current observation's features.
    X_baseline:
        Single-row DataFrame with the healthy-baseline features (typically
        the first cycle or the mean of the first *K* cycles).
    sensor_cols:
        Raw sensor column names (used to identify derived features).
    feature_cols:
        Full list of feature columns expected by the model.
    horizons:
        Subset of horizons to attribute.  If ``None``, all horizons are
        used and contributions are averaged.
    horizon:
        If provided, return attribution for this single horizon only.
        Overrides *horizons*.
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.

    Returns
    -------
    AttributionResult
        Per-sensor contribution scores.
    """
    if horizon is not None:
        target_horizons = [horizon]
    elif horizons is not None:
        target_horizons = horizons
    else:
        target_horizons = _FAILURE_HORIZONS

    # Ensure single-row inputs
    x_cur = X_current.iloc[[0]].copy()
    x_base = X_baseline.iloc[[0]].copy()

    # Current prediction
    proba_cur = model.predict_proba(x_cur)
    label_cols = [f"fail_h{h}" for h in target_horizons if f"fail_h{h}" in proba_cur.columns]

    cur_risk = float(proba_cur[label_cols].values.mean()) if label_cols else 0.0

    # Identify feature columns per sensor
    sensor_feature_map = _map_sensor_to_features(sensor_cols, feature_cols)

    contributions: dict[str, float] = {}
    for sensor in sensor_cols:
        # Build occluded version: replace sensor-derived features with baseline
        x_occluded = x_cur.copy()
        derived = sensor_feature_map.get(sensor, [])
        for feat in derived:
            if feat in x_occluded.columns and feat in x_base.columns:
                x_occluded[feat] = x_base[feat].values

        proba_occluded = model.predict_proba(x_occluded)
        occluded_risk = float(proba_occluded[label_cols].values.mean()) if label_cols else 0.0

        # Contribution = current risk - occluded risk
        contributions[sensor] = cur_risk - occluded_risk

    return AttributionResult(
        method="group_occlusion",
        sensor_contributions=contributions,
        baseline_prediction=cur_risk,
        current_prediction=cur_risk,
        horizon=horizon,
        metadata={
            "target_horizons": target_horizons,
            "n_features_replaced": {
                s: len(sensor_feature_map.get(s, [])) for s in sensor_cols
            },
        },
    )


def _map_sensor_to_features(
    sensor_cols: list[str],
    feature_cols: list[str],
) -> dict[str, list[str]]:
    """Map each raw sensor to its derived feature columns.

    A feature column belongs to a sensor if the sensor name appears as a
    prefix (e.g. ``sensor_1_rmean_10`` belongs to ``sensor_1``).
    Features not matching any sensor are grouped under ``"_other_"``.
    """
    mapping: dict[str, list[str]] = {s: [] for s in sensor_cols}
    for feat in feature_cols:
        matched = False
        for sensor in sensor_cols:
            if feat.startswith(sensor + "_") or feat == sensor:
                mapping[sensor].append(feat)
                matched = True
                break
        if not matched:
            # Feature does not derive from any known sensor – skip
            pass
    return mapping


# ---------------------------------------------------------------------------
# Risk-change decomposition
# ---------------------------------------------------------------------------

def risk_change_decomposition(
    model: ProbPredictor,
    X_now: pd.DataFrame,
    X_ref: pd.DataFrame,
    sensor_cols: list[str],
    feature_cols: list[str],
    stat_types: list[str] | None = None,
    horizon: int = 30,
) -> AttributionResult:
    """Decompose P(now) − P(reference) across sensors and statistic types.

    For each sensor *s* and each statistic type *t* (e.g. ``"mean"``,
    ``"std"``, ``"ewma"``):
    1. Start from ``X_ref``.
    2. Replace only the *t*-type features of sensor *s* with their
       values from ``X_now``.
    3. Predict: ``P_partial``.
    4. Contribution(s, t) = ``P_partial − P_ref``.

    Parameters
    ----------
    model:
        Fitted model with ``predict_proba``.
    X_now:
        Current observation features.
    X_ref:
        Reference (e.g. healthy-baseline or previous-period) features.
    sensor_cols:
        Raw sensor column names.
    feature_cols:
        Full list of feature columns.
    stat_types:
        Statistic type prefixes to decompose over (e.g.
        ``["rmean", "rstd", "ewma", "zscore", "slope"]``).  If ``None``,
        a default set is used.
    horizon:
        Failure horizon to explain.

    Returns
    -------
    AttributionResult
        Both sensor-level and statistic-type-level contributions.
    """
    if stat_types is None:
        stat_types = ["rmean", "rstd", "rmin", "rmax", "zscore", "ewma",
                       "slope", "pctchg", "cv", "skew", "kurt"]

    x_now = X_now.iloc[[0]].copy()
    x_ref = X_ref.iloc[[0]].copy()
    label_col = f"fail_h{horizon}"

    # Reference prediction
    proba_ref = model.predict_proba(x_ref)
    p_ref = float(proba_ref[label_col].iloc[0]) if label_col in proba_ref.columns else 0.0

    # Current prediction
    proba_now = model.predict_proba(x_now)
    p_now = float(proba_now[label_col].iloc[0]) if label_col in proba_now.columns else 0.0

    sensor_feature_map = _map_sensor_to_features(sensor_cols, feature_cols)

    # Sensor-level decomposition
    sensor_contributions: dict[str, float] = {}
    stat_contributions: dict[str, float] = {st: 0.0 for st in stat_types}

    for sensor in sensor_cols:
        derived = sensor_feature_map.get(sensor, [])
        sensor_total = 0.0

        for stat_type in stat_types:
            # Find features for this sensor + stat type
            matching_feats = [
                f for f in derived
                if f"_{stat_type}_" in f or f.endswith(f"_{stat_type}")
            ]
            if not matching_feats:
                continue

            x_partial = x_ref.copy()
            for feat in matching_feats:
                if feat in x_partial.columns and feat in x_now.columns:
                    x_partial[feat] = x_now[feat].values

            proba_partial = model.predict_proba(x_partial)
            p_partial = (
                float(proba_partial[label_col].iloc[0])
                if label_col in proba_partial.columns
                else p_ref
            )
            contribution = p_partial - p_ref

            sensor_total += contribution
            stat_contributions[stat_type] += abs(contribution)

        sensor_contributions[sensor] = sensor_total

    return AttributionResult(
        method="risk_change",
        sensor_contributions=sensor_contributions,
        statistic_contributions=stat_contributions,
        baseline_prediction=p_ref,
        current_prediction=p_now,
        horizon=horizon,
        metadata={"stat_types": stat_types},
    )


# ---------------------------------------------------------------------------
# SHAP TreeExplainer (cross-check)
# ---------------------------------------------------------------------------

def shap_attribution(
    model: Any,
    X_current: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 30,
    max_samples: int = 100,
) -> AttributionResult:
    """Compute SHAP values for tree-based models as a cross-check.

    Requires the ``shap`` package.  Uses ``TreeExplainer`` for exact
    Shapley values when the model is XGBoost or a scikit-learn tree
    ensemble.

    Parameters
    ----------
    model:
        A fitted tree-based model with a ``predict_proba`` or
        ``predict`` method.  Must be compatible with
        ``shap.TreeExplainer``.
    X_current:
        Single-row observation to explain.
    feature_cols:
        Feature columns used by the model.
    horizon:
        Horizon index to explain (used to select the classifier from
        per-horizon dict models).
    max_samples:
        Maximum background samples for SHAP (affects speed).

    Returns
    -------
    AttributionResult
        Mean absolute SHAP values per sensor, aggregated from feature
        level.

    Raises
    ------
    ImportError
        If ``shap`` is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap package is required for shap_attribution(). "
            "Install it with: pip install shap"
        ) from exc

    x = X_current[feature_cols].iloc[[0]].copy()

    # Handle per-horizon dict models (e.g. XGBoostModel._classifiers)
    inner_model = model
    if hasattr(model, "_classifiers") and isinstance(model._classifiers, dict):
        clf = model._classifiers.get(horizon)
        if clf is None:
            raise ValueError(f"No classifier found for horizon {horizon}")
        inner_model = clf

    explainer = shap.TreeExplainer(inner_model)
    shap_values = explainer.shap_values(x)

    # shap_values may be a list (one per class) or a 2-D array
    if isinstance(shap_values, list):
        # Binary classification: use class 1 (positive)
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = np.asarray(shap_values)

    # sv shape: (1, n_features) or (n_features,)
    sv = np.asarray(sv).flatten()
    if len(sv) != len(feature_cols):
        # Truncate or pad
        sv = sv[:len(feature_cols)]

    # Aggregate feature-level SHAP to sensor level
    sensor_feature_map = _map_sensor_to_features(
        [c for c in feature_cols if not any(
            c.endswith(suffix) for suffix in ["_unit_id", "_cycle"]
        )],
        feature_cols,
    )

    sensor_shap: dict[str, float] = {}
    for sensor, feats in sensor_feature_map.items():
        indices = [feature_cols.index(f) for f in feats if f in feature_cols]
        if indices:
            sensor_shap[sensor] = float(np.mean(np.abs(sv[indices])))
        else:
            sensor_shap[sensor] = 0.0

    return AttributionResult(
        method="shap",
        sensor_contributions=sensor_shap,
        horizon=horizon,
        metadata={
            "n_features": len(feature_cols),
            "max_abs_shap": float(np.max(np.abs(sv))) if len(sv) > 0 else 0.0,
        },
    )
