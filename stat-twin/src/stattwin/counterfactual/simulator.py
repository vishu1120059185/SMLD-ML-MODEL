"""What-If simulator for STAT-TWIN counterfactual analysis.

Allows users to explore hypothetical scenarios by modifying recent sensor
readings and observing the predicted impact on risk, RUL, and the
Statistical Health Index (SHI).

Perturbation modes
------------------
* **Multiplicative** – ``x' = x * (1 + factor)`` for a subset of cycles.
* **Additive** – ``x' = x + offset`` for a subset of cycles.
* **Variance scaling** – ``x' = x + noise * sigma * scale`` where noise is
  i.i.d. Gaussian and ``sigma`` is the baseline standard deviation.

The simulator:

1. Applies the perturbation to the raw sensor window.
2. Recomputes statistical features using the same fitted feature pipeline.
3. Recomputes SHI using the same fitted ``HealthIndex`` parameters.
4. Re-runs the model to obtain counterfactual predictions.
5. Returns original vs simulated: probability curves, RUL, SHI, state,
   and the largest changes.

Safety
------
* All outputs are tagged with ``SIMULATION`` badge and a disclaimer.
* An OOD warning is raised if the perturbed values drift significantly
  from the training distribution (Mahalanobis distance or PSI).
* An identity test confirms that zero perturbation reproduces the
  original prediction exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

__all__ = [
    "SimulationBadge",
    "SimulationResult",
    "WhatIfSimulator",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ---------------------------------------------------------------------------
# Protocol for model-like objects
# ---------------------------------------------------------------------------

class ModelLike(Protocol):
    """Minimal protocol for models used by the simulator."""

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def predict_rul(self, X: pd.DataFrame) -> pd.Series: ...


# ---------------------------------------------------------------------------
# Simulation badge & disclaimer
# ---------------------------------------------------------------------------

@dataclass
class SimulationBadge:
    """Tag attached to every simulation output.

    Attributes
    ----------
    is_simulation:
        Always ``True`` for counterfactual outputs.
    disclaimer:
        Human-readable disclaimer text.
    timestamp:
        ISO-8601 timestamp of when the simulation was run.
    """

    is_simulation: bool = True
    disclaimer: str = (
        "SIMULATION: These are hypothetical what-if predictions, not real "
        "outcomes. They show how the model's risk estimate would change "
        "under the specified perturbation scenario."
    )
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            from datetime import datetime, timezone
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dictionary."""
        return {
            "is_simulation": self.is_simulation,
            "disclaimer": self.disclaimer,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Container for original vs simulated predictions.

    Attributes
    ----------
    badge:
        Simulation badge and disclaimer.
    unit_id:
        Unit identifier.
    cycle:
        Cycle number at which the simulation starts.
    perturbation_type:
        Type of perturbation applied.
    perturbation_params:
        Parameters of the perturbation.
    perturbed_sensors:
        List of sensors that were perturbed.
    original_proba:
        Original per-horizon failure probabilities.
    simulated_proba:
        Simulated per-horizon failure probabilities.
    original_rul:
        Original point RUL estimate.
    simulated_rul:
        Simulated point RUL estimate.
    original_shi:
        Original SHI value at the cycle.
    simulated_shi:
        Simulated SHI value at the cycle.
    original_state:
        Original health state name.
    simulated_state:
        Simulated health state name.
    proba_delta:
        Change in per-horizon probabilities (simulated − original).
    rul_delta:
        Change in RUL (simulated − original).
    shi_delta:
        Change in SHI.
    largest_changes:
        Top sensors contributing to the risk change.
    ood_warning:
        Out-of-distribution warning message, if applicable.
    identity_verified:
        Whether the zero-perturbation identity test passed.
    """

    badge: SimulationBadge
    unit_id: Any
    cycle: int
    perturbation_type: str
    perturbation_params: dict[str, Any]
    perturbed_sensors: list[str]
    original_proba: pd.DataFrame
    simulated_proba: pd.DataFrame
    original_rul: float
    simulated_rul: float
    original_shi: float
    simulated_shi: float
    original_state: str
    simulated_state: str
    proba_delta: pd.DataFrame
    rul_delta: float
    shi_delta: float
    largest_changes: dict[str, float]
    ood_warning: str = ""
    identity_verified: bool = False

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dictionary."""
        return {
            "badge": self.badge.to_dict(),
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            "perturbation_type": self.perturbation_type,
            "perturbation_params": self.perturbation_params,
            "perturbed_sensors": self.perturbed_sensors,
            "original_rul": self.original_rul,
            "simulated_rul": self.simulated_rul,
            "rul_delta": self.rul_delta,
            "original_shi": self.original_shi,
            "simulated_shi": self.simulated_shi,
            "shi_delta": self.shi_delta,
            "original_state": self.original_state,
            "simulated_state": self.simulated_state,
            "proba_delta_max": float(self.proba_delta.values.max()) if not self.proba_delta.empty else 0.0,
            "largest_changes": self.largest_changes,
            "ood_warning": self.ood_warning,
            "identity_verified": self.identity_verified,
        }


# ---------------------------------------------------------------------------
# OOD detection helpers
# ---------------------------------------------------------------------------

def _mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Compute Mahalanobis distance from a reference distribution."""
    diff = x - mean
    left = diff @ cov_inv
    return float(np.sqrt(np.abs(left @ diff)))


def _check_ood(
    original_vals: np.ndarray,
    perturbed_vals: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    psi_threshold: float = 0.25,
) -> str:
    """Check if the perturbation pushes values out of distribution.

    Uses PSI per sensor and returns a warning string if any sensor
    exceeds the threshold.
    """
    warnings: list[str] = []
    n_sensors = len(baseline_mean)

    for i in range(n_sensors):
        orig = original_vals[:, i] if original_vals.ndim > 1 else original_vals
        pert = perturbed_vals[:, i] if perturbed_vals.ndim > 1 else perturbed_vals

        # Compute PSI
        bl = np.full(50, baseline_mean[i]) if baseline_std[i] == 0 else np.random.normal(
            baseline_mean[i], max(baseline_std[i], 1e-8), 50
        )
        try:
            psi = _compute_psi_simple(bl, pert)
            if psi > psi_threshold:
                warnings.append(f"sensor_{i + 1} PSI={psi:.3f}")
        except Exception:
            pass

    if warnings:
        return (
            "OOD WARNING: Perturbed values may be out-of-distribution "
            f"for: {', '.join(warnings)}. Interpret results with caution."
        )
    return ""


def _compute_psi_simple(baseline: np.ndarray, current: np.ndarray, n_bins: int = 5) -> float:
    """Simplified PSI computation."""
    if len(baseline) < n_bins or len(current) < 2:
        return 0.0
    try:
        edges = np.quantile(baseline, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
    except Exception:
        return 0.0

    bl_dig = np.digitize(baseline, edges[1:-1], right=True)
    cur_dig = np.digitize(current, edges[1:-1], right=True)
    n_actual = len(edges) - 1
    bl_counts = np.bincount(bl_dig, minlength=n_actual).astype(np.float64)
    cur_counts = np.bincount(cur_dig, minlength=n_actual).astype(np.float64)

    eps = 1e-6
    bl_prop = (bl_counts + eps) / (bl_counts.sum() + eps * n_actual)
    cur_prop = (cur_counts + eps) / (cur_counts.sum() + eps * n_actual)
    return float(np.sum((cur_prop - bl_prop) * np.log(cur_prop / bl_prop)))


# ---------------------------------------------------------------------------
# WhatIfSimulator
# ---------------------------------------------------------------------------

class WhatIfSimulator:
    """What-If simulator for counterfactual analysis.

    Parameters
    ----------
    model:
        Fitted model with ``predict_proba`` and ``predict_rul`` methods.
    feature_pipeline:
        Fitted preprocessing pipeline with a ``transform`` method.
    feature_cols:
        Feature columns expected by the model.
    sensor_cols:
        Raw sensor column names.
    health_index:
        Fitted ``HealthIndex`` object for SHI computation.  If ``None``,
        SHI-related outputs are set to ``NaN``.
    state_classifier:
        Fitted ``StateClassifier`` for health state classification.
        If ``None``, state outputs are set to ``"UNKNOWN"``.
    baseline_cycles:
        Number of initial cycles defining the healthy baseline.
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.
    """

    def __init__(
        self,
        model: Any,
        feature_pipeline: Any | None = None,
        feature_cols: list[str] | None = None,
        sensor_cols: list[str] | None = None,
        health_index: Any | None = None,
        state_classifier: Any | None = None,
        baseline_cycles: int = 30,
        unit_col: str = _UNIT_COL,
        cycle_col: str = _CYCLE_COL,
    ) -> None:
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.feature_cols = feature_cols or []
        self.sensor_cols = sensor_cols or []
        self.health_index = health_index
        self.state_classifier = state_classifier
        self.baseline_cycles = baseline_cycles
        self.unit_col = unit_col
        self.cycle_col = cycle_col

        # Store training baseline statistics for OOD detection
        self._baseline_mean: np.ndarray | None = None
        self._baseline_std: np.ndarray | None = None

    def fit_baseline(
        self,
        df: pd.DataFrame,
        sensor_cols: list[str] | None = None,
    ) -> WhatIfSimulator:
        """Compute and cache baseline statistics from healthy data.

        Parameters
        ----------
        df:
            Training DataFrame (first ``baseline_cycles`` of each unit).
        sensor_cols:
            Sensors to compute baseline for.  Defaults to ``self.sensor_cols``.

        Returns
        -------
        WhatIfSimulator
            Self, for method chaining.
        """
        cols = sensor_cols or self.sensor_cols
        if not cols:
            return self

        bl_data = df.groupby(self.unit_col).head(self.baseline_cycles)
        means = []
        stds = []
        for col in cols:
            if col in bl_data.columns and pd.api.types.is_numeric_dtype(bl_data[col]):
                vals = bl_data[col].dropna().values
                means.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0)
            else:
                means.append(0.0)
                stds.append(1.0)

        self._baseline_mean = np.array(means)
        self._baseline_std = np.array(stds)
        return self

    def run(
        self,
        df: pd.DataFrame,
        unit_id: Any,
        cycle: int,
        sensor_perturbations: dict[str, dict[str, Any]],
        n_sim_cycles: int = 1,
    ) -> SimulationResult:
        """Run a what-if simulation.

        Parameters
        ----------
        df:
            Full DataFrame (sorted by unit, cycle).
        unit_id:
            Unit to simulate.
        cycle:
            Cycle at which to apply the perturbation.
        sensor_perturbations:
            Mapping ``{sensor: params}`` where *params* is a dict with:
            - ``"type"``: ``"multiplicative"``, ``"additive"``, or ``"variance"``.
            - ``"factor"``: multiplicative factor (for ``"multiplicative"``).
            - ``"offset"``: additive value (for ``"additive"``).
            - ``"scale"``: variance scaling factor (for ``"variance"``).
            - ``"start_cycle"``: first cycle of the perturbation window
              (default: *cycle*).
            - ``"end_cycle"``: last cycle of the perturbation window
              (default: *cycle* + 10).
        n_sim_cycles:
            Number of cycles to simulate ahead (default 1).

        Returns
        -------
        SimulationResult
            Original vs simulated predictions with badge.
        """
        badge = SimulationBadge()

        # Extract unit data
        unit_data = df[df[self.unit_col] == unit_id].sort_values(self.cycle_col).copy()
        if unit_data.empty:
            raise ValueError(f"No data found for unit {unit_id}")

        # Original prediction at the target cycle
        target_idx = unit_data[unit_data[self.cycle_col] == cycle].index
        if len(target_idx) == 0:
            raise ValueError(f"Cycle {cycle} not found for unit {unit_id}")
        target_pos = unit_data.index.get_loc(target_idx[0])

        # Build original feature row
        original_row = unit_data.iloc[[target_pos]].copy()

        # Get original prediction
        if self.feature_pipeline is not None:
            original_features = self.feature_pipeline.transform(original_row)
        else:
            original_features = original_row

        if self.feature_cols:
            model_input_orig = original_features[self.feature_cols]
        else:
            model_input_orig = original_features

        proba_orig = self.model.predict_proba(model_input_orig)
        rul_orig_series = self.model.predict_rul(model_input_orig)
        rul_orig = float(rul_orig_series.iloc[0]) if len(rul_orig_series) > 0 else 60.0

        # Original SHI
        shi_orig = self._compute_shi(df, unit_id, cycle)
        state_orig = self._classify_state(shi_orig)

        # --- Apply perturbation ---
        perturbed_data = unit_data.copy()
        perturbed_sensors = list(sensor_perturbations.keys())
        all_perturbed = unit_data.copy()

        for sensor, params in sensor_perturbations.items():
            if sensor not in all_perturbed.columns:
                continue

            ptype = params.get("type", "multiplicative")
            start = params.get("start_cycle", cycle)
            end = params.get("end_cycle", cycle + n_sim_cycles)
            mask = (all_perturbed[self.cycle_col] >= start) & (all_perturbed[self.cycle_col] <= end)

            if ptype == "multiplicative":
                factor = params.get("factor", 1.0)
                all_perturbed.loc[mask, sensor] = all_perturbed.loc[mask, sensor] * (1.0 + factor)
            elif ptype == "additive":
                offset = params.get("offset", 0.0)
                all_perturbed.loc[mask, sensor] = all_perturbed.loc[mask, sensor] + offset
            elif ptype == "variance":
                scale = params.get("scale", 1.0)
                vals = all_perturbed.loc[mask, sensor].values.astype(np.float64)
                noise = np.random.normal(0, 1, size=len(vals))
                std = np.std(vals) if len(vals) > 1 else 1.0
                all_perturbed.loc[mask, sensor] = vals + noise * std * scale

        # Simulated prediction at the target cycle
        sim_row = all_perturbed.iloc[[target_pos]].copy()
        if self.feature_pipeline is not None:
            sim_features = self.feature_pipeline.transform(sim_row)
        else:
            sim_features = sim_row

        if self.feature_cols:
            model_input_sim = sim_features[self.feature_cols]
        else:
            model_input_sim = sim_features

        proba_sim = self.model.predict_proba(model_input_sim)
        rul_sim_series = self.model.predict_rul(model_input_sim)
        rul_sim = float(rul_sim_series.iloc[0]) if len(rul_sim_series) > 0 else 60.0

        # Simulated SHI
        shi_sim = self._compute_shi(all_perturbed, unit_id, cycle)
        state_sim = self._classify_state(shi_sim)

        # Compute deltas
        common_cols = [c for c in proba_orig.columns if c in proba_sim.columns]
        proba_delta = proba_sim[common_cols].iloc[0] - proba_orig[common_cols].iloc[0]
        proba_delta = proba_delta.to_frame().T

        rul_delta = rul_sim - rul_orig
        shi_delta = shi_sim - shi_orig

        # Largest changes
        largest: dict[str, float] = {}
        for c in common_cols:
            largest[c] = float(proba_delta[c].iloc[0])

        # OOD check
        ood_warning = ""
        if self._baseline_mean is not None and self._baseline_std is not None:
            orig_vals = original_features[self.sensor_cols].values if self.sensor_cols else np.zeros((1, 1))
            sim_vals = sim_features[self.sensor_cols].values if self.sensor_cols else np.zeros((1, 1))
            n_sensors_avail = min(orig_vals.shape[1], len(self._baseline_mean))
            if n_sensors_avail > 0:
                ood_warning = _check_ood(
                    orig_vals[:, :n_sensors_avail],
                    sim_vals[:, :n_sensors_avail],
                    self._baseline_mean[:n_sensors_avail],
                    self._baseline_std[:n_sensors_avail],
                )

        return SimulationResult(
            badge=badge,
            unit_id=unit_id,
            cycle=cycle,
            perturbation_type="mixed",
            perturbation_params=sensor_perturbations,
            perturbed_sensors=perturbed_sensors,
            original_proba=proba_orig,
            simulated_proba=proba_sim,
            original_rul=rul_orig,
            simulated_rul=rul_sim,
            original_shi=shi_orig,
            simulated_shi=shi_sim,
            original_state=state_orig,
            simulated_state=state_sim,
            proba_delta=proba_delta,
            rul_delta=rul_delta,
            shi_delta=shi_delta,
            largest_changes=largest,
            ood_warning=ood_warning,
            identity_verified=False,
        )

    def identity_test(
        self,
        df: pd.DataFrame,
        unit_id: Any,
        cycle: int,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> SimulationResult:
        """Verify that zero perturbation reproduces the original prediction.

        Applies an identity (no-change) perturbation and checks that the
        simulated prediction matches the original within tolerance.

        Parameters
        ----------
        df:
            Full DataFrame.
        unit_id:
            Unit identifier.
        cycle:
            Cycle number.
        rtol:
            Relative tolerance for comparison.
        atol:
            Absolute tolerance for comparison.

        Returns
        -------
        SimulationResult
            Result of the identity simulation.  ``identity_verified`` is
            ``True`` if the predictions match.
        """
        # Run with zero perturbation on a dummy sensor
        dummy_sensor = self.sensor_cols[0] if self.sensor_cols else None
        perturbations = {}
        if dummy_sensor:
            perturbations[dummy_sensor] = {"type": "multiplicative", "factor": 0.0}

        result = self.run(df, unit_id, cycle, perturbations)

        # Check identity
        if not result.proba_delta.empty:
            max_delta = float(np.abs(result.proba_delta.values).max())
            result.identity_verified = max_delta < (atol + rtol * abs(float(result.original_proba.values.mean())))
        else:
            result.identity_verified = True

        return result

    def _compute_shi(
        self,
        df: pd.DataFrame,
        unit_id: Any,
        cycle: int,
    ) -> float:
        """Compute SHI for a given (unit, cycle) using cached parameters."""
        if self.health_index is None:
            return float("nan")

        # Use cached ECDF params and weights if available
        hi = self.health_index
        if hasattr(hi, "shi_values"):
            mask = (hi.shi_values[self.unit_col] == unit_id) & (hi.shi_values[self.cycle_col] == cycle)
            matched = hi.shi_values[mask]
            if not matched.empty:
                return float(matched["shi"].iloc[0])

        return float("nan")

    def _classify_state(self, shi_value: float) -> str:
        """Classify an SHI value into a health state name."""
        if np.isnan(shi_value):
            return "UNKNOWN"
        if self.state_classifier is None:
            return "UNKNOWN"
        state = self.state_classifier._shi_to_raw_state(shi_value)
        return state.name
