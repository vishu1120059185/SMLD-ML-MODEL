"""Statistical Health Index (SHI) for STAT-TWIN.

Computes a composite degradation score in [0, 100] from multiple
evidence components.  Each component is normalised to [0, 1] using a
train-fitted Empirical CDF (ECDF), and sensors are aggregated with
informativeness weights.

SHI formula
-----------
SHI_t = 100 * (1 - sum_k(lambda_k * c_k,t))

where:
  - lambda_k are evidence-component weights (sum to 1)
  - c_k,t is the aggregated evidence for component k at cycle t
  - c_k,t = sum_i(omega_i * e_k,i,t) / sum_i(omega_i)
  - omega_i are train-fitted sensor informativeness weights

Degradation direction is determined by the sign of Spearman(sensor_i, cycle)
on training data.  The healthy baseline is the first K cycles of each unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

__all__ = [
    "EvidenceComponent",
    "HealthIndex",
    "compute_shi",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"
_DEFAULT_EVIDENCE_COMPONENTS = ("deviation", "trend", "ewma", "variance", "corr_shift")


# ---------------------------------------------------------------------------
# Evidence component computation helpers
# ---------------------------------------------------------------------------

def _compute_deviation(
    values: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    degradation_sign: np.ndarray,
) -> np.ndarray:
    """Standardised absolute deviation from healthy baseline mean.

    deviation_i,t = |x_i,t - mu_i| / sigma_i

    The sign is aligned with degradation direction so that higher
    values always indicate worse health.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.abs(values - baseline_mean) / np.where(baseline_std > 0, baseline_std, 1.0)
    # Align direction: if degradation_sign < 0, lower values are worse
    return raw


def _compute_trend(
    values: np.ndarray,
    baseline_mean: np.ndarray,
    degradation_sign: np.ndarray,
) -> np.ndarray:
    """Signed deviation from baseline mean aligned with degradation direction.

    trend_i,t = (x_i,t - mu_i) * sign(degradation)
    """
    return (values - baseline_mean) * degradation_sign


def _compute_ewma(
    values: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    """Exponentially weighted moving average of sensor values.

    EWMA_t = alpha * x_t + (1-alpha) * EWMA_{t-1}
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result[0] = values[0]
    for t in range(1, n):
        result[t] = alpha * values[t] + (1.0 - alpha) * result[t - 1]
    return result


def _compute_variance(
    values: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Rolling variance of sensor values.

    var_i,t = Var(x_i,t-w+1, ..., x_i,t)
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.var(values[start:i + 1], ddof=1) if (i - start + 1) >= 2 else 0.0
    return result


def _compute_corr_shift(
    values: np.ndarray,
    baseline_values: np.ndarray,
) -> np.ndarray:
    """Correlation shift: |1 - rho(current, baseline)|.

    Measures how much the correlation structure has shifted from baseline.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    if len(baseline_values) < 3:
        return result

    for i in range(n):
        win_start = max(0, i - len(baseline_values) + 1)
        current = values[win_start:i + 1]
        if len(current) < 3:
            continue
        # Use correlation distance
        rho, _ = sp_stats.spearmanr(baseline_values[:len(current)], current)
        if np.isnan(rho):
            continue
        result[i] = 1.0 - abs(rho)
    return result


# ---------------------------------------------------------------------------
# ECDF normalisation
# ---------------------------------------------------------------------------

def _fit_ecdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit an empirical CDF and return (sorted values, cumulative probs)."""
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    cum_probs = np.arange(1, n + 1, dtype=np.float64) / n
    return sorted_vals, cum_probs


def _ecdf_transform(
    values: np.ndarray,
    ecdf_vals: np.ndarray,
    ecdf_probs: np.ndarray,
    clamp_low: float = 0.05,
    clamp_high: float = 0.99,
) -> np.ndarray:
    """Map values to [0, 1] using fitted ECDF with percentile clamping."""
    # Interpolate to get percentile rank
    percentiles = np.interp(values, ecdf_vals, ecdf_probs)
    # Clamp to [clamp_low, clamp_high]
    percentiles = np.clip(percentiles, clamp_low, clamp_high)
    # Map to [0, 1]
    return (percentiles - clamp_low) / (clamp_high - clamp_low)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvidenceComponent:
    """Configuration for a single evidence component.

    Parameters
    ----------
    name:
        Component name (must be one of the supported types).
    weight:
        Weight lambda_k in the SHI formula.
    **kwargs:
        Additional parameters passed to the computation function.
    """

    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = set(_DEFAULT_EVIDENCE_COMPONENTS)
        if self.name not in valid:
            raise ValueError(
                f"Unknown evidence component '{self.name}'; use one of {sorted(valid)}"
            )


@dataclass
class HealthIndex:
    """Container for SHI computation results and fitted parameters.

    Attributes
    ----------
    shi_values:
        SHI time series per unit.
    evidence_components:
        Raw evidence component values before aggregation.
    weights:
        Evidence-component weights lambda_k.
    sensor_weights:
        Sensor informativeness weights omega_i.
    ecdf_params:
        Fitted ECDF parameters per (sensor, component) pair.
    baseline_stats:
        Healthy baseline statistics per sensor.
    degradation_directions:
        +1 or -1 per sensor, indicating degradation direction.
    """

    shi_values: pd.DataFrame
    evidence_components: dict[str, pd.DataFrame]
    weights: dict[str, float]
    sensor_weights: dict[str, float]
    ecdf_params: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    baseline_stats: dict[str, dict[str, float]]
    degradation_directions: dict[str, int]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_shi(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    evidence_components: Sequence[EvidenceComponent | str] | None = None,
    baseline_cycles: int = 30,
    ecdf_clamp_low: float = 0.05,
    ecdf_clamp_high: float = 0.99,
    ewma_alpha: float = 0.2,
    variance_window: int = 10,
    normalise_weights: bool = True,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> HealthIndex:
    """Compute the Statistical Health Index for all units.

    Parameters
    ----------
    df:
        Input DataFrame with columns: unit_col, cycle_col, sensor_cols.
    sensor_cols:
        Sensor columns to include. If None, all numeric columns except
        identifiers are used.
    evidence_components:
        Evidence components and their weights. If None, default components
        with equal weights are used.
    baseline_cycles:
        Number of initial cycles per unit defining the healthy baseline.
    ecdf_clamp_low:
        Lower percentile clamp for ECDF normalisation (default 5th).
    ecdf_clamp_high:
        Upper percentile clamp for ECDF normalisation (default 99th).
    ewma_alpha:
        Smoothing parameter for EWMA component.
    variance_window:
        Rolling window for variance component.
    normalise_weights:
        If True, normalise evidence-component weights to sum to 1.
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.

    Returns
    -------
    HealthIndex
        Container with SHI values and fitted parameters.
    """
    # Auto-detect sensor columns
    if sensor_cols is None:
        exclude = {unit_col, cycle_col, "RUL"}
        sensor_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not sensor_cols:
        raise ValueError("No sensor columns found")

    # Default evidence components
    if evidence_components is None:
        evidence_components = [
            EvidenceComponent(name="deviation", weight=0.25),
            EvidenceComponent(name="trend", weight=0.25),
            EvidenceComponent(name="ewma", weight=0.20),
            EvidenceComponent(name="variance", weight=0.15),
            EvidenceComponent(name="corr_shift", weight=0.15),
        ]

    # Convert string names to EvidenceComponent objects
    evidence_components = [
        ec if isinstance(ec, EvidenceComponent) else EvidenceComponent(name=ec)
        for ec in evidence_components
    ]

    # Normalise weights
    weights = {ec.name: ec.weight for ec in evidence_components}
    if normalise_weights:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    # Sort data
    df_sorted = df.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    units = df_sorted[unit_col].unique()

    # ---- Step 1: Compute degradation directions on training data ----
    degradation_directions: dict[str, int] = {}
    for col in sensor_cols:
        all_slopes = []
        for unit in units:
            unit_data = df_sorted[df_sorted[unit_col] == unit]
            if len(unit_data) < 5:
                continue
            rho, _ = sp_stats.spearmanr(unit_data[cycle_col].values, unit_data[col].values)
            if not np.isnan(rho):
                all_slopes.append(rho)
        # Median slope across units
        median_rho = np.median(all_slopes) if all_slopes else 0.0
        degradation_directions[col] = 1 if median_rho >= 0 else -1

    # ---- Step 2: Compute healthy baseline statistics ----
    baseline_stats: dict[str, dict[str, float]] = {}
    for col in sensor_cols:
        bl_means = []
        bl_stds = []
        bl_values = []
        for unit in units:
            unit_data = df_sorted[df_sorted[unit_col] == unit].head(baseline_cycles)
            if len(unit_data) >= 2:
                bl_means.append(unit_data[col].mean())
                bl_stds.append(unit_data[col].std())
                bl_values.append(unit_data[col].values)
        baseline_stats[col] = {
            "mean": np.mean(bl_means) if bl_means else 0.0,
            "std": np.mean(bl_stds) if bl_stds else 1.0,
            "values": np.concatenate(bl_values) if bl_values else np.array([]),
        }

    # ---- Step 3: Compute sensor informativeness weights ----
    sensor_weights: dict[str, float] = {}
    for col in sensor_cols:
        # Informativeness = |Spearman(sensor, RUL)| on training data
        if "RUL" in df.columns:
            all_rho = []
            for unit in units:
                unit_data = df_sorted[df_sorted[unit_col] == unit]
                if len(unit_data) < 5:
                    continue
                rho, _ = sp_stats.spearmanr(unit_data["RUL"].values, unit_data[col].values)
                if not np.isnan(rho):
                    all_rho.append(abs(rho))
            sensor_weights[col] = np.mean(all_rho) if all_rho else 0.0
        else:
            # Fallback: use inverse of coefficient of variation
            bl = baseline_stats[col]
            if bl["std"] > 0:
                sensor_weights[col] = abs(bl["mean"]) / bl["std"]
            else:
                sensor_weights[col] = 1.0

    # Normalise sensor weights
    total_sw = sum(sensor_weights.values())
    if total_sw > 0:
        sensor_weights = {k: v / total_sw for k, v in sensor_weights.items()}

    # ---- Step 4: Compute raw evidence components per sensor ----
    raw_evidence: dict[str, dict[str, np.ndarray]] = {
        ec.name: {} for ec in evidence_components
    }

    for col in sensor_cols:
        bl = baseline_stats[col]
        bl_mean = bl["mean"]
        bl_std = bl["std"] if bl["std"] > 0 else 1.0
        bl_values = bl["values"]
        deg_sign = degradation_directions[col]

        for unit in units:
            unit_data = df_sorted[df_sorted[unit_col] == unit].reset_index(drop=True)
            values = unit_data[col].values.astype(np.float64)

            for ec in evidence_components:
                if ec.name == "deviation":
                    raw = _compute_deviation(values, bl_mean, bl_std, deg_sign)
                elif ec.name == "trend":
                    raw = _compute_trend(values, bl_mean, deg_sign)
                elif ec.name == "ewma":
                    raw = _compute_ewma(values, alpha=ewma_alpha)
                elif ec.name == "variance":
                    raw = _compute_variance(values, window=variance_window)
                elif ec.name == "corr_shift":
                    raw = _compute_corr_shift(values, bl_values)
                else:
                    raw = np.zeros_like(values)

                if unit not in raw_evidence[ec.name]:
                    raw_evidence[ec.name][unit] = {}
                raw_evidence[ec.name][unit][col] = raw

    # ---- Step 5: Fit ECDFs on training data and normalise ----
    ecdf_params: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    normalised_evidence: dict[str, dict[str, dict[str, np.ndarray]]] = {
        ec.name: {} for ec in evidence_components
    }

    for ec in evidence_components:
        for col in sensor_cols:
            # Collect all training values for this (component, sensor)
            all_train_values = []
            for unit in units:
                if unit in raw_evidence[ec.name] and col in raw_evidence[ec.name][unit]:
                    all_train_values.append(raw_evidence[ec.name][unit][col])

            if all_train_values:
                all_train_arr = np.concatenate(all_train_values)
                all_train_arr = all_train_arr[~np.isnan(all_train_arr)]
                if len(all_train_arr) > 0:
                    ecdf_vals, ecdf_probs = _fit_ecdf(all_train_arr)
                    ecdf_params[(ec.name, col)] = (ecdf_vals, ecdf_probs)

    # Apply ECDF normalisation
    for ec in evidence_components:
        for unit in units:
            if unit not in normalised_evidence[ec.name]:
                normalised_evidence[ec.name][unit] = {}
            for col in sensor_cols:
                if (ec.name, col) in ecdf_params and unit in raw_evidence[ec.name]:
                    ecdf_vals, ecdf_probs = ecdf_params[(ec.name, col)]
                    raw = raw_evidence[ec.name][unit][col]
                    norm = _ecdf_transform(raw, ecdf_vals, ecdf_probs, ecdf_clamp_low, ecdf_clamp_high)
                    normalised_evidence[ec.name][unit][col] = norm
                else:
                    # Default to 0.5 (neutral) if no ECDF fitted
                    n = len(df_sorted[df_sorted[unit_col] == unit])
                    normalised_evidence[ec.name][unit][col] = np.full(n, 0.5)

    # ---- Step 6: Aggregate sensors and compute SHI ----
    shi_data = []
    evidence_data = {ec.name: [] for ec in evidence_components}

    for unit in units:
        unit_data = df_sorted[df_sorted[unit_col] == unit]
        n_cycles = len(unit_data)
        shi_unit = np.zeros(n_cycles, dtype=np.float64)
        evidence_unit = {ec.name: np.zeros(n_cycles, dtype=np.float64) for ec in evidence_components}

        for ec in evidence_components:
            # Aggregate across sensors: c_k = sum(omega_i * e_k,i) / sum(omega_i)
            weighted_sum = np.zeros(n_cycles, dtype=np.float64)
            weight_sum = 0.0
            for col in sensor_cols:
                omega = sensor_weights.get(col, 0.0)
                if omega > 0 and col in normalised_evidence[ec.name].get(unit, {}):
                    weighted_sum += omega * normalised_evidence[ec.name][unit][col]
                    weight_sum += omega

            if weight_sum > 0:
                aggregated = weighted_sum / weight_sum
            else:
                aggregated = np.full(n_cycles, 0.5)

            evidence_unit[ec.name] = aggregated
            shi_unit += weights[ec.name] * aggregated

        # SHI = 100 * (1 - sum_k(lambda_k * c_k))
        shi_values = 100.0 * (1.0 - shi_unit)
        shi_values = np.clip(shi_values, 0.0, 100.0)

        unit_result = unit_data[[unit_col, cycle_col]].copy()
        unit_result["shi"] = shi_values
        shi_data.append(unit_result)

        for ec in evidence_components:
            ev_df = unit_data[[unit_col, cycle_col]].copy()
            ev_df[f"evidence_{ec.name}"] = evidence_unit[ec.name]
            evidence_data[ec.name].append(ev_df)

    shi_df = pd.concat(shi_data, ignore_index=True)
    evidence_dfs = {
        name: pd.concat(dfs, ignore_index=True) for name, dfs in evidence_data.items()
    }

    return HealthIndex(
        shi_values=shi_df,
        evidence_components=evidence_dfs,
        weights=weights,
        sensor_weights=sensor_weights,
        ecdf_params=ecdf_params,
        baseline_stats=baseline_stats,
        degradation_directions=degradation_directions,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_shi_simple(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    weights: dict[str, float] | None = None,
    baseline_cycles: int = 30,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> pd.DataFrame:
    """Simplified SHI computation returning only the SHI DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    sensor_cols:
        Sensor columns to use.
    weights:
        Evidence-component weights. If None, defaults are used.
    baseline_cycles:
        Number of initial cycles for healthy baseline.
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: unit_col, cycle_col, shi.
    """
    if weights is not None:
        components = [
            EvidenceComponent(name=name, weight=wt)
            for name, wt in weights.items()
        ]
    else:
        components = None

    result = compute_shi(
        df=df,
        sensor_cols=sensor_cols,
        evidence_components=components,
        baseline_cycles=baseline_cycles,
        unit_col=unit_col,
        cycle_col=cycle_col,
    )
    return result.shi_values
