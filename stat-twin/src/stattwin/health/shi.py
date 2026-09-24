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
from dataclasses import dataclass
from typing import Any

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
    baseline_mean: float,
    baseline_std: float,
    deg_sign: int,
    sigma_min: float = 1e-6,
) -> np.ndarray:
    std = max(baseline_std, sigma_min)
    raw = deg_sign * (values - baseline_mean) / std
    return np.maximum(raw, 0.0)


def _compute_trend(
    values: np.ndarray,
    baseline_std: float,
    deg_sign: int,
    window: int = 10,
    sigma_min: float = 1e-6,
) -> np.ndarray:
    n = len(values)
    result = np.zeros(n, dtype=np.float64)
    std = max(baseline_std, sigma_min)
    for i in range(n):
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        if len(w) < 3:
            result[i] = 0.0
            continue
        x = np.arange(len(w), dtype=np.float64)
        slope = np.polyfit(x, w, 1)[0]
        result[i] = max(deg_sign * slope * 10.0 / std, 0.0)
    return result


def _compute_ewma(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _compute_variance(
    values: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    n = len(values)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        if len(w) < 2:
            result[i] = 0.0
            continue
        result[i] = np.log(max(np.std(w), 1e-10))
    return result


def _compute_corr_shift(
    values: np.ndarray,
    baseline_values: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    n = len(values)
    result = np.zeros(n, dtype=np.float64)
    bl_mean = np.mean(baseline_values) if len(baseline_values) > 0 else 0.0
    bl_std = max(np.std(baseline_values), 1e-10) if len(baseline_values) > 1 else 1.0
    for i in range(n):
        start = max(0, i - window + 1)
        w = values[start:i + 1]
        if len(w) < 3:
            result[i] = 0.0
            continue
        bl_norm = (baseline_values - bl_mean) / bl_std
        w_norm = (w - np.mean(w)) / max(np.std(w), 1e-10)
        rho, _ = sp_stats.spearmanr(bl_norm[:len(w)], w_norm)
        result[i] = abs(rho) if not np.isnan(rho) else 0.0
    return result


def _fit_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    probs = np.arange(1, n + 1) / n
    return sorted_vals, probs


def _ecdf_transform(
    values: np.ndarray,
    ecdf_vals: np.ndarray,
    ecdf_probs: np.ndarray,
    clamp_low: float = 0.05,
    clamp_high: float = 0.99,
) -> np.ndarray:
    indices = np.searchsorted(ecdf_vals, values, side="right")
    indices = np.clip(indices, 0, len(ecdf_probs) - 1)
    result = ecdf_probs[indices]
    result = np.clip(result, clamp_low, clamp_high)
    return result


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvidenceComponent:
    name: str
    weight: float = 1.0


@dataclass
class HealthIndex:
    shi_values: pd.DataFrame
    evidence_components: dict[str, pd.DataFrame]
    weights: dict[str, float]
    sensor_weights: dict[str, float]
    ecdf_params: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    baseline_stats: dict[str, dict[str, Any]]
    degradation_directions: dict[str, int]


# ---------------------------------------------------------------------------
# Vectorized SHI computation
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
    """Compute the Statistical Health Index for all units (vectorized)."""

    if sensor_cols is None:
        exclude = {unit_col, cycle_col, "RUL"}
        sensor_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not sensor_cols:
        raise ValueError("No sensor columns found")

    if evidence_components is None:
        evidence_components = [
            EvidenceComponent(name="deviation", weight=0.25),
            EvidenceComponent(name="trend", weight=0.25),
            EvidenceComponent(name="ewma", weight=0.20),
            EvidenceComponent(name="variance", weight=0.15),
            EvidenceComponent(name="corr_shift", weight=0.15),
        ]

    evidence_components = [
        ec if isinstance(ec, EvidenceComponent) else EvidenceComponent(name=ec)
        for ec in evidence_components
    ]

    weights = {ec.name: ec.weight for ec in evidence_components}
    if normalise_weights:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    df_sorted = df.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    units = df_sorted[unit_col].unique()

    # Step 1: Degradation directions (vectorized per sensor)
    degradation_directions: dict[str, int] = {}
    for col in sensor_cols:
        def _unit_spearman(group, _col: str = col):
            if len(group) < 5:
                return np.nan
            rho, _ = sp_stats.spearmanr(group[cycle_col].values, group[_col].values)
            return rho
        rhos = df_sorted.groupby(unit_col).apply(_unit_spearman, include_groups=False)
        rhos = rhos.dropna()
        median_rho = rhos.median() if len(rhos) > 0 else 0.0
        degradation_directions[col] = 1 if median_rho >= 0 else -1

    # Step 2: Baseline stats (vectorized)
    baseline_stats: dict[str, dict[str, Any]] = {}
    for col in sensor_cols:
        bl_means = []
        bl_stds = []
        for unit in units:
            unit_data = df_sorted[df_sorted[unit_col] == unit].head(baseline_cycles)
            if len(unit_data) >= 2:
                bl_means.append(unit_data[col].mean())
                bl_stds.append(unit_data[col].std())
        baseline_stats[col] = {
            "mean": np.mean(bl_means) if bl_means else 0.0,
            "std": np.mean(bl_stds) if bl_stds else 1.0,
        }

    # Step 3: Sensor informativeness weights (vectorized)
    sensor_weights: dict[str, float] = {}
    for col in sensor_cols:
        def _unit_rho_rul(group, _col: str = col):
            if len(group) < 5 or "RUL" not in group.columns:
                return np.nan
            rho, _ = sp_stats.spearmanr(group["RUL"].values, group[_col].values)
            return abs(rho) if not np.isnan(rho) else np.nan
        rhos = df_sorted.groupby(unit_col).apply(_unit_rho_rul, include_groups=False)
        rhos = rhos.dropna()
        sensor_weights[col] = rhos.mean() if len(rhos) > 0 else 0.0

    total_sw = sum(sensor_weights.values())
    if total_sw > 0:
        sensor_weights = {k: v / total_sw for k, v in sensor_weights.items()}

    # Step 4: Compute raw evidence per sensor per unit (with EWMA vectorization)
    raw_evidence: dict[str, dict[str, dict[str, np.ndarray]]] = {
        ec.name: {} for ec in evidence_components
    }

    for col in sensor_cols:
        bl = baseline_stats[col]
        bl_mean = bl["mean"]
        bl_std = bl["std"] if bl["std"] > 0 else 1.0
        deg_sign = degradation_directions[col]

        for unit in units:
            unit_data = df_sorted[df_sorted[unit_col] == unit].reset_index(drop=True)
            values = unit_data[col].values.astype(np.float64)

            for ec in evidence_components:
                if ec.name == "deviation":
                    raw = _compute_deviation(values, bl_mean, bl_std, deg_sign)
                elif ec.name == "trend":
                    raw = _compute_trend(values, bl_std, deg_sign)
                elif ec.name == "ewma":
                    raw = _compute_ewma(values, alpha=ewma_alpha)
                elif ec.name == "variance" or ec.name == "corr_shift":
                    raw = _compute_variance(values, window=variance_window)
                else:
                    raw = np.zeros_like(values)

                if unit not in raw_evidence[ec.name]:
                    raw_evidence[ec.name][unit] = {}
                raw_evidence[ec.name][unit][col] = raw

    # Step 5: ECDF normalization
    ecdf_params: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    normalised_evidence: dict[str, dict[str, dict[str, np.ndarray]]] = {
        ec.name: {} for ec in evidence_components
    }

    for ec in evidence_components:
        for col in sensor_cols:
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

    for ec in evidence_components:
        for unit in units:
            if unit not in normalised_evidence[ec.name]:
                normalised_evidence[ec.name][unit] = {}
            for col in sensor_cols:
                if (ec.name, col) in ecdf_params and unit in raw_evidence[ec.name]:
                    ecdf_vals, ecdf_probs = ecdf_params[(ec.name, col)]
                    raw = raw_evidence[ec.name][unit][col]
                    norm = _ecdf_transform(raw, ecdf_vals, ecdf_probs, ecdf_clamp_low, ecdf_clamp_high)  # noqa: E501
                    normalised_evidence[ec.name][unit][col] = norm
                else:
                    n = len(df_sorted[df_sorted[unit_col] == unit])
                    normalised_evidence[ec.name][unit][col] = np.full(n, 0.5)

    # Step 6: Aggregate and compute SHI
    shi_data = []
    evidence_data = {ec.name: [] for ec in evidence_components}

    for unit in units:
        unit_data = df_sorted[df_sorted[unit_col] == unit]
        n_cycles = len(unit_data)
        shi_unit = np.zeros(n_cycles, dtype=np.float64)
        evidence_unit = {ec.name: np.zeros(n_cycles, dtype=np.float64) for ec in evidence_components}  # noqa: E501

        for ec in evidence_components:
            weighted_sum = np.zeros(n_cycles, dtype=np.float64)
            weight_sum = 0.0
            for col in sensor_cols:
                omega = sensor_weights.get(col, 0.0)
                if omega > 0 and col in normalised_evidence[ec.name].get(unit, {}):
                    weighted_sum += omega * normalised_evidence[ec.name][unit][col]
                    weight_sum += omega

            aggregated = weighted_sum / weight_sum if weight_sum > 0 else np.full(n_cycles, 0.5)

            evidence_unit[ec.name] = aggregated
            shi_unit += weights[ec.name] * aggregated

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


def compute_shi_simple(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    weights: dict[str, float] | None = None,
    baseline_cycles: int = 30,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> pd.DataFrame:
    """Simplified SHI computation returning only the SHI DataFrame."""
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
