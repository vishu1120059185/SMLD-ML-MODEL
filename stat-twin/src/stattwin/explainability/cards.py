"""Evidence cards for top-k contributing sensors.

Each evidence card summarises *why* a sensor contributed to the model's
current risk estimate.  Cards contain:

* **Current value** – most recent reading.
* **Baseline mean / std** – healthy-period statistics.
* **Z-score** – standardised deviation from healthy baseline.
* **Trend %** – signed percentage change relative to baseline.
* **EWMA deviation** – deviation of the exponential weighted moving average
  from the baseline mean.
* **Variance ratio** – ratio of recent rolling variance to baseline variance.
* **Distribution shift (PSI)** – Population Stability Index vs baseline.
* **Severity chip** – categorical label from configurable bins.
* **Sensor chart link** – reference to the sensor's time-series plot.

Cards are sorted by absolute contribution (descending) and the top-k are
returned as an ``EvidenceCardSet``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "EvidenceCard",
    "EvidenceCardSet",
    "build_evidence_cards",
    "severity_label",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"

# Default severity bins: (upper_bound_inclusive, label)
# Lower values = worse health
DEFAULT_SEVERITY_BINS: list[tuple[float, str]] = [
    (1.0, "Nominal"),
    (2.0, "Mild"),
    (3.0, "Moderate"),
    (4.0, "Elevated"),
    (np.inf, "Severe"),
]


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def severity_label(
    z_score: float,
    bins: list[tuple[float, str]] | None = None,
) -> str:
    """Map an absolute z-score to a human-readable severity chip.

    Parameters
    ----------
    z_score:
        Absolute z-score of the current value vs baseline.
    bins:
        List of ``(upper_bound, label)`` tuples sorted ascending by bound.
        Lower z-scores map to milder labels.  Defaults to a 5-bin scheme.

    Returns
    -------
    str
        Severity label such as ``"Nominal"`` or ``"Severe"``.
    """
    if bins is None:
        bins = DEFAULT_SEVERITY_BINS
    abs_z = abs(z_score)
    for bound, label in bins:
        if abs_z <= bound:
            return label
    return bins[-1][1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvidenceCard:
    """Diagnostic summary for a single sensor at a given cycle.

    Attributes
    ----------
    sensor:
        Sensor column name.
    current_value:
        Most recent sensor reading.
    baseline_mean:
        Healthy-period mean.
    baseline_std:
        Healthy-period standard deviation.
    z_score:
        ``(current - baseline_mean) / baseline_std``.
    trend_pct:
        Signed percentage change: ``(current - baseline_mean) / |baseline_mean| * 100``.
    ewma_deviation:
        ``EWMA(alpha) - baseline_mean``, normalised by baseline std.
    variance_ratio:
        Recent rolling variance / baseline variance.
    psi:
        Population Stability Index vs baseline distribution.
    severity:
        Categorical severity chip.
    contribution:
        Absolute contribution to the model's risk estimate (from attribution).
    chart_ref:
        Optional reference path or URL for the sensor's time-series chart.
    """

    sensor: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    trend_pct: float
    ewma_deviation: float
    variance_ratio: float
    psi: float
    severity: str
    contribution: float
    chart_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "sensor": self.sensor,
            "current_value": self.current_value,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "z_score": self.z_score,
            "trend_pct": self.trend_pct,
            "ewma_deviation": self.ewma_deviation,
            "variance_ratio": self.variance_ratio,
            "psi": self.psi,
            "severity": self.severity,
            "contribution": self.contribution,
            "chart_ref": self.chart_ref,
        }


@dataclass
class EvidenceCardSet:
    """Collection of evidence cards for a single (unit, cycle) observation.

    Attributes
    ----------
    unit_id:
        Unit identifier.
    cycle:
        Cycle number.
    cards:
        List of ``EvidenceCard`` objects, sorted by descending contribution.
    top_k:
        Number of cards requested.
    """

    unit_id: Any
    cycle: int
    cards: list[EvidenceCard] = field(default_factory=list)
    top_k: int = 5

    @property
    def sensor_ranking(self) -> list[str]:
        """Return sensor names in order of contribution."""
        return [c.sensor for c in self.cards]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialise all cards to a list of dicts."""
        return [c.to_dict() for c in self.cards]

    def __repr__(self) -> str:
        return (
            f"EvidenceCardSet(unit={self.unit_id}, cycle={self.cycle}, "
            f"top_k={self.top_k}, sensors={self.sensor_ranking})"
        )


# ---------------------------------------------------------------------------
# Statistic helpers (vectorised)
# ---------------------------------------------------------------------------

def _compute_z_score(
    current: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
) -> np.ndarray:
    """Standardised z-score: (current - mean) / std."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(baseline_std > 0, (current - baseline_mean) / baseline_std, 0.0)
    return z


def _compute_trend_pct(
    current: np.ndarray,
    baseline_mean: np.ndarray,
) -> np.ndarray:
    """Signed percentage change from baseline mean."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(np.abs(baseline_mean) > 1e-12,
                        (current - baseline_mean) / np.abs(baseline_mean) * 100.0,
                        0.0)
    return pct


def _compute_ewma(values: np.ndarray, alpha: float = 0.3) -> float:
    """Compute the EWMA of a 1-D array and return the final value."""
    if len(values) == 0:
        return 0.0
    ewma = values[0]
    for v in values[1:]:
        ewma = alpha * v + (1.0 - alpha) * ewma
    return ewma


def _compute_variance_ratio(
    recent_values: np.ndarray,
    baseline_values: np.ndarray,
) -> float:
    """Ratio of recent variance to baseline variance."""
    if len(recent_values) < 2 or len(baseline_values) < 2:
        return 1.0
    recent_var = np.var(recent_values, ddof=1)
    baseline_var = np.var(baseline_values, ddof=1)
    if baseline_var <= 0:
        return 1.0 if recent_var == 0 else np.inf
    return float(recent_var / baseline_var)


def _compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two 1-D samples."""
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
    psi = float(np.sum((cur_prop - bl_prop) * np.log(cur_prop / bl_prop)))
    return psi


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def build_evidence_cards(
    df: pd.DataFrame,
    sensor_cols: list[str],
    contributions: dict[str, float],
    unit_id: Any,
    cycle: int,
    top_k: int = 5,
    baseline_cycles: int = 30,
    ewma_alpha: float = 0.3,
    recent_window: int = 10,
    psi_n_bins: int = 10,
    severity_bins: list[tuple[float, str]] | None = None,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> EvidenceCardSet:
    """Build evidence cards for the top-k contributing sensors.

    Parameters
    ----------
    df:
        Full DataFrame (sorted by unit, cycle) containing raw or
        preprocessed sensor values.
    sensor_cols:
        Sensor columns to consider.
    contributions:
        Mapping ``{sensor_col: absolute_contribution}`` from an
        attribution method (e.g. group occlusion).
    unit_id:
        Unit identifier for the observation of interest.
    cycle:
        Cycle number for the observation of interest.
    top_k:
        Number of top contributing sensors to include.
    baseline_cycles:
        Number of initial cycles defining the healthy baseline.
    ewma_alpha:
        Smoothing factor for EWMA computation.
    recent_window:
        Number of recent cycles used for variance ratio and PSI.
    psi_n_bins:
        Number of bins for PSI computation.
    severity_bins:
        Custom severity bin definitions.  If ``None``, the module
        default is used.
    unit_col:
        Unit identifier column name.
    cycle_col:
        Cycle identifier column name.

    Returns
    -------
    EvidenceCardSet
        Cards for the top-k contributing sensors at the specified
        (unit_id, cycle).
    """
    # Sort sensors by absolute contribution descending
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_sensors = [s for s, _ in ranked[:top_k]]

    # Extract unit data
    unit_mask = df[unit_col] == unit_id
    cycle_mask = df[cycle_col] <= cycle
    unit_data = df[unit_mask & cycle_mask].sort_values(cycle_col)

    if unit_data.empty:
        return EvidenceCardSet(unit_id=unit_id, cycle=cycle, top_k=top_k)

    # Get current row index (last row at or before target cycle)
    current_row = unit_data.iloc[-1]
    current_cycle_actual = int(current_row[cycle_col])

    # Baseline: first baseline_cycles of this unit
    all_unit_data = df[unit_mask].sort_values(cycle_col)
    baseline_data = all_unit_data.head(baseline_cycles)

    # Recent window: last ``recent_window`` rows up to current cycle
    recent_data = unit_data.tail(recent_window)

    cards: list[EvidenceCard] = []
    for sensor in top_sensors:
        if sensor not in current_row.index or not pd.api.types.is_numeric_dtype(df[sensor]):
            continue

        current_val = float(current_row[sensor])

        # Baseline statistics
        bl_vals = baseline_data[sensor].dropna().values.astype(np.float64)
        bl_mean = float(np.mean(bl_vals)) if len(bl_vals) > 0 else 0.0
        bl_std = float(np.std(bl_vals, ddof=1)) if len(bl_vals) > 1 else 1.0

        # Z-score
        z = float((current_val - bl_mean) / bl_std) if bl_std > 0 else 0.0

        # Trend %
        with np.errstate(divide="ignore", invalid="ignore"):
            trend = ((current_val - bl_mean) / abs(bl_mean) * 100.0
                     if abs(bl_mean) > 1e-12 else 0.0)

        # EWMA deviation
        recent_vals = recent_data[sensor].dropna().values.astype(np.float64)
        ewma_val = _compute_ewma(recent_vals, alpha=ewma_alpha)
        ewma_dev = (ewma_val - bl_mean) / bl_std if bl_std > 0 else 0.0

        # Variance ratio
        var_ratio = _compute_variance_ratio(recent_vals, bl_vals)

        # PSI
        psi = _compute_psi(bl_vals, recent_vals, n_bins=psi_n_bins)

        # Severity
        sev = severity_label(z, bins=severity_bins)

        # Contribution from the attribution map
        contrib = contributions.get(sensor, 0.0)

        # Chart reference (convention: sensor name is the identifier)
        chart_ref = sensor

        cards.append(EvidenceCard(
            sensor=sensor,
            current_value=current_val,
            baseline_mean=bl_mean,
            baseline_std=bl_std,
            z_score=round(z, 4),
            trend_pct=round(trend, 4),
            ewma_deviation=round(ewma_dev, 4),
            variance_ratio=round(var_ratio, 4),
            psi=round(psi, 6),
            severity=sev,
            contribution=round(abs(contrib), 6),
            chart_ref=chart_ref,
        ))

    # Sort by absolute contribution descending
    cards.sort(key=lambda c: abs(c.contribution), reverse=True)

    return EvidenceCardSet(
        unit_id=unit_id,
        cycle=cycle,
        cards=cards,
        top_k=top_k,
    )
