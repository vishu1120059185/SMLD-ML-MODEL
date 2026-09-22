"""Health index quality metrics for STAT-TWIN.

Evaluates how well a health index tracks degradation using four metrics:

* **Monotonicity** – fraction of consecutive differences with consistent sign.
* **Trendability** – fraction of unit pairs with consistent monotonic trend.
* **Prognosability** – separability between start-of-life and end-of-life values.
* **Spearman correlation with RUL** – rank correlation between health index and RUL.

All metrics are computed per unit and aggregated (mean ± std).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

__all__=[
    "compute_quality_metrics",
    "monotonicity",
    "prognosability",
    "spearman_rul_correlation",
    "trendability",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ---------------------------------------------------------------------------
# Per-unit metrics
# ---------------------------------------------------------------------------

def monotonicity(series: pd.Series) -> float:
    """Compute monotonicity of a health index time series.

    Monotonicity = (|n_inc| + |n_dec|) / (n_inc + n_dec)

    where n_inc is the number of positive consecutive differences and
    n_dec is the number of negative consecutive differences.

    A value of 1.0 indicates perfectly monotonic behaviour.

    Parameters
    ----------
    series:
        Health index time series (e.g., SHI for one unit).

    Returns
    -------
    float
        Monotonicity score in [0, 1].
    """
    valid = series.dropna().values
    if len(valid) < 3:
        return np.nan

    diffs = np.diff(valid)
    n_inc = np.sum(diffs > 0)
    n_dec = np.sum(diffs < 0)
    total = n_inc + n_dec

    if total == 0:
        return 0.0

    return float((np.abs(n_inc) + np.abs(n_dec)) / total)


def trendability(
    shi_df: pd.DataFrame,
    unit_col: str = _UNIT_COL,
    shi_col: str = "shi",
) -> float:
    """Compute trendability across units.

    Trendability measures whether different units follow similar
    monotonic trends.  For each pair of units, compute the Spearman
    correlation of their health index values aligned by normalised time
    (cycle / max_cycle).

    trendability = fraction of unit pairs with significant positive
    correlation (rho > 0, p < 0.05).

    Parameters
    ----------
    shi_df:
        DataFrame with columns: unit_col, shi_col.
    unit_col:
        Unit identifier column.
    shi_col:
        Column name for health index values.

    Returns
    -------
    float
        Trendability score in [0, 1].
    """
    units = shi_df[unit_col].unique()
    n_units = len(units)

    if n_units < 2:
        return np.nan

    # Normalise time for each unit
    unit_series: dict[Any, np.ndarray] = {}
    for unit in units:
        unit_data = shi_df[shi_df[unit_col] == unit].sort_values(_CYCLE_COL)
        max_cycle = unit_data[_CYCLE_COL].max()
        if max_cycle > 0:
            normalised_time = unit_data[_CYCLE_COL].values / max_cycle
        else:
            normalised_time = unit_data[_CYCLE_COL].values
        # Interpolate to common grid (0, 0.01, ..., 1.0)
        common_grid = np.linspace(0, 1, 101)
        try:
            interp_vals = np.interp(common_grid, normalised_time, unit_data[shi_col].values)
            unit_series[unit] = interp_vals
        except Exception:
            continue

    if len(unit_series) < 2:
        return np.nan

    # Compute pairwise Spearman correlations
    unit_ids = list(unit_series.keys())
    n_pairs = 0
    n_concordant = 0

    for i in range(len(unit_ids)):
        for j in range(i + 1, len(unit_ids)):
            rho, p_val = sp_stats.spearmanr(unit_series[unit_ids[i]], unit_series[unit_ids[j]])
            if not np.isnan(rho):
                n_pairs += 1
                if rho > 0 and p_val < 0.05:
                    n_concordant += 1

    if n_pairs == 0:
        return np.nan

    return n_concordant / n_pairs


def prognosability(
    shi_df: pd.DataFrame,
    unit_col: str = _UNIT_COL,
    shi_col: str = "shi",
    start_pct: float = 0.1,
    end_pct: float = 0.9,
) -> float:
    """Compute prognosability of the health index.

    Prognosability measures how well the health index separates
    healthy (start-of-life) from degraded (end-of-life) states.

    prognosability = |mean_start - mean_end| / (std_start + std_end)

    Higher values indicate better separability.

    Parameters
    ----------
    shi_df:
        DataFrame with columns: unit_col, _CYCLE_COL, shi_col.
    unit_col:
        Unit identifier column.
    shi_col:
        Column name for health index values.
    start_pct:
        Percentile of cycles defining "start" (e.g., 0.1 = first 10%).
    end_pct:
        Percentile of cycles defining "end" (e.g., 0.9 = last 10%).

    Returns
    -------
    float
        Prognosability score (higher is better).
    """
    all_start = []
    all_end = []

    for _, group in shi_df.groupby(unit_col):
        group_sorted = group.sort_values(_CYCLE_COL)
        n = len(group_sorted)
        if n < 10:
            continue

        start_idx = int(n * start_pct)
        end_idx = int(n * end_pct)

        start_vals = group_sorted[shi_col].iloc[:max(start_idx, 1)]
        end_vals = group_sorted[shi_col].iloc[end_idx:]

        all_start.extend(start_vals.dropna().values)
        all_end.extend(end_vals.dropna().values)

    if not all_start or not all_end:
        return np.nan

    mean_start = np.mean(all_start)
    mean_end = np.mean(all_end)
    std_start = np.std(all_start, ddof=1) if len(all_start) > 1 else 0.0
    std_end = np.std(all_end, ddof=1) if len(all_end) > 1 else 0.0

    denominator = std_start + std_end
    if denominator == 0:
        return np.nan

    return float(np.abs(mean_start - mean_end) / denominator)


def spearman_rul_correlation(
    shi_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
    shi_col: str = "shi",
) -> pd.DataFrame:
    """Compute Spearman correlation between SHI and RUL per unit.

    Parameters
    ----------
    shi_df:
        DataFrame with columns: unit_col, cycle_col, shi_col.
    rul_df:
        DataFrame with columns: unit_col, cycle_col, "RUL".
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.
    shi_col:
        Column name for SHI values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: unit_id, spearman_rho, p_value.
    """
    merged = shi_df.merge(
        rul_df[[unit_col, cycle_col, "RUL"]],
        on=[unit_col, cycle_col],
        how="inner",
    )

    results = []
    for unit, group in merged.groupby(unit_col):
        shi_vals = group[shi_col].dropna()
        rul_vals = group["RUL"].dropna()

        # Align on common indices
        common_idx = shi_vals.index.intersection(rul_vals.index)
        if len(common_idx) < 5:
            results.append({"unit_id": unit, "spearman_rho": np.nan, "p_value": np.nan})
            continue

        rho, p_val = sp_stats.spearmanr(
            shi_vals.loc[common_idx].values,
            rul_vals.loc[common_idx].values,
        )
        results.append({"unit_id": unit, "spearman_rho": rho, "p_value": p_val})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Aggregate quality metrics
# ---------------------------------------------------------------------------

@dataclass
class QualityMetrics:
    """Container for health index quality metrics.

    Attributes
    ----------
    monotonicity:
        Per-unit monotonicity scores.
    trendability:
        Overall trendability score.
    prognosability:
        Overall prognosability score.
    spearman_rho:
        Per-unit Spearman correlation with RUL.
    summary:
        Aggregated statistics (mean ± std).
    """

    monotonicity: pd.DataFrame
    trendability: float
    prognosability: float
    spearman_rho: pd.DataFrame
    summary: dict[str, float]


def compute_quality_metrics(
    shi_df: pd.DataFrame,
    rul_df: pd.DataFrame | None = None,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
    shi_col: str = "shi",
) -> QualityMetrics:
    """Compute all health index quality metrics.

    Parameters
    ----------
    shi_df:
        DataFrame with columns: unit_col, cycle_col, shi_col.
    rul_df:
        Optional DataFrame with RUL values for Spearman correlation.
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.
    shi_col:
        Column name for SHI values.

    Returns
    -------
    QualityMetrics
        Container with all quality metrics.
    """
    # Per-unit monotonicity
    mono_results = []
    for unit, group in shi_df.groupby(unit_col):
        mono = monotonicity(group[shi_col])
        mono_results.append({"unit_id": unit, "monotonicity": mono})
    mono_df = pd.DataFrame(mono_results)

    # Trendability
    trend = trendability(shi_df, unit_col=unit_col, shi_col=shi_col)

    # Prognosability
    prog = prognosability(shi_df, unit_col=unit_col, shi_col=shi_col)

    # Spearman correlation with RUL
    if rul_df is not None:
        spearman_df = spearman_rul_correlation(
            shi_df, rul_df, unit_col=unit_col, cycle_col=cycle_col, shi_col=shi_col
        )
    else:
        spearman_df = pd.DataFrame(columns=["unit_id", "spearman_rho", "p_value"])

    # Aggregate summary
    summary: dict[str, float] = {}

    if not mono_df.empty:
        mono_vals = mono_df["monotonicity"].dropna()
        summary["monotonicity_mean"] = float(mono_vals.mean()) if len(mono_vals) > 0 else np.nan
        summary["monotonicity_std"] = float(mono_vals.std()) if len(mono_vals) > 1 else np.nan

    summary["trendability"] = trend
    summary["prognosability"] = prog

    if not spearman_df.empty:
        rho_vals = spearman_df["spearman_rho"].dropna()
        summary["spearman_rho_mean"] = float(rho_vals.mean()) if len(rho_vals) > 0 else np.nan
        summary["spearman_rho_std"] = float(rho_vals.std()) if len(rho_vals) > 1 else np.nan

    return QualityMetrics(
        monotonicity=mono_df,
        trendability=trend,
        prognosability=prog,
        spearman_rho=spearman_df,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Utility: combined quality report
# ---------------------------------------------------------------------------

def quality_report(
    quality: QualityMetrics,
    decimals: int = 4,
) -> str:
    """Generate a human-readable quality report.

    Parameters
    ----------
    quality:
        Quality metrics container.
    decimals:
        Number of decimal places for formatting.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "Health Index Quality Report",
        "=" * 60,
        "",
    ]

    # Monotonicity
    mono = quality.monotonicity
    if not mono.empty:
        mono_vals = mono["monotonicity"].dropna()
        if len(mono_vals) > 0:
            lines.append(f"Monotonicity:")
            lines.append(f"  Mean: {mono_vals.mean():.{decimals}f}")
            lines.append(f"  Std:  {mono_vals.std():.{decimals}f}")
            lines.append(f"  Min:  {mono_vals.min():.{decimals}f}")
            lines.append(f"  Max:  {mono_vals.max():.{decimals}f}")
            lines.append("")

    # Trendability
    lines.append(f"Trendability: {quality.trendability:.{decimals}f}")
    lines.append("")

    # Prognosability
    lines.append(f"Prognosability: {quality.prognosability:.{decimals}f}")
    lines.append("")

    # Spearman correlation
    spearman = quality.spearman_rho
    if not spearman.empty:
        rho_vals = spearman["spearman_rho"].dropna()
        if len(rho_vals) > 0:
            lines.append(f"Spearman RUL Correlation:")
            lines.append(f"  Mean: {rho_vals.mean():.{decimals}f}")
            lines.append(f"  Std:  {rho_vals.std():.{decimals}f}")
            lines.append(f"  Min:  {rho_vals.min():.{decimals}f}")
            lines.append(f"  Max:  {rho_vals.max():.{decimals}f}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
