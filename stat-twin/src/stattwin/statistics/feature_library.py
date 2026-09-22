"""Full feature library computation for STAT-TWIN.

``compute_features`` is the single entry-point that produces the entire
statistical feature matrix from a preprocessed DataFrame.  All features
are:

* **Causal** – features at cycle *t* use only observations from
  cycles *≤ t* within the same unit.
* **Vectorised** – implemented via ``pandas.groupby().rolling()`` or
  NumPy; no per-row Python loops over cycles in the hot path.
* **Config-driven** – window sizes, EWMA alphas, baseline length,
  top-k pairs, and shift methods are all controlled by the
  ``STATTWINConfig.stats`` sub-config.

The function returns a DataFrame with the original columns plus all
engineered features, and a ``FeatureSpec`` metadata object recording
which columns were generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stattwin.config import STATTWINConfig
from stattwin.statistics.cross_sensor import (
    CrossSensorCorrelation,
    correlation_shift,
    delta_rho,
)
from stattwin.statistics.rolling import (
    rolling_cv,
    rolling_ewma,
    rolling_ewma_dev,
    rolling_kurtosis,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_pct_change,
    rolling_range,
    rolling_rate_of_change,
    rolling_skew,
    rolling_slope,
    rolling_std,
    z_score_vs_baseline,
)
from stattwin.statistics.shift import DistributionShift

__all__ = ["compute_features", "FeatureSpec"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Record of which feature columns were generated."""

    sensor_columns: list[str] = field(default_factory=list)
    rolling_mean_cols: list[str] = field(default_factory=list)
    rolling_std_cols: list[str] = field(default_factory=list)
    rolling_min_cols: list[str] = field(default_factory=list)
    rolling_max_cols: list[str] = field(default_factory=list)
    rolling_range_cols: list[str] = field(default_factory=list)
    zscore_cols: list[str] = field(default_factory=list)
    ewma_cols: list[str] = field(default_factory=list)
    ewma_dev_cols: list[str] = field(default_factory=list)
    slope_cols: list[str] = field(default_factory=list)
    pct_change_cols: list[str] = field(default_factory=list)
    rate_of_change_cols: list[str] = field(default_factory=list)
    cv_cols: list[str] = field(default_factory=list)
    skew_cols: list[str] = field(default_factory=list)
    kurtosis_cols: list[str] = field(default_factory=list)
    corr_features: list[str] = field(default_factory=list)
    shift_features: list[str] = field(default_factory=list)
    all_feature_cols: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _numeric_sensor_cols(
    df: pd.DataFrame, unit_col: str = _UNIT_COL
) -> list[str]:
    """Return numeric columns excluding identifiers."""
    exclude = {unit_col, _CYCLE_COL, "RUL"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _track_new_cols(before: set[str], after: set[str]) -> list[str]:
    """Return sorted list of columns that are in *after* but not *before*."""
    return sorted(after - before)


# ------------------------------------------------------------------
# Main entry-point
# ------------------------------------------------------------------


def compute_features(
    df: pd.DataFrame,
    config: STATTWINConfig | None = None,
    *,
    sensor_cols: list[str] | None = None,
    train_unit_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Compute the full statistical feature library.

    Parameters
    ----------
    df:
        Preprocessed DataFrame with columns ``unit_id``, ``cycle``,
        and sensor columns.
    config:
        STAT-TWIN configuration.  If *None*, a default config is used.
    sensor_cols:
        Subset of sensor columns to engineer features for.  If *None*,
        all numeric columns except identifiers are used.
    train_unit_ids:
        Array of unit IDs belonging to the training partition.  Used
        by ``CrossSensorCorrelation`` to select top-k pairs.  If
        *None*, cross-sensor features are skipped.

    Returns
    -------
    (out_df, spec)
        *out_df* is a copy of *df* with all feature columns added.
        *spec* is a ``FeatureSpec`` recording the generated columns.
    """
    if config is None:
        config = STATTWINConfig()

    stats_cfg = config.stats
    windows = stats_cfg.windows
    alphas = stats_cfg.ewma_alpha
    baseline = stats_cfg.baseline_cycles
    top_k = stats_cfg.corr.top_k
    corr_methods = stats_cfg.corr.methods
    shift_methods = stats_cfg.shift

    if sensor_cols is None:
        sensor_cols = _numeric_sensor_cols(df, _UNIT_COL)

    spec = FeatureSpec(sensor_columns=list(sensor_cols))
    out = df.copy()
    base_cols = set(out.columns)

    # ----------------------------------------------------------------
    # 1. Per-sensor rolling statistics
    # ----------------------------------------------------------------
    for col in sensor_cols:
        cols_before = set(out.columns)

        # Mean, std, min, max, range
        out = rolling_mean(out, col, windows=windows)
        out = rolling_std(out, col, windows=windows)
        out = rolling_min(out, col, windows=windows)
        out = rolling_max(out, col, windows=windows)
        out = rolling_range(out, col, windows=windows)

        # Z-score vs baseline
        out = z_score_vs_baseline(out, col, baseline_cycles=baseline)

        # EWMA and EWMA deviation
        out = rolling_ewma(out, col, alphas=alphas)
        out = rolling_ewma_dev(out, col, alphas=alphas)

        # Linear trend / slope
        out = rolling_slope(out, col, windows=windows)

        # Percentage change, rate of change
        out = rolling_pct_change(out, col, windows=windows)
        out = rolling_rate_of_change(out, col, windows=windows)

        # Coefficient of variation
        out = rolling_cv(out, col, windows=windows)

        # Higher moments
        out = rolling_skew(out, col, windows=windows)
        out = rolling_kurtosis(out, col, windows=windows)

        new_cols = _track_new_cols(cols_before, set(out.columns))
        spec.rolling_mean_cols.extend([c for c in new_cols if "_rmean_" in c])
        spec.rolling_std_cols.extend([c for c in new_cols if "_rstd_" in c])
        spec.rolling_min_cols.extend([c for c in new_cols if "_rmin_" in c])
        spec.rolling_max_cols.extend([c for c in new_cols if "_rmax_" in c])
        spec.rolling_range_cols.extend([c for c in new_cols if "_rrange_" in c])
        spec.zscore_cols.extend([c for c in new_cols if "_zscore" in c])
        spec.ewma_cols.extend([c for c in new_cols if "_ewma_" in c and "dev" not in c])
        spec.ewma_dev_cols.extend([c for c in new_cols if "_ewma_dev_" in c])
        spec.slope_cols.extend([c for c in new_cols if "_slope_" in c])
        spec.pct_change_cols.extend([c for c in new_cols if "_pctchg_" in c])
        spec.rate_of_change_cols.extend([c for c in new_cols if "_roc_" in c])
        spec.cv_cols.extend([c for c in new_cols if "_cv_" in c])
        spec.skew_cols.extend([c for c in new_cols if "_skew_" in c])
        spec.kurtosis_cols.extend([c for c in new_cols if "_kurt_" in c])

    # ----------------------------------------------------------------
    # 2. Cross-sensor correlations (top-k pairs from training data)
    # ----------------------------------------------------------------
    if train_unit_ids is not None and len(sensor_cols) >= 2:
        train_df = df[df[_UNIT_COL].isin(train_unit_ids)].copy()
        cross = CrossSensorCorrelation(
            methods=corr_methods,
            top_k=top_k,
            windows=windows,
        )
        cross.fit(train_df, sensor_cols=sensor_cols)
        out = cross.transform(out, sensor_cols=sensor_cols)

        # Correlation shift and delta_rho for each selected pair
        for col_a, col_b in cross.selected_pairs_:
            for method in corr_methods:
                for w in windows:
                    out = correlation_shift(
                        out, col_a, col_b, w,
                        baseline_cycles=baseline,
                        method=method,
                    )
                    out = delta_rho(
                        out, col_a, col_b, w,
                        baseline_cycles=baseline,
                        method=method,
                    )

        # Track cross-sensor feature columns
        per_sensor_end = base_cols | set(
            c for c in out.columns
            if any(
                tag in c
                for tag in ["_rmean_", "_rstd_", "_rmin_", "_rmax_", "_rrange_",
                             "_zscore", "_ewma_", "_slope_", "_pctchg_", "_roc_",
                             "_cv_", "_skew_", "_kurt_"]
            )
        )
        spec.corr_features = _track_new_cols(per_sensor_end, set(out.columns))

    # ----------------------------------------------------------------
    # 3. Distribution shift
    # ----------------------------------------------------------------
    if shift_methods:
        shift_engine = DistributionShift(
            methods=shift_methods,
            window=windows[0] if windows else 10,
            baseline_cycles=baseline,
        )
        before_shift = set(out.columns)
        out = shift_engine.compute(out, sensor_cols=sensor_cols)
        spec.shift_features = _track_new_cols(before_shift, set(out.columns))

    # ----------------------------------------------------------------
    # Finalise
    # ----------------------------------------------------------------
    spec.all_feature_cols = sorted(set(out.columns) - set(df.columns))

    return out, spec
