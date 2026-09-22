"""Statistical feature engineering module for STAT-TWIN.

Phase 2: per-unit causal rolling statistics, cross-sensor correlations,
distribution-shift indicators, and train-only feature selection.

All features are causal (cycle *t* uses only cycles *≤ t*), vectorised
via pandas ``groupby().rolling()`` / NumPy, and config-driven.
"""

from stattwin.statistics.cross_sensor import (
    CrossSensorCorrelation,
    correlation_shift,
    delta_rho,
)
from stattwin.statistics.feature_library import (
    compute_features,
)
from stattwin.statistics.feature_selection import (
    select_features,
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
from stattwin.statistics.shift import (
    DistributionShift,
    ks_statistic,
    population_stability_index,
    wasserstein_distance,
)

__all__ = [
    "compute_features",
    "correlation_shift",
    "CrossSensorCorrelation",
    "delta_rho",
    "DistributionShift",
    "ks_statistic",
    "population_stability_index",
    "rolling_cv",
    "rolling_ewma",
    "rolling_ewma_dev",
    "rolling_kurtosis",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_pct_change",
    "rolling_range",
    "rolling_rate_of_change",
    "rolling_skew",
    "rolling_slope",
    "rolling_std",
    "select_features",
    "wasserstein_distance",
    "z_score_vs_baseline",
]
