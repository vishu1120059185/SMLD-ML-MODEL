"""Forecasting module – probability curves, RUL estimation, and consistency checks."""

from stattwin.forecasting.forecast import (
    ConsistencyReport,
    ForecastProfile,
    build_probability_curve,
    build_rul_profile,
    check_consistency,
    predicted_failure_point,
)

__all__ = [
    "ForecastProfile",
    "ConsistencyReport",
    "build_probability_curve",
    "build_rul_profile",
    "check_consistency",
    "predicted_failure_point",
]
