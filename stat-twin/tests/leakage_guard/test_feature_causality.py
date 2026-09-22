"""Leakage Guard: Features at t computed on full trajectory = features on trajectory truncated at t."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_unit
from stattwin.statistics.rolling import (
    rolling_cv,
    rolling_ewma,
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
    ks_statistic,
    population_stability_index,
    wasserstein_distance,
)


class TestFeatureCausality:
    """Features at t computed on full trajectory = features on trajectory truncated at t (truncation invariance)."""

    @pytest.fixture()
    def unit_df(self):
        return make_synthetic_unit(unit_id=1, n_cycles=150, seed=42)

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            (rolling_mean, {"windows": [10]}),
            (rolling_std, {"windows": [10]}),
            (rolling_min, {"windows": [10]}),
            (rolling_max, {"windows": [10]}),
            (rolling_range, {"windows": [10]}),
            (rolling_ewma, {"alphas": [0.3]}),
            (rolling_slope, {"windows": [20]}),
            (rolling_pct_change, {"windows": [10]}),
            (rolling_rate_of_change, {"windows": [10]}),
            (rolling_cv, {"windows": [10]}),
            (rolling_skew, {"windows": [10]}),
            (rolling_kurtosis, {"windows": [10]}),
        ],
    )
    def test_rolling_truncation_invariance(self, unit_df, func, kwargs):
        """Rolling features at cycle t are identical whether computed on full or truncated trajectory."""
        t = 80
        col = "sensor_1"

        full_result = func(unit_df, col, **kwargs)
        new_cols = [c for c in full_result.columns if c not in unit_df.columns]
        for new_col in new_cols:
            val_full = full_result.loc[full_result["cycle"] == t, new_col].iloc[0]

            truncated = unit_df[unit_df["cycle"] <= t].copy()
            trunc_result = func(truncated, col, **kwargs)
            val_trunc = trunc_result.loc[trunc_result["cycle"] == t, new_col].iloc[0]

            if pd.isna(val_full) and pd.isna(val_trunc):
                continue
            assert val_full == pytest.approx(val_trunc, rel=1e-10), (
                f"{func.__name__}.{new_col} at t={t}: full={val_full}, trunc={val_trunc}"
            )

    def test_zscore_truncation_invariance(self, unit_df):
        t = 80
        col = "sensor_1"
        full_result = z_score_vs_baseline(unit_df, col, baseline_cycles=30)
        val_full = full_result.loc[full_result["cycle"] == t, f"{col}_zscore"].iloc[0]

        truncated = unit_df[unit_df["cycle"] <= t].copy()
        trunc_result = z_score_vs_baseline(truncated, col, baseline_cycles=30)
        val_trunc = trunc_result.loc[trunc_result["cycle"] == t, f"{col}_zscore"].iloc[0]

        assert val_full == pytest.approx(val_trunc, rel=1e-10)

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            (ks_statistic, {"window": 10, "baseline_cycles": 30}),
            (wasserstein_distance, {"window": 10, "baseline_cycles": 30}),
            (population_stability_index, {"window": 10, "baseline_cycles": 30, "n_bins": 5}),
        ],
    )
    def test_shift_truncation_invariance(self, unit_df, func, kwargs):
        t = 80
        col = "sensor_1"

        full_result = func(unit_df, col, **kwargs)
        new_cols = [c for c in full_result.columns if c not in unit_df.columns]
        for new_col in new_cols:
            val_full = full_result.loc[full_result["cycle"] == t, new_col].iloc[0]

            truncated = unit_df[unit_df["cycle"] <= t].copy()
            trunc_result = func(truncated, col, **kwargs)
            val_trunc = trunc_result.loc[trunc_result["cycle"] == t, new_col].iloc[0]

            if pd.isna(val_full) and pd.isna(val_trunc):
                continue
            assert val_full == pytest.approx(val_trunc, rel=1e-10), (
                f"{func.__name__}.{new_col} at t={t}: full={val_full}, trunc={val_trunc}"
            )
