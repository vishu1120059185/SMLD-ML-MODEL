"""Tests for feature causality (truncation invariance), slope, EWMA, PSI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_unit
from stattwin.statistics.rolling import (
    rolling_ewma,
    rolling_mean,
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


# ---------------------------------------------------------------------------
# Truncation invariance
# ---------------------------------------------------------------------------


class TestTruncationInvariance:
    """Features at t computed on full trajectory = features on trajectory truncated at t."""

    def test_rolling_mean_truncation_invariant(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100, seed=42)
        t = 50

        # Full trajectory
        full = rolling_mean(df, "sensor_1", windows=[10])
        val_full = full.loc[full["cycle"] == t, "sensor_1_rmean_10"].iloc[0]

        # Truncated at t (only cycles <= t)
        truncated = df[df["cycle"] <= t].copy()
        trunc = rolling_mean(truncated, "sensor_1", windows=[10])
        val_trunc = trunc.loc[trunc["cycle"] == t, "sensor_1_rmean_10"].iloc[0]

        assert val_full == pytest.approx(val_trunc, rel=1e-10)

    def test_rolling_std_truncation_invariant(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100, seed=42)
        t = 50

        full = rolling_std(df, "sensor_1", windows=[10])
        val_full = full.loc[full["cycle"] == t, "sensor_1_rstd_10"].iloc[0]

        truncated = df[df["cycle"] <= t].copy()
        trunc = rolling_std(truncated, "sensor_1", windows=[10])
        val_trunc = trunc.loc[trunc["cycle"] == t, "sensor_1_rstd_10"].iloc[0]

        assert val_full == pytest.approx(val_trunc, rel=1e-10)

    def test_ewma_truncation_invariant(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100, seed=42)
        t = 50

        full = rolling_ewma(df, "sensor_1", alphas=[0.3])
        val_full = full.loc[full["cycle"] == t, "sensor_1_ewma_3"].iloc[0]

        truncated = df[df["cycle"] <= t].copy()
        trunc = rolling_ewma(truncated, "sensor_1", alphas=[0.3])
        val_trunc = trunc.loc[trunc["cycle"] == t, "sensor_1_ewma_3"].iloc[0]

        assert val_full == pytest.approx(val_trunc, rel=1e-10)

    def test_zscore_truncation_invariant(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100, seed=42)
        t = 80

        full = z_score_vs_baseline(df, "sensor_1", baseline_cycles=30)
        val_full = full.loc[full["cycle"] == t, "sensor_1_zscore"].iloc[0]

        truncated = df[df["cycle"] <= t].copy()
        trunc = z_score_vs_baseline(truncated, "sensor_1", baseline_cycles=30)
        val_trunc = trunc.loc[trunc["cycle"] == t, "sensor_1_zscore"].iloc[0]

        assert val_full == pytest.approx(val_trunc, rel=1e-10)


# ---------------------------------------------------------------------------
# Slope on known line
# ---------------------------------------------------------------------------


class TestSlopeKnownLine:
    def test_slope_on_linear_signal(self):
        """A perfectly linear signal should produce a constant positive slope."""
        n = 100
        df = pd.DataFrame(
            {
                "unit_id": np.ones(n, dtype=int),
                "cycle": np.arange(1, n + 1),
                "sensor_1": np.arange(1, n + 1, dtype=float),
            }
        )
        result = rolling_slope(df, "sensor_1", windows=[20])
        slopes = result["sensor_1_slope_20"].dropna()
        # All slopes should be positive and roughly constant
        assert slopes.mean() > 0
        assert slopes.std() < 0.5

    def test_slope_on_constant_signal(self):
        """A constant signal should produce slopes near zero."""
        n = 100
        df = pd.DataFrame(
            {
                "unit_id": np.ones(n, dtype=int),
                "cycle": np.arange(1, n + 1),
                "sensor_1": np.full(n, 5.0),
            }
        )
        result = rolling_slope(df, "sensor_1", windows=[20])
        slopes = result["sensor_1_slope_20"].dropna()
        # Slopes should be near zero (may have division-by-std issues)
        assert abs(slopes.mean()) < 1.0


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


class TestEWMA:
    def test_ewma_monotone_alpha_effect(self):
        """Larger alpha should track the signal more closely (less smoothing)."""
        rng = np.random.default_rng(42)
        signal = np.cumsum(rng.normal(0, 1, 200)) + 100
        df = pd.DataFrame(
            {
                "unit_id": np.ones(200, dtype=int),
                "cycle": np.arange(1, 201),
                "sensor_1": signal,
            }
        )
        result = rolling_ewma(df, "sensor_1", alphas=[0.1, 0.9])
        # With alpha=0.9, the EWMA should be closer to the raw signal
        err_low = (result["sensor_1_ewma_1"] - result["sensor_1"]).abs().mean()
        err_high = (result["sensor_1_ewma_9"] - result["sensor_1"]).abs().mean()
        assert err_high < err_low

    def test_ewma_first_value_equals_input(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=50, seed=42)
        result = rolling_ewma(df, "sensor_1", alphas=[0.3])
        # First value of EWMA should equal first input
        assert result["sensor_1_ewma_3"].iloc[0] == pytest.approx(
            df["sensor_1"].iloc[0], rel=1e-10
        )


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


class TestPSI:
    def test_psi_identical_distributions(self):
        """PSI between identical distributions should be near 0."""
        rng = np.random.default_rng(42)
        signal = rng.normal(10, 1, 100)
        df = pd.DataFrame(
            {
                "unit_id": np.ones(100, dtype=int),
                "cycle": np.arange(1, 101),
                "sensor_1": signal,
            }
        )
        result = population_stability_index(
            df, "sensor_1", window=10, baseline_cycles=30, n_bins=5
        )
        col = "shift_psi_sensor_1_w10"
        # After baseline period, PSI should be near 0 for same distribution
        psi_vals = result[col].dropna()
        assert psi_vals.mean() < 0.1

    def test_psi_shifted_distributions(self):
        """PSI between shifted distributions should be > 0."""
        rng = np.random.default_rng(42)
        bl = rng.normal(10, 1, 50)
        shifted = rng.normal(20, 1, 50)
        signal = np.concatenate([bl, shifted])
        df = pd.DataFrame(
            {
                "unit_id": np.ones(100, dtype=int),
                "cycle": np.arange(1, 101),
                "sensor_1": signal,
            }
        )
        result = population_stability_index(
            df, "sensor_1", window=10, baseline_cycles=30, n_bins=5
        )
        col = "shift_psi_sensor_1_w10"
        psi_vals = result[col].dropna()
        # PSI in the shifted region should be elevated
        late_psi = psi_vals.iloc[-20:].mean()
        assert late_psi > 0.1

    def test_distribution_shift_wrapper(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:2]
        ds = DistributionShift(methods=["psi"], window=5, baseline_cycles=20)
        result = ds.compute(synthetic_dataset, sensor_cols=sensor_cols)
        assert result.shape[0] == synthetic_dataset.shape[0]
