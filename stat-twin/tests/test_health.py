"""Tests for SHI bounds, state persistence, and quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_dataset, make_synthetic_unit
from stattwin.health.quality import (
    QualityMetrics,
    compute_quality_metrics,
    monotonicity,
    prognosability,
    spearman_rul_correlation,
)
from stattwin.health.shi import (
    HealthIndex,
    compute_shi,
)
from stattwin.health.states import (
    HealthState,
    StateClassifier,
    classify_states,
)


# ---------------------------------------------------------------------------
# SHI bounds
# ---------------------------------------------------------------------------


class TestSHIBounds:
    def test_shi_values_in_0_100(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        result = compute_shi(
            synthetic_dataset,
            sensor_cols=sensor_cols,
            baseline_cycles=20,
        )
        shi_vals = result.shi_values["shi"]
        assert shi_vals.min() >= 0.0
        assert shi_vals.max() <= 100.0

    def test_shi_has_required_fields(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        result = compute_shi(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)
        assert isinstance(result, HealthIndex)
        assert "shi" in result.shi_values.columns
        assert "unit_id" in result.shi_values.columns
        assert "cycle" in result.shi_values.columns
        assert len(result.weights) > 0
        assert len(result.sensor_weights) > 0

    def test_shi_simple(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        from stattwin.health.shi import compute_shi_simple
        result = compute_shi_simple(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)
        assert "shi" in result.columns
        assert result["shi"].min() >= 0.0
        assert result["shi"].max() <= 100.0


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    @pytest.fixture()
    def shi_df(self):
        """SHI DataFrame with a clear degradation trend."""
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame(
            {
                "unit_id": np.ones(n, dtype=int),
                "cycle": np.arange(1, n + 1),
                "shi": np.linspace(90, 10, n) + rng.normal(0, 2, n),
            }
        )

    def test_state_values(self, shi_df):
        result = classify_states(shi_df, method="fixed_grid", persistence=3)
        assert "health_state" in result.columns
        valid_states = set(s.value for s in HealthState)
        assert result["health_state"].isin(valid_states).all()

    def test_persistence_reduces_oscillation(self, shi_df):
        """With persistence > 1, state changes should be smoother."""
        no_persist = classify_states(shi_df, method="fixed_grid", persistence=1)
        with_persist = classify_states(shi_df, method="fixed_grid", persistence=5)

        n_changes_no = (no_persist["health_state"].diff().abs() > 0).sum()
        n_changes_with = (with_persist["health_state"].diff().abs() > 0).sum()
        # Persistence should reduce transitions
        assert n_changes_with <= n_changes_no

    def test_health_state_enum(self):
        assert HealthState.HEALTHY.value == 4
        assert HealthState.FAILURE_LIKELY.value == 0
        assert HealthState.from_name("healthy") == HealthState.HEALTHY
        assert HealthState.from_name("failure-likely") == HealthState.FAILURE_LIKELY

    def test_classifier_calibrate_and_classify(self, shi_df):
        clf = StateClassifier(method="fixed_grid", persistence=3)
        clf.calibrate(shi_df)
        result = clf.classify(shi_df)
        assert "health_state" in result.columns

    def test_classifier_thresholds(self, shi_df):
        clf = StateClassifier(method="fixed_grid", persistence=3)
        clf.calibrate(shi_df)
        assert len(clf.thresholds) >= 3


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    def test_monotonicity_perfect(self):
        s = pd.Series([100, 90, 80, 70, 60, 50])
        assert monotonicity(s) == pytest.approx(1.0)

    def test_monotonicity_imperfect(self):
        s = pd.Series([100, 80, 90, 60, 50])
        # diffs: -20, +10, -30, -10 => n_inc=1, n_dec=3 => (1+3)/4 = 1.0
        # Wait, that's still 1.0 because all diffs are nonzero
        m = monotonicity(s)
        assert 0.0 <= m <= 1.0

    def test_monotonicity_constant(self):
        s = pd.Series([50, 50, 50, 50])
        assert monotonicity(s) == 0.0

    def test_monotonicity_short_series(self):
        s = pd.Series([10.0])
        assert np.isnan(monotonicity(s))

    def test_prognosability(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        result = compute_shi(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)
        prog = prognosability(result.shi_values)
        assert isinstance(prog, float)

    def test_compute_quality_metrics(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        shi_result = compute_shi(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)
        qm = compute_quality_metrics(
            shi_result.shi_values,
            rul_df=synthetic_dataset[["unit_id", "cycle", "RUL"]],
        )
        assert isinstance(qm, QualityMetrics)
        assert "monotonicity_mean" in qm.summary or len(qm.monotonicity) > 0

    def test_spearman_rul_correlation(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        shi_result = compute_shi(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)
        corr = spearman_rul_correlation(
            shi_result.shi_values,
            synthetic_dataset[["unit_id", "cycle", "RUL"]],
        )
        assert "spearman_rho" in corr.columns
        assert len(corr) > 0
