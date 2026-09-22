"""Tests for conformal intervals on synthetic data (empirical coverage ~ 1-alpha)."""

from __future__ import annotations

import numpy as np
import pytest

from stattwin.uncertainty.conformal import (
    ConformalReport,
    conformal_intervals,
    winkler_score,
)


class TestConformalIntervals:
    @pytest.fixture()
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        n_cal = 200
        n_test = 200
        # True y from a known distribution
        y_cal_true = rng.normal(50, 10, n_cal)
        y_cal_pred = y_cal_true + rng.normal(0, 3, n_cal)
        sigma_cal = np.full(n_cal, 3.0)

        y_test_true = rng.normal(50, 10, n_test)
        y_test_pred = y_test_true + rng.normal(0, 3, n_test)
        sigma_test = np.full(n_test, 3.0)

        return y_cal_true, y_cal_pred, sigma_cal, y_test_true, y_test_pred, sigma_test

    def test_conformal_report_structure(self, synthetic_data):
        y_cal_true, y_cal_pred, sigma_cal, y_test_true, y_test_pred, sigma_test = synthetic_data
        report = conformal_intervals(
            y_cal_true, y_cal_pred, sigma_cal,
            y_test_pred, sigma_test, alpha=0.10,
        )
        assert isinstance(report, ConformalReport)
        assert report.alpha == 0.10
        assert report.quantile_q > 0
        assert report.intervals is not None
        assert len(report.intervals) == len(y_test_pred)

    def test_conformal_coverage_near_nominal(self, synthetic_data):
        """Empirical coverage should be near 1-alpha for well-calibrated data."""
        y_cal_true, y_cal_pred, sigma_cal, y_test_true, y_test_pred, sigma_test = synthetic_data
        alpha = 0.10
        report = conformal_intervals(
            y_cal_true, y_cal_pred, sigma_cal,
            y_test_pred, sigma_test, alpha=alpha,
        )
        # Coverage on calibration set should be close to 1-alpha
        assert report.coverage >= 0.80
        assert report.coverage <= 1.0

    def test_conformal_interval_width_positive(self, synthetic_data):
        y_cal_true, y_cal_pred, sigma_cal, y_test_true, y_test_pred, sigma_test = synthetic_data
        report = conformal_intervals(
            y_cal_true, y_cal_pred, sigma_cal,
            y_test_pred, sigma_test, alpha=0.10,
        )
        assert (report.intervals["width"] >= 0).all()

    def test_conformal_lower_upper_ordering(self, synthetic_data):
        y_cal_true, y_cal_pred, sigma_cal, y_test_true, y_test_pred, sigma_test = synthetic_data
        report = conformal_intervals(
            y_cal_true, y_cal_pred, sigma_cal,
            y_test_pred, sigma_test, alpha=0.10,
        )
        assert (report.intervals["lower"] <= report.intervals["upper"]).all()

    def test_conformal_with_rul_buckets(self, synthetic_data):
        y_cal_true, y_cal_pred, sigma_cal, y_test_true, y_test_pred, sigma_test = synthetic_data
        rul_test = np.linspace(0, 100, len(y_test_pred))
        report = conformal_intervals(
            y_cal_true, y_cal_pred, sigma_cal,
            y_test_pred, sigma_test, alpha=0.10, rul_test=rul_test,
        )
        assert len(report.coverage_by_rul_bucket) > 0


class TestWinklerScore:
    def test_winkler_perfect(self):
        y_true = np.array([50.0, 60.0, 70.0])
        lower = np.array([40.0, 50.0, 60.0])
        upper = np.array([60.0, 70.0, 80.0])
        ws = winkler_score(y_true, lower, upper, alpha=0.10)
        # Perfect coverage -> just width
        expected_width = np.mean(upper - lower)
        assert ws == pytest.approx(expected_width, rel=1e-10)

    def test_winkler_penalty(self):
        y_true = np.array([100.0])  # Outside interval
        lower = np.array([40.0])
        upper = np.array([60.0])
        ws = winkler_score(y_true, lower, upper, alpha=0.10)
        # Penalty for missing above
        width = 20.0
        penalty = (2.0 / 0.10) * (100.0 - 60.0)
        assert ws == pytest.approx(width + penalty, rel=1e-10)
