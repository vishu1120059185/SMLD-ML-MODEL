"""Tests for probability curve, RUL estimation, and monotone enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.forecasting.forecast import (
    ConsistencyReport,
    ForecastProfile,
    build_probability_curve,
    build_rul_profile,
    check_consistency,
    predicted_failure_point,
)


class TestProbabilityCurve:
    def test_build_probability_curve(self):
        proba_row = pd.Series(
            {label_col_for(h): p for h, p in zip(FAILURE_HORIZONS, [0.1, 0.2, 0.4, 0.6, 0.9])}
        )
        horizons_arr, proba_arr, interp_x, interp_y = build_probability_curve(proba_row)

        assert len(horizons_arr) == len(FAILURE_HORIZONS)
        assert len(proba_arr) == len(FAILURE_HORIZONS)
        assert len(interp_x) == 200
        assert len(interp_y) == 200
        # Interpolated values should be in [0, 1]
        assert interp_y.min() >= 0.0
        assert interp_y.max() <= 1.0

    def test_probability_curve_monotone_input(self):
        """Monotonically increasing input should produce approximately monotone output."""
        proba_row = pd.Series(
            {label_col_for(h): p for h, p in zip(FAILURE_HORIZONS, [0.1, 0.3, 0.5, 0.7, 0.9])}
        )
        _, _, _, interp_y = build_probability_curve(proba_row)
        diffs = np.diff(interp_y)
        # PCHIP preserves monotonicity when data is monotone
        # Allow small numerical violations
        n_violations = (diffs < -0.01).sum()
        assert n_violations < len(diffs) * 0.05

    def test_probability_curve_missing_columns(self):
        proba_row = pd.Series({"fail_h10": 0.1})
        with pytest.raises(ValueError, match="Missing"):
            build_probability_curve(proba_row)


class TestRULProfile:
    def test_build_rul_profile_basic(self):
        rul_clipped, failure_hat, interval = build_rul_profile(
            rul_value=50.0, t_now=100.0
        )
        assert rul_clipped == 50.0
        assert failure_hat == 150.0
        assert interval is None

    def test_build_rul_profile_negative_clipped(self):
        rul_clipped, failure_hat, _ = build_rul_profile(rul_value=-10.0, t_now=100.0)
        assert rul_clipped == 0.0
        assert failure_hat == 100.0

    def test_build_rul_profile_with_conformal(self):
        rul_clipped, failure_hat, interval = build_rul_profile(
            rul_value=50.0, t_now=100.0, sigma_ens=5.0, conformal_q=1.645
        )
        assert interval is not None
        lower, upper = interval
        assert lower < failure_hat < upper

    def test_build_rul_profile_with_sigma(self):
        rul_clipped, failure_hat, interval = build_rul_profile(
            rul_value=50.0, t_now=100.0, sigma_ens=5.0
        )
        assert interval is not None
        lower, upper = interval
        assert lower < failure_hat < upper


class TestConsistencyCheck:
    def test_consistent_probas(self):
        proba_row = pd.Series(
            {label_col_for(h): p for h, p in zip(FAILURE_HORIZONS, [0.1, 0.3, 0.5, 0.7, 0.9])}
        )
        report = check_consistency(proba_row, rul_point=40.0, t_now=0.0)
        assert isinstance(report, ConsistencyReport)
        assert report.spearman_rho > 0.8
        assert report.monotonicity_violations == 0
        assert report.is_consistent

    def test_inconsistent_probas(self):
        proba_row = pd.Series(
            {label_col_for(h): p for h, p in zip(FAILURE_HORIZONS, [0.9, 0.3, 0.1, 0.7, 0.5])}
        )
        report = check_consistency(proba_row, rul_point=40.0, t_now=0.0)
        assert report.spearman_rho < 0.8 or report.monotonicity_violations > 0


class TestPredictedFailurePoint:
    def test_predicted_failure_point(self):
        rul_series = pd.Series([50.0, 40.0, 30.0])
        cycle_series = pd.Series([100.0, 110.0, 120.0])
        result = predicted_failure_point(rul_series, cycle_series)
        assert "rul_hat" in result.columns
        assert "failure_hat" in result.columns
        pd.testing.assert_series_equal(
            result["failure_hat"], cycle_series + rul_series, check_names=False
        )

    def test_predicted_failure_point_with_sigma(self):
        rul_series = pd.Series([50.0, 40.0])
        cycle_series = pd.Series([100.0, 110.0])
        sigma = pd.Series([5.0, 5.0])
        result = predicted_failure_point(rul_series, cycle_series, sigma_ens=sigma)
        assert "interval_lower" in result.columns
        assert "interval_upper" in result.columns
        assert (result["interval_lower"] <= result["failure_hat"]).all()
        assert (result["interval_upper"] >= result["failure_hat"]).all()
