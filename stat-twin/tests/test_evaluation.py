"""Tests for metric implementations vs sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from stattwin.evaluation.metrics import (
    ClassificationReport,
    IntervalReport,
    ProbabilisticReport,
    RULReport,
    evaluate_classification,
    evaluate_rul,
    interval_metrics,
)


class TestEvaluateClassification:
    @pytest.fixture()
    def perfect_predictions(self):
        y_true = {10: np.array([0, 0, 1, 1]), 20: np.array([0, 1, 0, 1])}
        y_prob = {10: np.array([0.0, 0.1, 0.9, 1.0]), 20: np.array([0.0, 0.9, 0.1, 0.8])}
        return y_true, y_prob

    def test_perfect_classification(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        reports = evaluate_classification(y_true, y_prob, horizons=[10, 20])
        assert len(reports) == 2
        for r in reports:
            assert isinstance(r, ClassificationReport)

    def test_roc_auc_vs_sklearn(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)

        reports = evaluate_classification(
            {30: y_true}, {30: y_prob}, horizons=[30]
        )
        sklearn_auc = roc_auc_score(y_true, y_prob)
        assert reports[0].roc_auc == pytest.approx(sklearn_auc, abs=1e-6)

    def test_f1_vs_sklearn(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        y_pred = (y_prob >= 0.5).astype(int)

        reports = evaluate_classification(
            {30: y_true}, {30: y_prob}, horizons=[30]
        )
        sklearn_f1 = f1_score(y_true, y_pred, zero_division=0)
        assert reports[0].f1 == pytest.approx(sklearn_f1, abs=1e-6)

    def test_missing_horizon(self):
        reports = evaluate_classification({}, {}, horizons=[30])
        assert len(reports) == 1
        assert np.isnan(reports[0].roc_auc)


class TestEvaluateRUL:
    def test_perfect_predictions(self):
        y_true = np.array([100, 80, 60, 40, 20])
        y_pred = np.array([100, 80, 60, 40, 20])
        report = evaluate_rul(y_true, y_pred)
        assert report.mae == pytest.approx(0.0)
        assert report.rmse == pytest.approx(0.0)
        assert report.nasa_score == pytest.approx(0.0)

    def test_mae_vs_sklearn(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0, 100, 200)
        y_pred = y_true + rng.normal(0, 10, 200)

        report = evaluate_rul(y_true, y_pred)
        sklearn_mae = mean_absolute_error(y_true, y_pred)
        assert report.mae == pytest.approx(sklearn_mae, abs=1e-6)

    def test_rmse_vs_sklearn(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0, 100, 200)
        y_pred = y_true + rng.normal(0, 10, 200)

        report = evaluate_rul(y_true, y_pred)
        sklearn_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert report.rmse == pytest.approx(sklearn_rmse, abs=1e-6)

    def test_nasa_score_asymmetric(self):
        """NASA score is asymmetric: exp(d/10)-1 for d>=0, exp(-d/13)-1 for d<0."""
        y_true = np.array([50.0])
        # d = +20 (predicted 30, too early): exp(20/10) - 1 ≈ 6.39
        early_pred = evaluate_rul(y_true, np.array([30.0]))
        # d = -20 (predicted 70, too late): exp(20/13) - 1 ≈ 3.66
        late_pred = evaluate_rul(y_true, np.array([70.0]))
        # Both nonzero; asymmetric
        assert early_pred.nasa_score > 0
        assert late_pred.nasa_score > 0
        assert early_pred.nasa_score != late_pred.nasa_score

    def test_empty_input(self):
        report = evaluate_rul(np.array([]), np.array([]))
        assert report.n == 0


class TestIntervalMetrics:
    def test_perfect_coverage(self):
        y_true = np.array([50.0, 60.0, 70.0])
        lower = np.array([40.0, 50.0, 60.0])
        upper = np.array([60.0, 70.0, 80.0])
        report = interval_metrics(y_true, lower, upper, alpha=0.10)
        assert report.picp == pytest.approx(1.0)
        assert report.mean_width == pytest.approx(20.0)

    def test_zero_coverage(self):
        y_true = np.array([100.0, 200.0, 300.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        report = interval_metrics(y_true, lower, upper, alpha=0.10)
        assert report.picp == pytest.approx(0.0)

    def test_empty_input(self):
        report = interval_metrics(np.array([]), np.array([]), np.array([]))
        assert report.n == 0
