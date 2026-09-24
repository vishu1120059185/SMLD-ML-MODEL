"""Tests for occlusion attribution (top contributor lowers risk)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.explainability.attribution import (
    AttributionResult,
    _map_sensor_to_features,
    group_occlusion_attribution,
)

# ---------------------------------------------------------------------------
# Simple model for attribution tests
# ---------------------------------------------------------------------------


class SimpleProbModel:
    """Model where risk increases with sensor_1 and decreases with sensor_2."""

    def __init__(self):
        pass

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        # Risk proportional to sensor_1, inversely to sensor_2
        s1 = X["sensor_1"].values if "sensor_1" in X.columns else np.zeros(n)
        s2 = X["sensor_2"].values if "sensor_2" in X.columns else np.zeros(n)
        risk = 0.5 + 0.01 * s1 - 0.005 * s2
        risk = np.clip(risk, 0.0, 1.0)
        return pd.DataFrame(
            {f"fail_h{h}": risk for h in [10, 20, 30, 40, 50]},
            index=X.index,
        )


class TestGroupOcclusion:
    @pytest.fixture()
    def model(self):
        return SimpleProbModel()

    @pytest.fixture()
    def x_current(self):
        return pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [100],
                "sensor_1": [20.0],
                "sensor_1_rmean_10": [19.0],
                "sensor_1_rstd_10": [1.5],
                "sensor_2": [10.0],
                "sensor_2_rmean_10": [11.0],
                "sensor_2_rstd_10": [0.8],
            }
        )

    @pytest.fixture()
    def x_baseline(self):
        return pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [1],
                "sensor_1": [10.0],
                "sensor_1_rmean_10": [10.0],
                "sensor_1_rstd_10": [0.5],
                "sensor_2": [20.0],
                "sensor_2_rmean_10": [20.0],
                "sensor_2_rstd_10": [0.3],
            }
        )

    def test_attribution_result_structure(self, model, x_current, x_baseline):
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = [
            "sensor_1", "sensor_1_rmean_10", "sensor_1_rstd_10",
            "sensor_2", "sensor_2_rmean_10", "sensor_2_rstd_10",
        ]
        result = group_occlusion_attribution(
            model, x_current, x_baseline,
            sensor_cols=sensor_cols,
            feature_cols=feature_cols,
            horizon=30,
        )
        assert isinstance(result, AttributionResult)
        assert result.method == "group_occlusion"
        assert "sensor_1" in result.sensor_contributions
        assert "sensor_2" in result.sensor_contributions

    def test_top_contributor_lowers_risk(self, model, x_current, x_baseline):
        """Occluding the top contributor should lower the predicted risk."""
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = [
            "sensor_1", "sensor_1_rmean_10", "sensor_1_rstd_10",
            "sensor_2", "sensor_2_rmean_10", "sensor_2_rstd_10",
        ]
        result = group_occlusion_attribution(
            model, x_current, x_baseline,
            sensor_cols=sensor_cols,
            feature_cols=feature_cols,
            horizon=30,
        )
        # sensor_1 has high current value -> positive contribution (increases risk)
        assert result.sensor_contributions["sensor_1"] > 0

    def test_top_sensors_sorted(self, model, x_current, x_baseline):
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = [
            "sensor_1", "sensor_1_rmean_10", "sensor_1_rstd_10",
            "sensor_2", "sensor_2_rmean_10", "sensor_2_rstd_10",
        ]
        result = group_occlusion_attribution(
            model, x_current, x_baseline,
            sensor_cols=sensor_cols,
            feature_cols=feature_cols,
            horizon=30,
        )
        top = result.top_sensors
        assert len(top) == 2
        # sensor_1 should be first (highest absolute contribution)
        assert top[0] == "sensor_1"

    def test_to_dict(self, model, x_current, x_baseline):
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = [
            "sensor_1", "sensor_1_rmean_10", "sensor_1_rstd_10",
            "sensor_2", "sensor_2_rmean_10", "sensor_2_rstd_10",
        ]
        result = group_occlusion_attribution(
            model, x_current, x_baseline,
            sensor_cols=sensor_cols,
            feature_cols=feature_cols,
            horizon=30,
        )
        d = result.to_dict()
        assert "method" in d
        assert "sensor_contributions" in d


class TestMapSensorToFeatures:
    def test_basic_mapping(self):
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = [
            "sensor_1", "sensor_1_rmean_10", "sensor_1_rstd_10",
            "sensor_2", "sensor_2_rmean_10",
        ]
        mapping = _map_sensor_to_features(sensor_cols, feature_cols)
        assert len(mapping["sensor_1"]) == 3
        assert len(mapping["sensor_2"]) == 2

    def test_unmatched_features(self):
        sensor_cols = ["sensor_1"]
        feature_cols = ["sensor_1", "unknown_feature"]
        mapping = _map_sensor_to_features(sensor_cols, feature_cols)
        assert len(mapping["sensor_1"]) == 1
        assert "unknown_feature" not in mapping["sensor_1"]
