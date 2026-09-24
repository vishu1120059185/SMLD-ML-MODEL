"""Tests for missing value imputation, outlier detection, scaling, and DQ flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.preprocessing.dq import DQConfig, DQEngine, DQFlags
from stattwin.preprocessing.missing import (
    CausalForwardFill,
    CausalLinearInterpolation,
    MissingnessIndicators,
    TrainMedianFill,
)
from stattwin.preprocessing.outliers import (
    IQRFencesDetector,
    RobustZDetector,
    Winsorizer,
    ZScoreDetector,
)
from stattwin.preprocessing.scaler import PerConditionScaler, TrainFittedScaler

# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------


class TestMissingValues:
    @pytest.fixture()
    def df_with_nans(self):
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "cycle": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                "sensor_1": [1.0, np.nan, 3.0, 4.0, 5.0, 10.0, np.nan, 12.0, 13.0, 14.0],
                "sensor_2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, np.nan, 90.0, 100.0],
            }
        )
        return df

    def test_causal_forward_fill(self, df_with_nans):
        imputer = CausalForwardFill(columns=["sensor_1", "sensor_2"])
        result = imputer.fit_transform(df_with_nans)
        # Unit 1: cycle 2 was NaN -> filled with cycle 1 value (1.0)
        assert result.loc[1, "sensor_1"] == 1.0
        # Unit 2: cycle 2 was NaN -> filled with cycle 1 value (10.0)
        assert result.loc[6, "sensor_2"] == 70.0

    def test_causal_linear_interpolation(self, df_with_nans):
        imputer = CausalLinearInterpolation(columns=["sensor_1"])
        result = imputer.fit_transform(df_with_nans)
        # No NaNs should remain in sensor_1
        assert result["sensor_1"].isna().sum() == 0

    def test_train_median_fill(self, df_with_nans):
        imputer = TrainMedianFill(columns=["sensor_1", "sensor_2"])
        imputer.fit(df_with_nans)
        result = imputer.transform(df_with_nans)
        # NaNs replaced by training medians
        assert result["sensor_1"].isna().sum() == 0
        assert result["sensor_2"].isna().sum() == 0

    def test_train_median_fill_unfitted(self, df_with_nans):
        imputer = TrainMedianFill(columns=["sensor_1"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            imputer.transform(df_with_nans)

    def test_missingness_indicators(self, df_with_nans):
        ind = MissingnessIndicators(columns=["sensor_1", "sensor_2"])
        ind.fit(df_with_nans)
        result = ind.transform(df_with_nans)
        assert "sensor_1_missing" in result.columns
        assert "sensor_2_missing" in result.columns
        assert result.loc[1, "sensor_1_missing"] == 1
        assert result.loc[0, "sensor_1_missing"] == 0


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


class TestOutliers:
    @pytest.fixture()
    def clean_df(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "sensor_1": rng.normal(10, 1, 200),
                "sensor_2": rng.normal(20, 2, 200),
            }
        )

    def test_iqr_detector_fit_detect(self, clean_df):
        det = IQRFencesDetector(k=1.5)
        det.fit(clean_df, columns=["sensor_1", "sensor_2"])
        flags = det.detect(clean_df)
        assert flags.shape == clean_df.shape
        assert all(flags.dtypes == np.dtype(bool))

    def test_iqr_detector_unfitted(self, clean_df):
        det = IQRFencesDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.detect(clean_df)

    def test_robust_z_detector(self, clean_df):
        det = RobustZDetector(threshold=4.0)
        det.fit(clean_df, columns=["sensor_1"])
        flags = det.detect(clean_df)
        # With clean data and high threshold, very few outliers
        assert flags["sensor_1"].sum() < len(clean_df) * 0.1

    def test_zscore_detector(self, clean_df):
        det = ZScoreDetector(threshold=3.0)
        det.fit(clean_df, columns=["sensor_1"])
        flags = det.detect(clean_df)
        assert flags["sensor_1"].sum() < len(clean_df) * 0.1

    def test_winsorizer(self, clean_df):
        wins = Winsorizer(quantile_low=0.01, quantile_high=0.99)
        wins.fit(clean_df, columns=["sensor_1"])
        result = wins.transform(clean_df)
        lo = clean_df["sensor_1"].quantile(0.01)
        hi = clean_df["sensor_1"].quantile(0.99)
        assert result["sensor_1"].min() >= lo - 1e-10
        assert result["sensor_1"].max() <= hi + 1e-10

    def test_winsorizer_explicit_bounds(self, clean_df):
        wins = Winsorizer(lower={"sensor_1": 5.0}, upper={"sensor_1": 15.0})
        wins.fit(clean_df, columns=["sensor_1"])
        result = wins.transform(clean_df)
        assert result["sensor_1"].min() >= 5.0
        assert result["sensor_1"].max() <= 15.0


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


class TestScaling:
    @pytest.fixture()
    def train_df(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "sensor_1": rng.normal(10, 2, 200),
                "sensor_2": rng.normal(20, 3, 200),
                "op_setting_1": rng.choice([0.5, 0.7, 0.9], 200),
            }
        )

    def test_train_fitted_scaler_standard(self, train_df):
        scaler = TrainFittedScaler(method="standard")
        result = scaler.fit_transform(train_df, columns=["sensor_1", "sensor_2"])
        assert abs(result["sensor_1"].mean()) < 0.1
        assert abs(result["sensor_1"].std() - 1.0) < 0.15

    def test_train_fitted_scaler_robust(self, train_df):
        scaler = TrainFittedScaler(method="robust")
        result = scaler.fit_transform(train_df, columns=["sensor_1"])
        assert abs(result["sensor_1"].median()) < 0.5

    def test_train_fitted_scaler_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            TrainFittedScaler(method="invalid")

    def test_train_fitted_scaler_unfitted(self, train_df):
        scaler = TrainFittedScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(train_df)

    def test_per_condition_scaler(self, train_df):
        scaler = PerConditionScaler(
            method="standard",
            condition_cols=["op_setting_1"],
        )
        result = scaler.fit_transform(train_df, columns=["sensor_1"])
        assert "sensor_1" in result.columns


# ---------------------------------------------------------------------------
# DQ Engine
# ---------------------------------------------------------------------------


class TestDQ:
    def test_dq_flags_decode(self):
        mask = DQFlags.MISSING | DQFlags.STUCK
        decoded = DQFlags.decode(mask)
        assert "missing" in decoded
        assert "stuck" in decoded
        assert "spike" not in decoded

    def test_dq_engine_fit_transform(self, synthetic_dataset):
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]  # noqa: E501
        config = DQConfig(
            sensor_columns=sensor_cols,
            dropout_burst_k=3,
            stuck_window=10,
            spike_threshold=6.0,
        )
        engine = DQEngine(config)
        result = engine.fit_transform(synthetic_dataset)

        assert "dq_badge" in result.columns
        assert "data_quality_degraded" in result.columns
        assert result["dq_badge"].dtype == np.int32

    def test_dq_engine_unfitted(self, synthetic_dataset):
        engine = DQEngine()
        with pytest.raises(RuntimeError, match="not been fitted"):
            engine.transform(synthetic_dataset)

    def test_dq_flags_all_constants(self):
        assert DQFlags.MISSING == 1
        assert DQFlags.DROPOUT_BURST == 2
        assert DQFlags.STUCK == 4
        assert DQFlags.SPIKE == 8
        assert DQFlags.OUT_OF_RANGE == 16
        assert DQFlags.ALL == 0b11111
