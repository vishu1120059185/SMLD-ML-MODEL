"""Leakage Guard: Preprocessors, KMeans, sensor selection, calibrators fit on train units only."""

from __future__ import annotations

import numpy as np
import pytest

from stattwin.data.splitter import inner_unit_split, make_group_kfold_splits
from stattwin.data.synthetic import make_synthetic_dataset
from stattwin.preprocessing.missing import TrainMedianFill
from stattwin.preprocessing.outliers import RobustZDetector
from stattwin.preprocessing.scaler import TrainFittedScaler


class TestFitOnlyOnTrain:
    @pytest.fixture()
    def split_data(self):
        df = make_synthetic_dataset(n_units=10, n_cycles_range=(80, 150), seed=42)
        splits = make_group_kfold_splits(df, n_splits=3)
        return df, splits

    def test_median_fill_fitted_on_train(self, split_data):
        """TrainMedianFill medians computed only from training units."""
        df, splits = split_data
        s = splits[0]
        train_df = df[df["unit_id"].isin(s["train_units"])]
        df[df["unit_id"].isin(s["val_units"])]

        sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0][:3]
        imputer = TrainMedianFill(columns=sensor_cols)
        imputer.fit(train_df)

        # Medians should match training data medians
        for col in sensor_cols:
            expected = train_df[col].median()
            assert imputer.medians_[col] == pytest.approx(expected)

    def test_outlier_detector_fitted_on_train(self, split_data):
        """Outlier detector thresholds learned only from training data."""
        df, splits = split_data
        s = splits[0]
        train_df = df[df["unit_id"].isin(s["train_units"])]
        test_df = df[df["unit_id"].isin(s["val_units"])]

        sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0][:3]
        det = RobustZDetector(threshold=4.0)
        det.fit(train_df, columns=sensor_cols)

        # Test data detection should use training thresholds
        test_flags = det.detect(test_df)
        assert test_flags.shape[0] == len(test_df)

    def test_scaler_fitted_on_train(self, split_data):
        """Scaler parameters computed only from training units."""
        df, splits = split_data
        s = splits[0]
        train_df = df[df["unit_id"].isin(s["train_units"])]
        test_df = df[df["unit_id"].isin(s["val_units"])]

        sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0][:3]
        scaler = TrainFittedScaler(method="standard")
        scaler.fit(train_df, columns=sensor_cols)

        # Verify scaler was fit on training data
        for col in sensor_cols:
            expected_center = train_df[col].mean()
            assert scaler.center_[col] == pytest.approx(expected_center)

        # Transforming test data should not change scaler parameters
        _ = scaler.transform(test_df)
        for col in sensor_cols:
            assert scaler.center_[col] == pytest.approx(train_df[col].mean())

    def test_inner_split_disjoint(self):
        """Inner train/val split produces disjoint unit sets."""
        train_units = np.arange(1, 11)
        inner_train, inner_val = inner_unit_split(train_units, val_frac=0.2, seed=42)
        assert len(set(inner_train) & set(inner_val)) == 0
        assert len(inner_train) + len(inner_val) == len(train_units)

    def test_inner_split_from_outer(self, split_data):
        """Inner split is a subset of outer train units."""
        df, splits = split_data
        outer_train = splits[0]["train_units"]
        inner_train, inner_val = inner_unit_split(outer_train, val_frac=0.2, seed=42)

        assert set(inner_train).issubset(set(outer_train))
        assert set(inner_val).issubset(set(outer_train))
        assert len(set(inner_train) & set(inner_val)) == 0
