"""Leakage Guard: Warning tau and state thresholds never fitted on test units."""

from __future__ import annotations

from stattwin.data.splitter import make_group_kfold_splits
from stattwin.data.synthetic import make_synthetic_dataset
from stattwin.health.shi import compute_shi
from stattwin.health.states import StateClassifier


class TestThresholdSelectedOnVal:
    def test_state_thresholds_from_train_only(self, synthetic_dataset):
        """StateClassifier thresholds should be calibrated on training data, not test data."""
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=3)
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]  # noqa: E501

        for s in splits:
            train_units = s["train_units"]
            val_units = s["val_units"]

            train_df = synthetic_dataset[synthetic_dataset["unit_id"].isin(train_units)]
            val_df = synthetic_dataset[synthetic_dataset["unit_id"].isin(val_units)]

            # Compute SHI on train
            hi_train = compute_shi(train_df, sensor_cols=sensor_cols, baseline_cycles=20)

            # Calibrate classifier on train SHI
            clf = StateClassifier(method="calibrated", persistence=3)
            clf.calibrate(hi_train.shi_values, rul_df=train_df[["unit_id", "cycle", "RUL"]])
            train_thresholds = clf.thresholds.copy()

            # Compute SHI on val
            hi_val = compute_shi(val_df, sensor_cols=sensor_cols, baseline_cycles=20)

            # Classify val using train-calibrated thresholds
            result = clf.classify(hi_val.shi_values)

            # Thresholds should not have changed
            assert clf.thresholds == train_thresholds

            # All states should be valid
            from stattwin.health.states import HealthState
            valid_states = set(s.value for s in HealthState)
            assert result["health_state"].isin(valid_states).all()

    def test_fixed_grid_not_fitted(self, synthetic_dataset):
        """Fixed-grid thresholds should not depend on any data."""
        clf1 = StateClassifier(method="fixed_grid", fixed_grid=[80, 60, 40, 20])
        clf2 = StateClassifier(method="fixed_grid", fixed_grid=[80, 60, 40, 20])

        # Both should have same thresholds regardless of data
        df1 = make_synthetic_dataset(n_units=3, seed=42)
        df2 = make_synthetic_dataset(n_units=3, seed=99)

        hi1 = compute_shi(df1, sensor_cols=["sensor_1", "sensor_2", "sensor_3"], baseline_cycles=20)
        hi2 = compute_shi(df2, sensor_cols=["sensor_1", "sensor_2", "sensor_3"], baseline_cycles=20)

        clf1.calibrate(hi1.shi_values)
        clf2.calibrate(hi2.shi_values)

        assert clf1.thresholds == clf2.thresholds
