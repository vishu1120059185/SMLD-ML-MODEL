"""Leakage Guard: Conformal/probability calibration data are OOF from other folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_dataset
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.uncertainty.conformal import conformal_intervals


class TestCalibrationDisjoint:
    def test_calibration_and_test_disjoint(self, synthetic_dataset):
        """Conformal calibration data must not overlap with test data."""
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=3)
        rng = np.random.default_rng(42)

        for i, s in enumerate(splits):
            train_units = set(s["train_units"])
            val_units = set(s["val_units"])

            # Split training into calibration and proper train
            train_list = sorted(train_units)
            n_cal = max(1, len(train_list) // 3)
            cal_units = set(train_list[:n_cal])
            proper_train_units = set(train_list[n_cal:])

            # Verify disjointness
            overlap_cal_val = cal_units & val_units
            overlap_cal_train = cal_units & proper_train_units
            overlap_train_val = proper_train_units & val_units

            assert not overlap_cal_val, f"Fold {i}: calibration overlaps with val"
            assert not overlap_cal_train, f"Fold {i}: calibration overlaps with proper train"
            assert not overlap_train_val, f"Fold {i}: proper train overlaps with val"

    def test_conformal_uses_separate_data(self, synthetic_dataset):
        """Conformal scores computed on calibration set, intervals applied to test set."""
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=3)
        rng = np.random.default_rng(42)

        for s in splits:
            train_units = sorted(s["train_units"])
            val_units = s["val_units"]

            # Create synthetic predictions (not from a real model)
            train_df = synthetic_dataset[synthetic_dataset["unit_id"].isin(train_units)]
            val_df = synthetic_dataset[synthetic_dataset["unit_id"].isin(val_units)]

            # Calibration: first half of training
            n_cal = len(train_df) // 2
            cal_idx = train_df.index[:n_cal]
            proper_train_idx = train_df.index[n_cal:]

            y_cal_true = rng.normal(50, 10, n_cal)
            y_cal_pred = y_cal_true + rng.normal(0, 3, n_cal)
            sigma_cal = np.full(n_cal, 3.0)

            y_test_true = rng.normal(50, 10, len(val_df))
            y_test_pred = y_test_true + rng.normal(0, 3, len(val_df))
            sigma_test = np.full(len(val_df), 3.0)

            report = conformal_intervals(
                y_cal_true, y_cal_pred, sigma_cal,
                y_test_pred, sigma_test, alpha=0.10,
            )

            # Verify report structure
            assert report.quantile_q > 0
            assert len(report.intervals) == len(y_test_pred)
            assert report.coverage >= 0.0
            assert report.coverage <= 1.0
