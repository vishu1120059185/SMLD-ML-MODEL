"""Leakage Guard: Same seed and config -> identical metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_dataset
from stattwin.data.splitter import make_group_kfold_splits


class TestDeterminism:
    def test_synthetic_dataset_deterministic(self):
        """Same seed produces identical synthetic data."""
        df1 = make_synthetic_dataset(n_units=5, seed=42)
        df2 = make_synthetic_dataset(n_units=5, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_dataset_different_seeds(self):
        """Different seeds produce different data."""
        df1 = make_synthetic_dataset(n_units=5, seed=42)
        df2 = make_synthetic_dataset(n_units=5, seed=99)
        assert not df1.equals(df2)

    def test_split_deterministic(self, synthetic_dataset):
        """Same seed produces identical splits."""
        splits1 = make_group_kfold_splits(synthetic_dataset, n_splits=3, seed=42)
        splits2 = make_group_kfold_splits(synthetic_dataset, n_splits=3, seed=42)
        for s1, s2 in zip(splits1, splits2):
            np.testing.assert_array_equal(s1["train_units"], s2["train_units"])
            np.testing.assert_array_equal(s1["val_units"], s2["val_units"])

    def test_split_different_n_splits(self, synthetic_dataset):
        """Different n_splits produces different fold structures."""
        splits3 = make_group_kfold_splits(synthetic_dataset, n_splits=3)
        splits5 = make_group_kfold_splits(synthetic_dataset, n_splits=5)
        assert len(splits3) != len(splits5)

    def test_metrics_deterministic(self, synthetic_dataset):
        """Same data and seed produce identical metric results."""
        from stattwin.health.shi import compute_shi
        from stattwin.health.quality import compute_quality_metrics

        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]

        hi1 = compute_shi(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)
        hi2 = compute_shi(synthetic_dataset, sensor_cols=sensor_cols, baseline_cycles=20)

        pd.testing.assert_frame_equal(hi1.shi_values, hi2.shi_values)
