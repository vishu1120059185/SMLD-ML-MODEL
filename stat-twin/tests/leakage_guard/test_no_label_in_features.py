"""Leakage Guard: No feature name or derivation depends on RUL (blacklist)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_dataset
from stattwin.statistics.feature_library import compute_features


# Patterns that would indicate RUL leakage in feature names
RUL_LEAKAGE_PATTERNS = [
    re.compile(r"\brul\b", re.IGNORECASE),
    re.compile(r"\bfail_h\d+\b", re.IGNORECASE),
    re.compile(r"\by_\d+\b", re.IGNORECASE),
    re.compile(r"\blabel\b", re.IGNORECASE),
    re.compile(r"\btarget\b", re.IGNORECASE),
]


class TestNoLabelInFeatures:
    def test_no_rul_in_feature_names(self, synthetic_dataset):
        """No generated feature column name should contain 'rul', 'fail_h', etc."""
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        out_df, spec = compute_features(synthetic_dataset, sensor_cols=sensor_cols)

        for feat_col in spec.all_feature_cols:
            for pattern in RUL_LEAKAGE_PATTERNS:
                assert not pattern.search(feat_col), (
                    f"Feature '{feat_col}' matches RUL leakage pattern '{pattern.pattern}'"
                )

    def test_no_rul_in_derived_columns(self, synthetic_dataset):
        """No generated column should derive from the RUL column."""
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        out_df, spec = compute_features(synthetic_dataset, sensor_cols=sensor_cols)

        for feat_col in spec.all_feature_cols:
            assert feat_col != "RUL"
            assert "RUL_" not in feat_col
            assert not feat_col.startswith("RUL")

    def test_original_rul_column_not_used(self, synthetic_dataset):
        """The original RUL column should not appear in the feature matrix."""
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        out_df, spec = compute_features(synthetic_dataset, sensor_cols=sensor_cols)

        # RUL should still exist but not be in feature cols
        assert "RUL" not in spec.all_feature_cols

    def test_label_columns_not_in_features(self, synthetic_dataset):
        """Binary label columns (fail_h*) should not be feature columns."""
        from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
        sensor_cols = [c for c in synthetic_dataset.columns if c.startswith("sensor_") and synthetic_dataset[c].sum() != 0][:3]
        out_df, spec = compute_features(synthetic_dataset, sensor_cols=sensor_cols)

        for h in FAILURE_HORIZONS:
            assert label_col_for(h) not in spec.all_feature_cols
