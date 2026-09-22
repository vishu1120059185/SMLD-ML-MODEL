"""Tests for data loading, schema validation, RUL computation, and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.loader import load_cmapss
from stattwin.data.schema import (
    COLUMN_NAMES,
    FAILURE_HORIZONS,
    SENSOR_NAMES,
    label_col_for,
)
from stattwin.data.splitter import (
    inner_unit_split,
    make_group_kfold_splits,
    validate_dataset,
)
from stattwin.data.synthetic import make_synthetic_dataset, make_synthetic_unit


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_column_count(self):
        assert len(COLUMN_NAMES) == 26

    def test_sensor_names_cover_21_sensors(self):
        assert len(SENSOR_NAMES) == 21

    def test_label_col_for(self):
        assert label_col_for(10) == "fail_h10"
        assert label_col_for(50) == "fail_h50"

    def test_failure_horizons(self):
        assert FAILURE_HORIZONS == [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


class TestSynthetic:
    def test_single_unit_shape(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100)
        assert len(df) == 100
        assert set(COLUMN_NAMES).issubset(df.columns)
        assert "RUL" in df.columns

    def test_single_unit_rul(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100)
        assert df["RUL"].iloc[0] == 99
        assert df["RUL"].iloc[-1] == 0

    def test_single_unit_labels_present(self):
        df = make_synthetic_unit(unit_id=1, n_cycles=100)
        for h in FAILURE_HORIZONS:
            assert label_col_for(h) in df.columns

    def test_single_unit_min_cycles(self):
        with pytest.raises(ValueError):
            make_synthetic_unit(unit_id=1, n_cycles=1)

    def test_multi_unit_dataset(self):
        df = make_synthetic_dataset(n_units=5, seed=42)
        assert df["unit_id"].nunique() == 5
        assert len(df) > 0

    def test_multi_unit_sorted(self):
        df = make_synthetic_dataset(n_units=5, seed=42)
        sorted_df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df, sorted_df)


# ---------------------------------------------------------------------------
# Loader (from synthetic data written to disk)
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_cmapss(self, temp_dir):
        df = make_synthetic_unit(unit_id=1, n_cycles=50)
        path = temp_dir / "test.txt"
        df[COLUMN_NAMES].to_csv(path, sep=" ", header=False, index=False)

        loaded = load_cmapss(path)
        assert len(loaded) == 50
        assert "RUL" in loaded.columns

    def test_load_cmapss_with_labels(self, temp_dir):
        df = make_synthetic_unit(unit_id=1, n_cycles=50)
        path = temp_dir / "test.txt"
        df[COLUMN_NAMES].to_csv(path, sep=" ", header=False, index=False)

        loaded = load_cmapss(path, add_labels=True)
        for h in FAILURE_HORIZONS:
            assert label_col_for(h) in loaded.columns

    def test_load_cmapss_no_labels(self, temp_dir):
        df = make_synthetic_unit(unit_id=1, n_cycles=50)
        path = temp_dir / "test.txt"
        df[COLUMN_NAMES].to_csv(path, sep=" ", header=False, index=False)

        loaded = load_cmapss(path, add_labels=False)
        assert "RUL" in loaded.columns
        for h in FAILURE_HORIZONS:
            assert label_col_for(h) not in loaded.columns

    def test_load_cmapss_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_cmapss("/nonexistent/file.txt")

    def test_rul_computation(self, temp_dir):
        df = make_synthetic_unit(unit_id=1, n_cycles=60)
        path = temp_dir / "test.txt"
        df[COLUMN_NAMES].to_csv(path, sep=" ", header=False, index=False)

        loaded = load_cmapss(path)
        max_cycle = loaded["cycle"].max()
        assert loaded["RUL"].max() == max_cycle - loaded["cycle"].min()
        assert loaded["RUL"].min() == 0


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplitter:
    def test_group_kfold_splits(self, synthetic_dataset):
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=3)
        assert len(splits) == 3
        all_val_units = set()
        for s in splits:
            assert "train_units" in s
            assert "val_units" in s
            val_set = set(s["val_units"])
            assert not val_set & all_val_units, "Val units overlap across folds"
            all_val_units.update(val_set)

    def test_inner_unit_split(self):
        train_units = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        inner_train, inner_val = inner_unit_split(train_units, val_frac=0.2, seed=42)
        assert len(inner_train) + len(inner_val) == len(train_units)
        assert len(set(inner_train) & set(inner_val)) == 0

    def test_validate_dataset_valid(self, synthetic_dataset):
        validate_dataset(synthetic_dataset)

    def test_validate_dataset_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataset(pd.DataFrame())

    def test_validate_dataset_no_unit_id(self):
        df = pd.DataFrame({"cycle": [1, 2], "x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="unit_id"):
            validate_dataset(df)

    def test_validate_dataset_requires_rul(self, synthetic_dataset):
        validate_dataset(synthetic_dataset, require_rul=True)

    def test_validate_dataset_requires_labels(self, synthetic_dataset):
        validate_dataset(synthetic_dataset, require_labels=True)
