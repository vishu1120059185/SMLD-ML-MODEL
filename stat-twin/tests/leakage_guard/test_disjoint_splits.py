"""Leakage Guard: No unit_id appears in more than one split within a fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.splitter import make_group_kfold_splits


class TestDisjointSplits:
    def test_no_unit_in_multiple_val_folds(self, synthetic_dataset):
        """Each unit appears in at most one validation fold."""
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=5)
        seen_val_units: set = set()
        for i, s in enumerate(splits):
            val_set = set(s["val_units"])
            overlap = val_set & seen_val_units
            assert not overlap, f"Fold {i} shares val units with previous folds: {overlap}"
            seen_val_units.update(val_set)

    def test_all_units_covered(self, synthetic_dataset):
        """Every unit appears in exactly one val fold."""
        all_units = set(synthetic_dataset["unit_id"].unique())
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=5)
        all_val_units = set()
        for s in splits:
            all_val_units.update(s["val_units"])
        assert all_val_units == all_units

    def test_train_and_val_disjoint(self, synthetic_dataset):
        """Train and val unit sets are disjoint within each fold."""
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=5)
        for s in splits:
            overlap = set(s["train_units"]) & set(s["val_units"])
            assert not overlap, f"Train/val overlap: {overlap}"

    def test_rows_assigned_correctly(self, synthetic_dataset):
        """Rows in val split only contain val unit_ids."""
        splits = make_group_kfold_splits(synthetic_dataset, n_splits=5)
        for s in splits:
            val_units = set(s["val_units"])
            val_rows = synthetic_dataset[synthetic_dataset["unit_id"].isin(val_units)]
            assert len(val_rows) > 0
            assert set(val_rows["unit_id"].unique()) == val_units
