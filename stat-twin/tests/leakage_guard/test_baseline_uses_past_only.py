"""Leakage Guard: Healthy baseline uses only first K cycles; warm-up rows flagged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.synthetic import make_synthetic_unit
from stattwin.statistics.rolling import z_score_vs_baseline


class TestBaselineUsesPastOnly:
    def test_baseline_only_first_k_cycles(self):
        """Z-score baseline uses only the first K cycles, not future data."""
        df = make_synthetic_unit(unit_id=1, n_cycles=100, seed=42)
        K = 30

        result = z_score_vs_baseline(df, "sensor_1", baseline_cycles=K)

        # The baseline mean/std should be identical to the first K cycles' stats
        first_k = df["sensor_1"].iloc[:K]
        baseline_mean = first_k.mean()
        baseline_std = first_k.std(ddof=1)

        # Verify the z-score at cycle K+1 uses only first K values
        t = K + 5
        zscore_at_t = result.loc[result["cycle"] == t, "sensor_1_zscore"].iloc[0]
        val_at_t = df.loc[df["cycle"] == t, "sensor_1"].iloc[0]

        # Manual computation
        expected_z = (val_at_t - baseline_mean) / baseline_std
        assert zscore_at_t == pytest.approx(expected_z, rel=1e-6)

    def test_warmup_period_uses_expanding(self):
        """During warmup (t <= K), z-score uses expanding baseline."""
        df = make_synthetic_unit(unit_id=1, n_cycles=50, seed=42)
        K = 30

        result = z_score_vs_baseline(df, "sensor_1", baseline_cycles=K)

        # At cycle 1, baseline is just [x1]
        t1_zscore = result.loc[result["cycle"] == 1, "sensor_1_zscore"].iloc[0]
        x1 = df.loc[df["cycle"] == 1, "sensor_1"].iloc[0]
        # With only 1 value, sigma=1.0, so z = (x1 - x1)/1 = 0
        assert t1_zscore == pytest.approx(0.0, abs=0.1)

    def test_zscore_causality_multiple_units(self):
        """Each unit's z-score is independent (no cross-unit leakage)."""
        from stattwin.data.synthetic import make_synthetic_dataset

        df = make_synthetic_dataset(n_units=3, n_cycles_range=(50, 80), seed=42)
        sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0][:1]
        col = sensor_cols[0]

        result = z_score_vs_baseline(df, col, baseline_cycles=20)

        # Verify each unit's z-scores are independent
        for uid in df["unit_id"].unique():
            unit_result = result[result["unit_id"] == uid]
            unit_data = df[df["unit_id"] == uid]
            K = min(20, len(unit_data))
            bl_mean = unit_data[col].iloc[:K].mean()
            bl_std = unit_data[col].iloc[:K].std(ddof=1)

            # Check a cycle well after baseline
            if len(unit_data) > K + 5:
                t = K + 5
                z = unit_result.loc[unit_result["cycle"] == t, f"{col}_zscore"].iloc[0]
                val = unit_data.loc[unit_data["cycle"] == t, col].iloc[0]
                expected = (val - bl_mean) / bl_std
                assert z == pytest.approx(expected, rel=1e-6)
