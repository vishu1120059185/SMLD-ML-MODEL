"""Shared fixtures for STAT-TWIN test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stattwin.config import STATTWINConfig
from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.data.synthetic import make_synthetic_dataset, make_synthetic_unit


@pytest.fixture()
def rng():
    """Global deterministic RNG."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_dataset():
    """Multi-unit synthetic C-MAPSS-like DataFrame (10 units, 120-250 cycles)."""
    return make_synthetic_dataset(
        n_units=10,
        n_cycles_range=(120, 250),
        n_sensors=5,
        drift_slope=0.01,
        noise_std=0.05,
        seed=42,
    )


@pytest.fixture()
def small_synthetic_dataset():
    """Smaller dataset for fast tests (5 units, 80-150 cycles, 3 sensors)."""
    return make_synthetic_dataset(
        n_units=5,
        n_cycles_range=(80, 150),
        n_sensors=3,
        drift_slope=0.02,
        noise_std=0.05,
        seed=42,
    )


@pytest.fixture()
def single_unit():
    """Single synthetic unit trajectory (200 cycles)."""
    return make_synthetic_unit(
        unit_id=1,
        n_cycles=200,
        n_sensors=5,
        drift_slope=0.01,
        noise_std=0.05,
        seed=42,
    )


@pytest.fixture()
def sample_config():
    """Minimal STATTWINConfig for tests."""
    return STATTWINConfig(
        seed=42,
        split={"n_splits": 3},
        stats={"windows": [5, 10], "ewma_alpha": [0.3], "baseline_cycles": 20},
        forecast={"horizons": [10, 20, 30, 40, 50]},
        uncertainty={"alpha": 0.10},
    )


@pytest.fixture()
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def synthetic_features(synthetic_dataset):
    """Synthetic dataset with a few rolling features pre-computed."""
    from stattwin.statistics.rolling import rolling_mean, rolling_std

    df = synthetic_dataset.copy()
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0]
    for col in sensor_cols[:3]:
        df = rolling_mean(df, col, windows=[5, 10])
        df = rolling_std(df, col, windows=[5, 10])
    return df


@pytest.fixture()
def feature_and_label_cols(synthetic_features):
    """Return (feature_cols, label_cols, sensor_cols) from synthetic_features."""
    df = synthetic_features
    exclude = {"unit_id", "cycle", "RUL"}
    exclude.update(label_col_for(h) for h in FAILURE_HORIZONS)
    exclude.update(c for c in df.columns if c.startswith("op_setting_"))
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0]
    return feature_cols, label_cols, sensor_cols
