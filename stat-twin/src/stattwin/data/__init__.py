"""Data module for STAT-TWIN.

Provides C-MAPSS file loading, column schemas, unit-level splitting,
synthetic fixtures, and the dataset abstraction used throughout the pipeline.
"""

from stattwin.data.datasets import TelemetryDataset
from stattwin.data.loader import load_cmapss
from stattwin.data.schema import COLUMN_NAMES, FAILURE_HORIZONS, SENSOR_NAMES
from stattwin.data.splitter import (
    inner_unit_split,
    make_group_kfold_splits,
    validate_dataset,
)
from stattwin.data.synthetic import make_synthetic_dataset, make_synthetic_unit

__all__ = [
    "COLUMN_NAMES",
    "FAILURE_HORIZONS",
    "SENSOR_NAMES",
    "TelemetryDataset",
    "inner_unit_split",
    "load_cmapss",
    "make_group_kfold_splits",
    "make_synthetic_dataset",
    "make_synthetic_unit",
    "validate_dataset",
]
