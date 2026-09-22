"""Shared helpers for all STAT-TWIN experiments.

Every experiment script imports ``experiment_setup`` to load config,
load data, and prepare the output directory tree.  This avoids
duplicating boilerplate across e0–e9.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stattwin.config import STATTWINConfig, load_config
from stattwin.data.loader import load_cmapss
from stattwin.data.schema import COLUMN_NAMES, FAILURE_HORIZONS, OP_SETTINGS, label_col_for
from stattwin.manifest import write_manifest

# ---------------------------------------------------------------------------
# Project root (three levels up from this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard ``--ds``, ``--profile``, and ``--config`` arguments."""
    parser.add_argument(
        "--ds", default="FD001",
        help="C-MAPSS dataset name (default: FD001)",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Config profile (e.g. smoke, fast, full)",
    )
    parser.add_argument(
        "--config", default=str(_PROJECT_ROOT / "configs" / "base.yaml"),
        help="Path to base config YAML",
    )
    return parser


def resolve_raw_path(ds_name: str) -> Path:
    """Return the expected train raw-data file path for a C-MAPSS dataset."""
    return _PROJECT_ROOT / "data" / "raw" / "CMAPSS" / f"train_{ds_name.upper()}.txt"


def experiment_setup(args: argparse.Namespace) -> tuple[STATTWINConfig, pd.DataFrame, Path]:
    """Load config, load data, and prepare the output directory.

    Returns
    -------
    (cfg, df, out_dir)
        The validated config, the raw (labelled) DataFrame, and the
        experiment output directory (created if missing).
    """
    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)

    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw C-MAPSS file not found: {raw_path}\n"
            f"Download from https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository\n"
            f"and place '{cfg.dataset.name}.txt' in data/raw/CMAPSS/"
        )

    df = load_cmapss(raw_path, add_labels=True)

    exp_tag = Path(__file__).stem  # overridden by caller
    return cfg, df, _PROJECT_ROOT / "results" / exp_tag


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path) -> None:
    """Serialise *data* to a JSON file (float precision: 6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for numpy / pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_manifest(
    out_dir: Path,
    cfg: STATTWINConfig,
    config_path: str,
    experiment_name: str,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` into *out_dir*."""
    meta = extras or {}
    meta["experiment"] = experiment_name
    return write_manifest(out_dir, cfg, config_path, extras=meta)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


# ---------------------------------------------------------------------------
# Sensor / feature helpers
# ---------------------------------------------------------------------------

SENSOR_COLS: list[str] = [f"sensor_{i}" for i in range(1, 22)]
META_COLS: list[str] = ["unit_id", "cycle"]
LABEL_COLS: list[str] = [label_col_for(h) for h in FAILURE_HORIZONS]


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return sensor columns present in *df*."""
    return [c for c in SENSOR_COLS if c in df.columns]


def feature_columns(df: pd.DataFrame, include_health: bool = True) -> list[str]:
    """Return all numeric feature columns (excluding meta + label cols)."""
    exclude = set(META_COLS) | set(LABEL_COLS) | {"RUL"}
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not include_health:
        cols = [c for c in cols if "shi" not in c.lower() and "evidence_" not in c.lower()]
    return cols


def setup_output(experiment_name: str) -> Path:
    """Create ``results/<experiment>/`` and ``results/<experiment>/figures/``."""
    out = _PROJECT_ROOT / "results" / experiment_name
    ensure_dir(out)
    ensure_dir(out / "figures")
    return out
