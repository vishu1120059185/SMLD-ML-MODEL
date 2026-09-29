"""Shared helpers for all STAT-TWIN experiments.

Every experiment script imports ``experiment_setup`` to load config,
load data, and prepare the output directory tree.  This avoids
duplicating boilerplate across e0–e9.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stattwin.config import STATTWINConfig, load_config
from stattwin.data.loader import load_cmapss
from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
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
            f"Download from https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository\n"  # noqa: E501
            f"and place '{cfg.dataset.name}.txt' in data/raw/CMAPSS/"
        )

    df = load_cmapss(raw_path, add_labels=True, rul_clip=cfg.dataset.rul_clip)

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

    def __enter__(self) -> Timer:
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


# ---------------------------------------------------------------------------
# Feature hygiene + screening
# ---------------------------------------------------------------------------

def impute_causal(
    frames: list[pd.DataFrame],
    feature_cols: list[str],
    unit_col: str = "unit_id",
) -> None:
    """Fill missing feature values **in place**, causally and train-only.

    Applied to every frame in *frames* in the given order — the first
    frame **must be the training portion**:

    1. forward-fill within each unit — uses only rows ``<= t``, so a
       leading ``NaN`` (no history yet) stays ``NaN`` while interior gaps
       inherit the previous cycle of the *same* unit;
    2. fill whatever remains with the column median of the
       **forward-filled training frame** (never validation rows), falling
       back to ``0.0`` for an all-missing column.

    The engineered library leaves ``NaN`` in the leading cycles of a unit
    (rolling / correlation windows have no history yet) — 1 029 of 1 117
    columns on FD001.  Tree models tolerate ``NaN``, but linear models and
    the ensemble weight search do not, so this must run before any fit.
    """
    present = [c for c in feature_cols if len(frames) and c in frames[0].columns]
    if not present:
        return
    for frame in frames:
        if unit_col in frame.columns:
            frame[present] = frame.groupby(unit_col, sort=False)[present].ffill()
    medians = frames[0][present].median(numeric_only=True).fillna(0.0)
    for frame in frames:
        frame[present] = frame[present].fillna(medians)


def screen_features(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feature_cols: list[str],
    *,
    top_k: int = 300,
    min_keep: int = 30,
) -> list[str]:
    """Return the ``top_k`` most target-associated columns (train-only).

    Score = max(|corr| with RUL, mean |corr| across failure horizons),
    computed with vectorised Pearson correlation on the **training units
    only**, so validation units never influence the selection.  Raw sensor
    columns are always retained — they are the physical signal, and they
    are cheap to keep.
    """
    present = [c for c in feature_cols if c in X_train.columns]
    if top_k <= 0 or len(present) <= top_k:
        return feature_cols

    block = X_train[present]
    scores = pd.Series(0.0, index=present)

    targets: list[pd.Series] = []
    if "RUL" in X_train.columns:
        targets.append(X_train["RUL"].astype(float))
    for col in y_train.columns:
        if col in X_train.columns or col in y_train.columns:
            targets.append(y_train[col].reindex(block.index).astype(float))

    for target in targets:
        if target.isna().all():
            continue
        with warnings.catch_warnings():
            # constant columns give a 0/0 correlation; they are simply
            # uninformative and must not raise or pollute the ranking
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = block.corrwith(target)
        scores = scores.combine(corr.abs().fillna(0.0), max)

    raw = [c for c in present if c.startswith("sensor_") or c.startswith("op_setting_")]
    ranked = [c for c in scores.sort_values(ascending=False).index if c not in raw]
    keep = raw + ranked[: max(top_k - len(raw), min_keep)]
    keep = [c for c in present if c in set(keep)]
    return keep[: max(top_k, min_keep)]


def resolve_feature_cols(
    cfg,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    """Apply the configured per-fold screening policy."""
    screen = cfg.model.feature_screen
    if not screen.enabled or screen.method == "none" or screen.top_k <= 0:
        return feature_cols
    return screen_features(
        X_train,
        y_train,
        feature_cols,
        top_k=screen.top_k,
        min_keep=screen.min_keep,
    )


def setup_output(experiment_name: str) -> Path:
    """Create ``results/<experiment>/`` and ``results/<experiment>/figures/``."""
    out = _PROJECT_ROOT / "results" / experiment_name
    ensure_dir(out)
    ensure_dir(out / "figures")
    return out
