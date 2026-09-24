"""Shared dashboard artifact loading.

All views read JSON artifacts from ``results/`` via this module so that
search order, caching, and freshness tracking stay consistent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = [
    "RESULTS_DIR",
    "artifact_freshness",
    "load_artifact",
    "write_artifact",
]

RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"

_FRESH_KEY = "_stattwin_artifact_mtime::"

_ARTIFACT_NAMES = frozenset(
    {
        "health_summary.json",
        "failure_forecast.json",
        "recommendations.json",
        "data_quality.json",
        "shi_timeline.json",
        "risk_timeline.json",
        "sensor_data.json",
        "dq_flags.json",
        "explanation.json",
        "contributions.json",
        "evidence.json",
        "z_heatmap.json",
        "variance_change.json",
        "distribution_shift.json",
        "correlation_matrix.json",
        "warning_timeline.json",
        "conformal_bands.json",
        "model_comparison.json",
        "meta.json",
    }
)


def load_artifact(name: str, machine: str | None = None) -> Any | None:
    """Load a JSON artifact.

    Search order: ``results/<machine>/``, ``results/global/``, ``results/``.
    Returns ``None`` when the file is missing or unparsable.
    """
    candidates: list[Path] = []
    if machine:
        candidates.append(RESULTS_DIR / machine / name)
    candidates.append(RESULTS_DIR / "global" / name)
    candidates.append(RESULTS_DIR / name)
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                import streamlit as st

                st.session_state[_FRESH_KEY + name] = path.stat().st_mtime
            except Exception:
                pass
            return data
    return None


def artifact_freshness(name: str) -> float | None:
    """Seconds since the artifact was last written, or ``None`` if unknown."""
    try:
        import streamlit as st

        mtime = st.session_state.get(_FRESH_KEY + name)
        if mtime is None:
            return None
        return max(0.0, time.time() - float(mtime))
    except Exception:
        return None


def write_artifact(name: str, payload: Any, *, machine: str | None = None) -> Path:
    """Write *payload* under ``results/`` (machine subdir when given)."""
    if name not in _ARTIFACT_NAMES and not name.endswith(".json"):
        raise ValueError(f"Unexpected artifact name: {name}")
    out_dir = RESULTS_DIR / machine if machine else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path
