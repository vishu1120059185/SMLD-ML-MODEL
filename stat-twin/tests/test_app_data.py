"""Tests for the app-data dashboard artifact builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stattwin.app_data import build_artifacts, state_from_shi_norm
from stattwin.experiments._common import resolve_raw_path


def test_state_from_shi_norm_bands():
    assert state_from_shi_norm(0.05) == "HEALTHY"
    assert state_from_shi_norm(0.30) == "WATCH"
    assert state_from_shi_norm(0.60) == "DEGRADING"
    assert state_from_shi_norm(0.90) == "CRITICAL"


def test_build_artifacts_writes_real_json(tmp_path: Path):
    raw = resolve_raw_path("FD001")
    if not raw.exists():
        pytest.skip("C-MAPSS FD001 train file not present")

    written = build_artifacts("FD001", "MACHINE-001", unit_id=1, out_root=tmp_path)
    assert written

    health_path = tmp_path / "global" / "health_summary.json"
    assert health_path.exists()
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert 0.0 <= float(health["shi"]) <= 1.0
    assert health["provenance"] == "OBSERVED"
    assert health["unit_id"] == 1

    forecast = json.loads(
        (tmp_path / "global" / "failure_forecast.json").read_text(encoding="utf-8")
    )
    assert 0.0 <= float(forecast["failure_prob_30"]) <= 1.0
    assert float(forecast["rul"]) >= 0.0
    assert forecast["horizon_probs"]

    recs = json.loads(
        (tmp_path / "global" / "recommendations.json").read_text(encoding="utf-8")
    )
    assert isinstance(recs, list) and recs
    assert "text" in recs[0]
    assert recs[0]["provenance"] == "PREDICTED"

    timeline = json.loads(
        (tmp_path / "global" / "shi_timeline.json").read_text(encoding="utf-8")
    )
    assert len(timeline["shi"]) == len(timeline["timestamps"])
    assert all(0.0 <= v <= 1.0 for v in timeline["shi"])

    sensors = json.loads(
        (tmp_path / "global" / "sensor_data.json").read_text(encoding="utf-8")
    )
    assert sensors["sensors"]
    first = next(iter(sensors["sensors"].values()))
    assert len(first["values"]) == len(first["timestamps"])

    dq = json.loads(
        (tmp_path / "global" / "data_quality.json").read_text(encoding="utf-8")
    )
    assert 0.0 <= dq["overall"] <= 1.0


def test_build_artifacts_fails_loudly_when_missing(tmp_path: Path, monkeypatch):
    import stattwin.experiments._common as common

    monkeypatch.setattr(
        common,
        "resolve_raw_path",
        lambda ds: tmp_path / f"train_{ds}.txt",
    )
    # Also patch the reference imported into app_data
    import stattwin.app_data as ad

    monkeypatch.setattr(ad, "resolve_raw_path", lambda ds: tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError):
        ad.build_artifacts("FD001", "MACHINE-001", out_root=tmp_path)
