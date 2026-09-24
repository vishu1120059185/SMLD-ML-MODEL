"""Tests for decision guidance, shared artifact loader, and comparison payload."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from stattwin.dashboard.components.artifacts import (
    RESULTS_DIR,
)
from stattwin.decision.guidance import (
    RiskThresholds,
    RiskTier,
    generate_maintenance_guidance,
)


def _proba(p30: float) -> pd.DataFrame:
    return pd.DataFrame([{"fail_h30": p30}])


def test_low_risk_defaults_to_low_tier():
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.02), horizons=[30], health_state="HEALTHY"
    )
    assert g.risk_tier is RiskTier.LOW
    assert g.recommendation
    assert "routine monitoring" in g.recommendation.lower()


def test_medium_threshold():
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.15), horizons=[30], health_state="WATCH"
    )
    assert g.risk_tier is RiskTier.MEDIUM


def test_high_threshold():
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.40), horizons=[30], health_state="WATCH"
    )
    assert g.risk_tier is RiskTier.HIGH


def test_critical_threshold():
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.85), horizons=[30], health_state="WATCH"
    )
    assert g.risk_tier is RiskTier.CRITICAL


def test_state_override_forces_high():
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.01), horizons=[30], health_state="DEGRADING"
    )
    assert g.risk_tier is RiskTier.HIGH
    assert "DEGRADING" in g.rule_triggered


def test_state_override_forces_critical():
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.01), horizons=[30], health_state="FAILURE_LIKELY"
    )
    assert g.risk_tier is RiskTier.CRITICAL


def test_degraded_dq_lowers_confidence_not_tier():
    g = generate_maintenance_guidance(
        unit_id=1,
        cycle=10,
        proba=_proba(0.15),
        horizons=[30],
        health_state="WATCH",
        dq_status="DEGRADED",
    )
    assert g.risk_tier is RiskTier.MEDIUM
    assert "Reduced" in g.confidence_level


def test_custom_thresholds():
    thr = RiskThresholds(p30_low=0.05, p30_medium=0.10, p30_critical=0.20)
    g = generate_maintenance_guidance(
        unit_id=1, cycle=10, proba=_proba(0.12), horizons=[30], thresholds=thr
    )
    assert g.risk_tier is RiskTier.HIGH


def test_guidance_serialises_to_dict():
    g = generate_maintenance_guidance(
        unit_id=7, cycle=42, proba=_proba(0.5), horizons=[30], shi=0.4
    )
    d = g.to_dict()
    assert d["unit_id"] == 7
    assert d["risk_tier"] in {"Low", "Medium", "High", "Critical"}
    assert 0.0 <= d["p30"] <= 1.0
    json.dumps(d)  # must be JSON-serialisable


def test_nearest_horizon_used_when_30_missing():
    g = generate_maintenance_guidance(
        unit_id=1,
        cycle=10,
        proba=pd.DataFrame([{"fail_h20": 0.9}]),
        horizons=[10, 20, 40],
    )
    assert g.horizon_used == 20
    assert g.p30 == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Shared artifact loader
# ---------------------------------------------------------------------------


def test_load_artifact_returns_none_for_missing(tmp_path: Path, monkeypatch):
    import stattwin.dashboard.components.artifacts as art

    monkeypatch.setattr(art, "RESULTS_DIR", tmp_path)
    assert art.load_artifact("nope.json") is None


def test_write_and_load_artifact_roundtrip(tmp_path: Path, monkeypatch):
    import stattwin.dashboard.components.artifacts as art

    monkeypatch.setattr(art, "RESULTS_DIR", tmp_path)
    path = art.write_artifact("health_summary.json", {"shi": 0.5})
    assert path.exists()
    data = art.load_artifact("health_summary.json")
    assert data == {"shi": 0.5}


def test_load_artifact_prefers_machine_dir(tmp_path: Path, monkeypatch):
    import stattwin.dashboard.components.artifacts as art

    monkeypatch.setattr(art, "RESULTS_DIR", tmp_path)
    (tmp_path / "global").mkdir()
    (tmp_path / "MACHINE-001").mkdir()
    (tmp_path / "global" / "health_summary.json").write_text(
        json.dumps({"shi": 0.1}), encoding="utf-8"
    )
    (tmp_path / "MACHINE-001" / "health_summary.json").write_text(
        json.dumps({"shi": 0.9}), encoding="utf-8"
    )
    data = art.load_artifact("health_summary.json", "MACHINE-001")
    assert data["shi"] == pytest.approx(0.9)


def test_write_artifact_rejects_unknown_name(tmp_path: Path, monkeypatch):
    import stattwin.dashboard.components.artifacts as art

    monkeypatch.setattr(art, "RESULTS_DIR", tmp_path)
    with pytest.raises(ValueError):
        art.write_artifact("evil.exe", {})


# ---------------------------------------------------------------------------
# Model comparison payload (no fabricated Cox/RSF/LSTM)
# ---------------------------------------------------------------------------


def _load_comparison_mod():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stattwin"
        / "dashboard"
        / "views"
        / "7_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("cmp7", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["cmp7"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_comparison_builds_from_real_e2_e5():
    mod = _load_comparison_mod()
    payload = mod._comparison_from_experiments()
    # Real results are committed under results/
    e2 = (RESULTS_DIR / "e2_model_comparison" / "e2_results.json").exists()
    if not e2:
        pytest.skip("e2 results not present")
    assert payload is not None
    assert payload["models"]
    assert "AUC" in payload["metrics"]
    # Must NOT invent survival-model names from the old demo
    banned = {"Cox PH", "RSF", "LSTM", "Survival SVM"}
    assert banned.isdisjoint(set(payload["models"]))


def test_lower_is_better_set():
    mod = _load_comparison_mod()
    assert "Brier" in mod._LOWER_IS_BETTER
    assert "ECE" in mod._LOWER_IS_BETTER
    assert "AUC" not in mod._LOWER_IS_BETTER


def test_delta_card_direction():
    # pure logic check of colour decision is embedded; ensure callable signature
    import inspect

    from stattwin.dashboard.components.cards import delta_card  # noqa: F401

    sig = inspect.signature(delta_card)
    assert "higher_is_better" in sig.parameters
    assert "provenance" in sig.parameters
