"""Dashboard tests: live helpers and fragment rendering for all 7 pages."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1 import local_script_runner as lsr

from stattwin.dashboard.components.live import (
    clip01,
    current_tick,
    live_jitter,
    live_scalar,
    live_series,
    live_status,
    live_wave,
    live_window,
    rul_countdown,
    slide_window,
    stable_seed,
    state_from_shi,
)

APP = Path(__file__).resolve().parents[1] / "src" / "stattwin" / "dashboard" / "app.py"

PAGES = [
    "⚙️  Overview",
    "📡  Sensor Monitoring",
    "🏥  Statistical Health",
    "🔮  Failure Forecast",
    "🔍  Explainability",
    "🔧  What-If Simulator",
    "📊  Model Comparison",
]


def test_current_tick_advances_concept():
    tick = current_tick()
    assert isinstance(tick, int)
    assert tick == current_tick(2000)


def test_stable_seed_is_deterministic():
    assert stable_seed("VIBRATION_X") == stable_seed("VIBRATION_X")
    assert stable_seed("VIBRATION_X") != stable_seed("TEMPERATURE")
    assert 0 <= stable_seed("x") < 2**31


def test_state_from_shi_thresholds():
    assert state_from_shi(0.10) == "HEALTHY"
    assert state_from_shi(0.30) == "WATCH"
    assert state_from_shi(0.60) == "DEGRADING"
    assert state_from_shi(0.90) == "CRITICAL"


def test_clip01():
    assert clip01(-0.5) == 0.0
    assert clip01(0.42) == 0.42
    assert clip01(1.5) == 1.0


def test_slide_window_wraps_within_bounds():
    for tick in range(0, 50):
        start = slide_window(500, tick, 150, step=5)
        assert 0 <= start <= 350
    assert slide_window(50, 3, 150) == 0


def test_live_scalar_is_bounded_and_changes():
    base = 0.3
    values = [live_scalar(base, tick, amp=0.05, name="shi") for tick in range(20)]
    assert all(abs(v - base) <= 0.06 for v in values)
    assert len(set(round(v, 6) for v in values)) > 1


def test_live_series_shape_and_slide():
    rng = np.random.default_rng(0)
    base = rng.normal(2.0, 0.3, 400)
    a = live_series("s", 1, n=100, base_values=base)
    b = live_series("s", 2, n=100, base_values=base)
    assert a.shape == (100,)
    assert b.shape == (100,)
    assert not np.allclose(a, b)


def test_live_series_without_base_is_synthetic():
    vals = live_series("no-data", 5, n=64)
    assert vals.shape == (64,)
    assert np.all(np.isfinite(vals))


def test_live_window_keeps_timestamp_alignment():
    values = list(np.arange(300, dtype=float))
    timestamps = [f"c{v:.0f}" for v in values]
    ts, vals = live_window("aligned", 3, n=100, base_values=values, timestamps=timestamps)
    assert len(ts) == 100
    assert len(vals) == 100
    assert ts[0].startswith("c")


def test_live_wave_bounds():
    wave = live_wave("w", 4, n=80, base=0.5, scale=0.1, low=0.0, high=1.0)
    assert wave.shape == (80,)
    assert wave.min() >= 0.0
    assert wave.max() <= 1.0


def test_live_jitter_preserves_shape_and_moves():
    z = np.zeros((4, 10))
    a = live_jitter(z, 1, amp=0.2, name="zmap")
    b = live_jitter(z, 2, amp=0.2, name="zmap")
    assert a.shape == (4, 10)
    assert not np.allclose(a, b)
    assert np.all(np.abs(a) <= 0.21)


def test_rul_countdown_decreases_then_wraps():
    values = [rul_countdown(45.0, tick, per_tick=0.15, cycle=200) for tick in range(50)]
    assert values[0] > values[10]
    assert values[-1] >= 1.0
    assert rul_countdown(0.0, 3) == 0.0


def test_live_status_renders_in_session_state():
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\n"
        "from stattwin.dashboard.components.live import live_status\n"
        "live_status('unit-test')\n"
    )
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert "_stattwin_live_ts::unit-test" in at.session_state


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_and_fragments_rerun(page: str):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["nav_page"] = page
    at.run()
    assert not at.exception, f"full run failed for {page}: {at.exception}"

    storage = getattr(at, "_fragment_storage", None)
    if storage is None:
        pytest.skip("AppTest fragment storage API unavailable")
    fragment_ids = list(storage._fragments.keys())
    assert fragment_ids, f"{page} registered no live fragment"

    original = lsr.RerunData
    for fragment_id in fragment_ids:

        def _patched(*args, fragment_id=fragment_id, **kwargs):
            kwargs["fragment_id"] = fragment_id
            return original(*args, **kwargs)

        lsr.RerunData = _patched
        try:
            at.run()
        finally:
            lsr.RerunData = original
        assert not at.exception, f"fragment rerun failed for {page}: {at.exception}"


def test_whatif_registers_two_fragments():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["nav_page"] = "🔧  What-If Simulator"
    at.run()
    assert not at.exception
    storage = getattr(at, "_fragment_storage", None)
    if storage is None:
        pytest.skip("AppTest fragment storage API unavailable")
    assert len(storage._fragments) == 2


def test_sensor_controls_drive_live_charts():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["nav_page"] = "📡  Sensor Monitoring"
    at.run()
    assert not at.exception
    at.multiselect[0].set_value(["VIBRATION_X", "TEMPERATURE"])
    at.run()
    assert not at.exception
    assert len(at.slider) >= 2


def test_whatif_slider_triggers_simulation():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["nav_page"] = "🔧  What-If Simulator"
    at.run()
    assert not at.exception
    slider = at.slider[0]
    slider.set_value(float(slider.max) * 0.4)
    at.run()
    assert not at.exception


def test_comparison_tabs_present():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["nav_page"] = "📊  Model Comparison"
    at.run()
    assert not at.exception
    labels = [tab.label for tab in at.tabs]
    assert len(labels) == 6
    assert any("Metrics" in label for label in labels)
