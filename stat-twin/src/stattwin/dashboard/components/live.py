"""Live auto-refresh helpers for the STAT-TWIN dashboard pages.

Every page wraps its data and chart rendering in an
``@st.fragment(run_every="2s")`` section so the UI updates like a real-time
monitoring system.  This module provides the shared building blocks:

* ``current_tick``   - monotonic 2-second tick driving every animation
* ``live_status``    - the LIVE badge plus the "Updated X.Xs ago" ticker
* ``live_scalar``    - bounded tick-driven variation around a baseline value
* ``live_series``    - sensor window that advances with the tick
* ``live_window``    - aligned ``(timestamps, values)`` live sensor window
* ``live_wave``      - synthetic scrolling waveform (demo fallback)
* ``live_jitter``    - bounded per-cell variation for matrices / arrays
* ``rul_countdown``  - RUL value that counts down between ticks
* ``slide_window``   - sliding-window index over artifact series
* ``state_from_shi``  - SHI value mapped to its canonical state label
"""
from __future__ import annotations

import time
import zlib

import numpy as np
import streamlit as st

LIVE_INTERVAL = "2s"
LIVE_INTERVAL_MS = 2000

CARD_BG = "#111827"
BORDER = "#1F2937"
MUTED = "#9CA3AF"
DANGER = "#EF4444"

DEFAULT_FEED = "live stream @ 2s (baseline = artifacts / demo)"


def current_tick(interval_ms: int = LIVE_INTERVAL_MS) -> int:
    """Return a monotonically increasing tick that advances every *interval_ms*."""
    return int(time.time() * 1000) // interval_ms


def stable_seed(name: str) -> int:
    """Return a process-independent 31-bit seed derived from *name*."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def clip01(value: float) -> float:
    """Clip *value* to the inclusive [0, 1] range."""
    return float(min(1.0, max(0.0, value)))


def state_from_shi(shi: float) -> str:
    """Map an SHI value to its canonical state label."""
    if shi < 0.25:
        return "HEALTHY"
    if shi < 0.50:
        return "WATCH"
    if shi < 0.75:
        return "DEGRADING"
    return "CRITICAL"


def live_status(scope: str, *, feed: str = DEFAULT_FEED) -> None:
    """Render the LIVE badge, freshness indicator, and the refresh ticker.

    Call this *inside* the ``@st.fragment(run_every="2s")`` body so the
    "Updated X.Xs ago" text re-renders on every fragment rerun.  *scope*
    keeps the per-section timestamps independent.
    """
    key = f"_stattwin_live_ts::{scope}"
    now = time.time()
    previous = st.session_state.get(key)
    st.session_state[key] = now
    age = 0.0 if previous is None else max(0.0, now - float(previous))
    tick = current_tick()
    freshness = freshness_badge(age)

    st.markdown(
        f"""
<div style="display:flex;align-items:center;justify-content:space-between;gap:14px;
            flex-wrap:wrap;background:{CARD_BG};border:1px solid {BORDER};
            border-radius:8px;padding:7px 14px;margin-bottom:10px;">
    <span style="display:inline-flex;align-items:center;gap:7px;color:{DANGER};
                 font-size:0.72rem;font-weight:800;letter-spacing:1.4px;">
        <span style="width:8px;height:8px;border-radius:50%;background:{DANGER};
                     box-shadow:0 0 8px {DANGER};display:inline-block;
                     animation:stattwin-blink 1.2s ease-in-out infinite;"></span>
        LIVE · AUTO-REFRESH {LIVE_INTERVAL}
    </span>
    <span style="display:flex;align-items:center;gap:12px;">
        {freshness}
        <span style="color:{MUTED};font-family:'JetBrains Mono',monospace;font-size:0.7rem;">
            tick #{tick}
        </span>
    </span>
    <span style="color:{MUTED};font-size:0.68rem;">{feed}</span>
</div>
<style>
@keyframes stattwin-blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.2; }} }}
</style>
""",
        unsafe_allow_html=True,
    )


def slide_window(length: int, tick: int, window: int, step: int = 4) -> int:
    """Return the start index of a slice that advances by *step* each tick."""
    if length <= window or window <= 0:
        return 0
    span = length - window
    return int((tick * step) % (span + 1))


def _clean(values) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float)
    if arr.size:
        arr = arr[np.isfinite(arr)]
    return arr


def _shimmer(n: int, tick: int, scale: float) -> np.ndarray:
    return 0.004 * scale * np.sin(np.arange(n, dtype=float) / 6.0 + tick * 0.6)


def _synthetic(name: str, tick: int, n: int, arr: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(name))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    if arr.size:
        base = float(np.mean(arr))
        scale = float(np.std(arr)) or max(abs(base) * 0.05, 0.05)
    else:
        base = float(rng.uniform(1.0, 5.0))
        scale = 0.12
    noise = rng.normal(0.0, 0.35, n)
    t = np.arange(n, dtype=float)
    return (
        base
        + 0.10 * scale * np.sin(t / 9.0 + tick * 0.4 + phase)
        + 0.08 * scale * np.sin(tick * 0.3 + phase)
        + scale * noise
    )


def live_series(name: str, tick: int, n: int = 100, base_values=None) -> np.ndarray:
    """Return an *n*-point sensor window that advances with *tick*.

    When *base_values* is long enough a sliding window moves across it so the
    chart scrolls without inventing measurements.  Shorter or absent series
    fall back to a bounded synthetic stream seeded by *name*.
    """
    arr = _clean(base_values)
    if arr.size >= n:
        start = slide_window(int(arr.size), tick, n, step=5)
        window = arr[start : start + n].copy()
        return window + _shimmer(n, tick, float(np.std(window) or 1.0))
    return _synthetic(name, tick, n, arr)


def live_window(
    name: str,
    tick: int,
    n: int = 100,
    base_values=None,
    timestamps=None,
) -> tuple[list, np.ndarray]:
    """Return ``(timestamps, values)`` for a live sensor window.

    Timestamps are sliced together with the values whenever they align with
    *base_values*; otherwise a synthetic 0..n-1 index is returned.
    """
    arr = _clean(base_values)
    if arr.size >= n:
        start = slide_window(int(arr.size), tick, n, step=5)
        values = arr[start : start + n].copy()
        values = values + _shimmer(n, tick, float(np.std(values) or 1.0))
        if timestamps is not None and len(timestamps) == arr.size:
            return list(timestamps[start : start + n]), values
        return list(range(n)), values
    values = _synthetic(name, tick, n, arr)
    return list(range(n)), values


def live_wave(
    name: str,
    tick: int,
    n: int = 120,
    *,
    base: float = 0.5,
    scale: float = 0.08,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Return a smooth synthetic waveform that scrolls with *tick*."""
    rng = np.random.default_rng(stable_seed(name))
    t = np.arange(n, dtype=float)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    noise = rng.normal(0.0, scale * 0.35, n)
    wave = (
        float(base)
        + 2.0 * scale * np.sin(t / 25.0 + tick * 0.08 + phase)
        + 0.5 * scale * np.sin(t / 6.0 + tick * 0.45)
        + noise
    )
    if low is not None or high is not None:
        wave = np.clip(wave, low, high)
    return wave


def live_scalar(baseline: float, tick: int, *, amp: float, name: str) -> float:
    """Return a bounded oscillation around *baseline* that changes each tick."""
    rng = np.random.default_rng(stable_seed(name))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    value = (
        float(baseline)
        + amp * np.sin(tick * 0.45 + phase)
        + 0.4 * amp * np.sin(tick * 1.7 + phase * 2.1)
    )
    return float(value)


def live_jitter(data, tick: int, *, amp: float, name: str) -> np.ndarray:
    """Return a copy of *data* with bounded per-cell variation from *tick*."""
    arr = np.array(data, dtype=float, copy=True)
    if arr.size == 0:
        return arr
    rng = np.random.default_rng(stable_seed(name))
    phases = rng.uniform(-np.pi, np.pi, size=arr.shape)
    return arr + amp * np.sin(tick * 0.4 + phases)


def rul_countdown(
    baseline: float,
    tick: int,
    *,
    per_tick: float = 0.15,
    cycle: int = 200,
) -> float:
    """Return an RUL value that counts down every tick and wraps over *cycle*."""
    base = float(baseline)
    if base <= 0:
        return 0.0
    floor = max(1.0, base * 0.05)
    elapsed = (tick % int(cycle)) * float(per_tick)
    return float(max(floor, base - elapsed))


def freshness_badge(updated_seconds_ago: float | None = None) -> str:
    """Return an HTML freshness indicator with pulsing dot.

    Args:
        updated_seconds_ago: Seconds since last data update. None = live.
    """
    if updated_seconds_ago is None or updated_seconds_ago < 5:
        dot_cls = "st-freshness-dot st-freshness-ok"
        label = "Live"
    elif updated_seconds_ago < 30:
        dot_cls = "st-freshness-dot st-freshness-ok"
        label = f"{updated_seconds_ago:.0f}s ago"
    elif updated_seconds_ago < 300:
        dot_cls = "st-freshness-dot st-freshness-stale"
        label = f"{updated_seconds_ago:.0f}s ago (stale)"
    else:
        dot_cls = "st-freshness-dot st-freshness-stale"
        label = f"{updated_seconds_ago / 60:.1f}m ago (stale)"
    return (
        f'<span class="st-freshness"><span class="{dot_cls}"></span>{label}</span>'
    )


def stale_data_warning(message: str = "Data may be outdated") -> None:
    """Render a stale data warning banner."""
    st.markdown(
        f"""
        <div style="
            background:rgba(245,158,11,0.10); border:1px solid #F59E0B;
            border-radius:8px; padding:8px 14px; margin-bottom:10px;
            color:#FCD34D; font-size:0.78rem; font-weight:600;
            letter-spacing:0.3px; display:flex; align-items:center; gap:8px;">
            <span style="font-size:0.9rem;">⏳</span>
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
