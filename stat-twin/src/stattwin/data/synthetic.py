"""Synthetic telemetry fixture generator for tests.

Creates small, self-contained ``DataFrame`` objects that mimic C-MAPSS
structure (monotone cycles, monotone drift, optional noise, optional regime
switches) for unit tests and notebook experiments.

These generators are **not** used in production; they exist solely to make
tests deterministic, fast, and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stattwin.data.schema import (
    COLUMN_NAMES,
    FAILURE_HORIZONS,
    label_col_for,
)

__all__ = ["make_synthetic_unit", "make_synthetic_dataset"]


def make_synthetic_unit(
    unit_id: int,
    n_cycles: int = 200,
    *,
    n_sensors: int = 5,
    drift_slope: float = 0.01,
    noise_std: float = 0.05,
    regime_length: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a single synthetic engine run-to-failure trajectory.

    Parameters
    ----------
    unit_id:
        Unique identifier for the synthetic unit.
    n_cycles:
        Total number of operating cycles (must be ≥ 2).
    n_sensors:
        Number of sensor columns to generate (max 21).
    drift_slope:
        Linear drift coefficient added to each sensor signal per cycle.
    noise_std:
        Standard deviation of i.i.d. Gaussian measurement noise.
    regime_length:
        If set, the unit switches to a second drift regime every
        ``regime_length`` cycles (simulating operating-condition changes).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``COLUMN_NAMES + ["RUL"]`` plus ``fail_h{h}`` columns.
        Contains exactly *n_cycles* rows with cycles ``1 … n_cycles``.
    """
    if n_cycles < 2:
        raise ValueError("n_cycles must be ≥ 2.")
    n_sensors = min(max(n_sensors, 1), 21)

    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)

    # --- operational settings (constant by default) -----------------------
    op_data = {
        "op_setting_1": np.full(n_cycles, 0.5),
        "op_setting_2": np.full(n_cycles, 0.3),
        "op_setting_3": np.full(n_cycles, 100.0),
    }

    # --- sensor signals with monotone drift ------------------------------
    sensor_data: dict[str, np.ndarray] = {}
    for i in range(1, n_sensors + 1):
        key = f"sensor_{i}"
        base = 10.0 + rng.standard_normal() * 0.1  # per-sensor offset
        drift = drift_slope * cycles

        # optional regime switch
        if regime_length is not None and regime_length > 0:
            regime2_start = regime_length
            drift = np.copy(drift)
            drift[regime2_start:] += drift_slope * 0.5 * (
                cycles[regime2_start:] - regime2_start
            )

        sensor_data[key] = base + drift + rng.normal(0, noise_std, n_cycles)

    # fill remaining sensor slots with zeros (if n_sensors < 21)
    for i in range(n_sensors + 1, 22):
        sensor_data[f"sensor_{i}"] = np.zeros(n_cycles)

    # --- assemble DataFrame -----------------------------------------------
    data = {
        "unit_id": np.full(n_cycles, unit_id, dtype=int),
        "cycle": cycles.astype(int),
        **op_data,
        **sensor_data,
    }
    df = pd.DataFrame(data, columns=COLUMN_NAMES)

    # --- RUL and labels ---------------------------------------------------
    max_cycle = n_cycles
    df["RUL"] = (max_cycle - df["cycle"]).astype(int)
    for h in FAILURE_HORIZONS:
        df[label_col_for(h)] = (df["RUL"] <= h).astype(int)

    return df


def make_synthetic_dataset(
    n_units: int = 10,
    *,
    n_cycles_range: tuple[int, int] = (120, 250),
    n_sensors: int = 5,
    drift_slope: float = 0.01,
    noise_std: float = 0.05,
    regime_length: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a multi-unit synthetic dataset.

    Parameters
    ----------
    n_units:
        Number of independent engine units to simulate.
    n_cycles_range:
        ``(min_cycles, max_cycles)`` – each unit gets a random cycle count
        drawn uniformly from this range.
    n_sensors:
        Number of active sensor columns (max 21).
    drift_slope:
        Linear drift coefficient per sensor.
    noise_std:
        Measurement noise standard deviation.
    regime_length:
        If set, triggers regime switches in each unit at the given cycle count.
    seed:
        Master random seed.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all units, sorted by ``(unit_id, cycle)``.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for uid in range(1, n_units + 1):
        n_cyc = int(rng.integers(n_cycles_range[0], n_cycles_range[1] + 1))
        # use a child seed so units are independent
        child_seed = int(rng.integers(0, 2**31))
        unit_df = make_synthetic_unit(
            uid,
            n_cycles=n_cyc,
            n_sensors=n_sensors,
            drift_slope=drift_slope,
            noise_std=noise_std,
            regime_length=regime_length,
            seed=child_seed,
        )
        frames.append(unit_df)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)
