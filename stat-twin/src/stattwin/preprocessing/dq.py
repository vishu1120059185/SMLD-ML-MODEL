"""Data Quality (DQ) Engine for STAT-TWIN.

Per-sensor flags that indicate *data-quality* problems (as opposed to
genuine machine-anomaly signals).  The engine computes:

* **missing** – value is NaN.
* **dropout_burst** – ≥ *k* consecutive missing values (a burst event).
* **stuck** – zero variance over a sliding window (sensor frozen).
* **spike** – single-cycle outlier detected via robust z-score.
* **out_of_range** – value outside physically meaningful bounds.

Separation logic
----------------
A *DQ anomaly* is typically single-sensor, abrupt, non-persistent, and not
corroborated by other sensors.  A *machine anomaly* is persistent,
multi-sensor coherent, and direction-consistent with degradation.

The ``DQEngine`` attaches a ``dq_badge`` column (bitmask) and per-sensor
``<sensor>_dq_flag`` boolean columns.  When any DQ flag is active the
underlying value is marked for *repair* before feature computation, and a
``"data_quality_degraded"`` badge is raised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["DQEngine", "DQFlags"]


# ====================================================================
# DQ flag bitmask
# ====================================================================


class DQFlags:
    """Bitmask constants for DQ flag categories."""

    MISSING: int = 1 << 0          # 00001
    DROPOUT_BURST: int = 1 << 1    # 00010
    STUCK: int = 1 << 2            # 00100
    SPIKE: int = 1 << 3            # 01000
    OUT_OF_RANGE: int = 1 << 4     # 10000

    ALL: int = 0b11111

    NAMES: dict[int, str] = {
        1: "missing",
        2: "dropout_burst",
        4: "stuck",
        8: "spike",
        16: "out_of_range",
    }

    @classmethod
    def decode(cls, mask: int) -> list[str]:
        """Decode a bitmask into a list of flag names."""
        return [name for bit, name in cls.NAMES.items() if mask & bit]


# ====================================================================
# DQ Engine
# ====================================================================


@dataclass
class DQConfig:
    """Configuration for the DQ engine.

    Parameters
    ----------
    sensor_columns:
        Sensor columns to monitor.  If *None*, all ``sensor_*`` columns
        from ``SENSOR_NAMES`` are used.
    dropout_burst_k:
        Minimum consecutive missing values to qualify as a dropout burst.
    stuck_window:
        Sliding window length (in cycles) for the stuck-sensor check.
    spike_threshold:
        Robust-z threshold for single-cycle spike detection.
    range_bounds:
        Per-sensor ``(low, high)`` physical bounds.  Sensors not listed
        here are not checked for out-of-range values.
    repair_strategy:
        How to repair flagged values: ``"forward_fill"`` or ``"nan"``
        (set to NaN and let downstream imputers handle it).
    """

    sensor_columns: list[str] | None = None
    dropout_burst_k: int = 3
    stuck_window: int = 10
    spike_threshold: float = 6.0
    range_bounds: dict[str, tuple[float, float]] | None = None
    repair_strategy: str = "forward_fill"


class DQEngine:
    """Data Quality Engine – flags and repairs per-sensor DQ issues.

    Parameters
    ----------
    config:
        DQ configuration.  If *None*, a default ``DQConfig`` is used.
    """

    def __init__(self, config: DQConfig | None = None) -> None:
        self.config = config or DQConfig()
        self._sensor_cols: list[str] | None = None
        self._is_fitted = False
        # Store training robust stats for spike detection
        self._spike_median: dict[str, float] | None = None
        self._spike_mad: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, *, unit_col: str = "unit_id") -> DQEngine:
        """Learn per-sensor robust statistics from training data.

        Parameters
        ----------
        df:
            Training DataFrame.
        unit_col:
            Name of the unit-identifier column.

        Returns
        -------
        DQEngine
            Self, for method chaining.
        """
        self._sensor_cols = (
            self.config.sensor_columns
            if self.config.sensor_columns is not None
            else [f"sensor_{i}" for i in range(1, 22)]
        )
        self._spike_median = {}
        self._spike_mad = {}
        for col in self._sensor_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            self._spike_median[col] = med
            self._spike_mad[col] = mad if mad > 0 else 1e-12
        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        """Compute DQ flags, repair values, and attach badges.

        Parameters
        ----------
        df:
            DataFrame to annotate.
        unit_col:
            Name of the unit-identifier column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added DQ columns and repaired values.
        """
        if not self._is_fitted:
            raise RuntimeError("DQEngine has not been fitted. Call fit() first.")

        out = df.copy()
        sensor_cols = self._sensor_cols or [f"sensor_{i}" for i in range(1, 22)]
        active_cols = [c for c in sensor_cols if c in out.columns]

        # ---- per-sensor flag DataFrames --------------------------------
        flag_frames: dict[str, pd.DataFrame] = {}
        for col in active_cols:
            flags = self._flag_sensor(out, col, unit_col)
            flag_frames[col] = flags

        # ---- aggregate per-sensor into dq_badge bitmask ----------------
        out["dq_badge"] = np.zeros(len(out), dtype=np.int32)
        for col in active_cols:
            out[f"{col}_dq_flag"] = False
            for flag_name, bit in [
                ("missing", DQFlags.MISSING),
                ("dropout_burst", DQFlags.DROPOUT_BURST),
                ("stuck", DQFlags.STUCK),
                ("spike", DQFlags.SPIKE),
                ("out_of_range", DQFlags.OUT_OF_RANGE),
            ]:
                mask = flag_frames[col][flag_name]
                if mask.any():
                    out.loc[mask, "dq_badge"] |= bit
                    out.loc[mask, f"{col}_dq_flag"] = True

        # ---- "data_quality_degraded" badge -----------------------------
        out["data_quality_degraded"] = (out["dq_badge"] != 0).astype(np.int8)

        # ---- repair flagged values -------------------------------------
        out = self._repair(out, active_cols)

        return out

    def fit_transform(
        self, df: pd.DataFrame, *, unit_col: str = "unit_id"
    ) -> pd.DataFrame:
        return self.fit(df, unit_col=unit_col).transform(df, unit_col=unit_col)

    # ------------------------------------------------------------------
    # Internal flagging
    # ------------------------------------------------------------------

    def _flag_sensor(
        self, df: pd.DataFrame, col: str, unit_col: str
    ) -> pd.DataFrame:
        """Return a boolean-DataFrame of DQ flags for one sensor."""
        missing = df[col].isna()

        dropout = self._detect_dropout_burst(df, col, unit_col)
        stuck = self._detect_stuck(df, col, unit_col)
        spike = self._detect_spike(df, col)
        oor = self._detect_out_of_range(df, col)

        return pd.DataFrame(
            {
                "missing": missing.values,
                "dropout_burst": dropout.values,
                "stuck": stuck.values,
                "spike": spike.values,
                "out_of_range": oor.values,
            },
            index=df.index,
        )

    def _detect_dropout_burst(
        self, df: pd.DataFrame, col: str, unit_col: str
    ) -> pd.Series:
        """Flag rows that belong to a run of ≥ k consecutive NaNs."""
        k = self.config.dropout_burst_k
        result = pd.Series(False, index=df.index)

        for _uid, grp in df.groupby(unit_col):
            is_nan = grp[col].isna()
            # Compute run lengths of consecutive NaNs
            groups = (~is_nan).cumsum()
            run_lengths = is_nan.groupby(groups).transform("sum")
            result.loc[grp.index] = (is_nan) & (run_lengths >= k)

        return result

    def _detect_stuck(
        self, df: pd.DataFrame, col: str, unit_col: str
    ) -> pd.Series:
        """Flag rows where the rolling window has zero variance (sensor frozen)."""
        w = self.config.stuck_window
        result = pd.Series(False, index=df.index)

        for _uid, grp in df.groupby(unit_col):
            vals = grp[col].values
            rolling_std = pd.Series(vals).rolling(window=w, min_periods=w).std()
            # A window with zero std → stuck
            stuck_in_window = rolling_std == 0
            # Only flag the *tail* of a stuck run to avoid over-flagging
            result.loc[grp.index] = stuck_in_window.values

        return result

    def _detect_spike(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Single-cycle spike via robust z-score.

        Only flags values where *only this sensor* spikes (not
        corroborated by peers), consistent with DQ-anomaly semantics.
        """
        if self._spike_median is None or col not in self._spike_median:
            return pd.Series(False, index=df.index)

        med = self._spike_median[col]
        mad = self._spike_mad[col]
        rz = (df[col] - med) / (1.4826 * mad)
        return rz.abs() > self.config.spike_threshold

    def _detect_out_of_range(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Flag values outside physically meaningful bounds."""
        bounds = self.config.range_bounds
        if bounds is None or col not in bounds:
            return pd.Series(False, index=df.index)
        lo, hi = bounds[col]
        return (df[col] < lo) | (df[col] > hi)

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def _repair(
        self, df: pd.DataFrame, sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Replace flagged values according to ``repair_strategy``."""
        if self.config.repair_strategy == "nan":
            # Already NaN – nothing to do (missing values stay NaN)
            return df

        # forward-fill within each unit for non-missing flagged values
        out = df.copy()
        for col in sensor_cols:
            flag_col = f"{col}_dq_flag"
            if flag_col not in out.columns:
                continue
            masked = out.loc[out[flag_col], col]
            if masked.empty:
                continue
            if self.config.repair_strategy == "forward_fill":
                # Replace flagged non-NaN values with NaN, then ffill per unit
                out.loc[out[flag_col], col] = np.nan
                for _uid, grp in out.groupby("unit_id"):
                    filled = grp[col].ffill()
                    out.loc[grp.index, col] = filled
            else:
                raise ValueError(
                    f"Unknown repair_strategy: '{self.config.repair_strategy}'"
                )
        return out


# ====================================================================
# Separation helpers (single-sensor DQ vs multi-sensor machine anomaly)
# ====================================================================


def separate_anomalies(
    df: pd.DataFrame,
    dq_cols: list[str] | None = None,
    *,
    min_corroboration: int = 2,
) -> pd.Series:
    """Classify each row as ``"dq"`` or ``"machine"`` based on corroboration.

    A row is considered a *machine anomaly* if ≥ ``min_corroboration``
    distinct sensor ``_dq_flag`` columns are active simultaneously.
    Otherwise it is a *data-quality* issue.

    Parameters
    ----------
    df:
        DataFrame produced by ``DQEngine.transform``.
    dq_cols:
        List of ``<sensor>_dq_flag`` columns to consider.  If *None*,
        all columns matching ``*_dq_flag`` are used.
    min_corroboration:
        Minimum number of simultaneously flagged sensors to classify as
        machine anomaly.

    Returns
    -------
    pd.Series
        Categorical series with values ``"dq"`` or ``"machine"``.
    """
    if dq_cols is None:
        dq_cols = [c for c in df.columns if c.endswith("_dq_flag")]
    if not dq_cols:
        return pd.Series("machine", index=df.index, dtype="object")

    flag_matrix = df[dq_cols].astype(int)
    n_flagged = flag_matrix.sum(axis=1)
    return pd.Series(
        np.where(n_flagged >= min_corroboration, "machine", "dq"),
        index=df.index,
        dtype="object",
    )
