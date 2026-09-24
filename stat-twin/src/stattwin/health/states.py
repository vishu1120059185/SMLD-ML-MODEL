"""Health state classification for STAT-TWIN.

Discretises the continuous SHI into five ordered states:

* HEALTHY (4) – normal operation
* WATCH (3) – mild degradation detected
* DEGRADING (2) – significant degradation
* CRITICAL (1) – severe degradation, near failure
* FAILURE_LIKELY (0) – failure imminent

State transitions are subject to persistence (hysteresis): a new state
must persist for *p* consecutive cycles before the transition is
committed.  This prevents rapid oscillation between adjacent states.

Two threshold variants are supported:

* **Calibrated** – thresholds derived from training-data SHI distribution
  (e.g., 25th, 50th, 75th percentiles of failed-unit SHI).
* **Fixed-grid** – user-specified thresholds ``[80, 60, 40, 20]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "HealthState",
    "StateClassifier",
    "classify_states",
]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# ---------------------------------------------------------------------------
# Health state enum
# ---------------------------------------------------------------------------

class HealthState(IntEnum):
    """Ordered health states (higher = healthier)."""

    FAILURE_LIKELY = 0
    CRITICAL = 1
    DEGRADING = 2
    WATCH = 3
    HEALTHY = 4

    @classmethod
    def from_name(cls, name: str) -> HealthState:
        """Convert a string name to HealthState (case-insensitive)."""
        mapping = {s.name.lower(): s for s in cls}
        # Also handle "failure-likely" -> "failure_likely"
        mapping["failure-likely"] = cls.FAILURE_LIKELY
        key = name.lower().replace("-", "_")
        if key not in mapping:
            raise ValueError(
                f"Unknown state '{name}'; use one of {[s.name for s in cls]}"
            )
        return mapping[key]


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def _calibrate_thresholds(
    shi_series: pd.Series,
    rul_series: pd.Series | None = None,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> list[float]:
    """Derive thresholds from training-data SHI distribution.

    If RUL is provided, thresholds are computed from the SHI values of
    units near failure (RUL < 30).  Otherwise, all SHI values are used.

    Parameters
    ----------
    shi_series:
        SHI values from training data.
    rul_series:
        Corresponding RUL values (optional).
    quantiles:
        Quantiles to compute (must have 3 values for 4 states).

    Returns
    -------
    list[float]
        Thresholds [t1, t2, t3] where:
        - SHI >= t1 => HEALTHY
        - t2 <= SHI < t1 => WATCH
        - t3 <= SHI < t2 => DEGRADING
        - t4 <= SHI < t3 => CRITICAL
        - SHI < t4 => FAILURE_LIKELY
    """
    if rul_series is not None:
        # Use SHI values near failure
        near_failure_mask = rul_series < 30
        if near_failure_mask.sum() > 10:
            shi_vals = shi_series[near_failure_mask].dropna()
        else:
            shi_vals = shi_series.dropna()
    else:
        shi_vals = shi_series.dropna()

    if len(shi_vals) < 4:
        # Fallback to fixed grid
        return [80.0, 60.0, 40.0]

    thresholds = np.quantile(shi_vals.values, quantiles)
    return sorted(thresholds.tolist(), reverse=True)


# ---------------------------------------------------------------------------
# State classifier
# ---------------------------------------------------------------------------

@dataclass
class StateClassifier:
    """Classify SHI values into health states with persistence.

    Parameters
    ----------
    method:
        Threshold method: ``"calibrated"`` or ``"fixed_grid"``.
    fixed_grid:
        Thresholds for fixed_grid method.  Must have 3 values for
        4 states: [healthy_watch, watch_degrading, degrading_critical].
    persistence:
        Number of consecutive cycles a new state must persist before
        the transition is committed (hysteresis).
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.
    """

    method: Literal["calibrated", "fixed_grid"] = "calibrated"
    fixed_grid: list[float] = field(default_factory=lambda: [80.0, 60.0, 40.0, 20.0])
    persistence: int = 3
    unit_col: str = _UNIT_COL
    cycle_col: str = _CYCLE_COL

    # Set after calibration
    _thresholds: list[float] = field(default_factory=lambda: [80.0, 60.0, 40.0], repr=False)
    _is_calibrated: bool = False

    def calibrate(
        self,
        shi_df: pd.DataFrame,
        rul_df: pd.DataFrame | None = None,
        shi_col: str = "shi",
    ) -> None:
        """Calibrate thresholds from training data.

        Parameters
        ----------
        shi_df:
            DataFrame with SHI values (must contain unit_col, cycle_col, shi_col).
        rul_df:
            Optional DataFrame with RUL values for calibration.
        shi_col:
            Column name for SHI values.
        """
        if self.method == "fixed_grid":
            self._thresholds = sorted(self.fixed_grid[:3], reverse=True)
            self._is_calibrated = True
            return

        # Calibrated method
        if rul_df is not None:
            # Merge SHI and RUL
            merged = shi_df.merge(
                rul_df[[self.unit_col, self.cycle_col, "RUL"]],
                on=[self.unit_col, self.cycle_col],
                how="left",
            )
            rul_series = merged["RUL"]
        else:
            rul_series = None

        self._thresholds = _calibrate_thresholds(shi_df[shi_col], rul_series)
        self._is_calibrated = True

    @property
    def thresholds(self) -> list[float]:
        """Return calibrated thresholds [t1, t2, t3]."""
        if not self._is_calibrated:
            raise RuntimeError("Classifier not calibrated. Call calibrate() first.")
        return self._thresholds

    def _shi_to_raw_state(self, shi: float) -> HealthState:
        """Map SHI value to raw state without persistence."""
        t = self._thresholds
        if len(t) < 3:
            t = [80.0, 60.0, 40.0]

        if shi >= t[0]:
            return HealthState.HEALTHY
        elif shi >= t[1]:
            return HealthState.WATCH
        elif shi >= t[2]:
            return HealthState.DEGRADING
        elif shi >= 20.0:  # Lower bound for CRITICAL
            return HealthState.CRITICAL
        else:
            return HealthState.FAILURE_LIKELY

    def classify(
        self,
        shi_df: pd.DataFrame,
        shi_col: str = "shi",
    ) -> pd.DataFrame:
        """Classify SHI values into health states with persistence.

        Parameters
        ----------
        shi_df:
            DataFrame with SHI values (must contain unit_col, cycle_col, shi_col).
        shi_col:
            Column name for SHI values.

        Returns
        -------
        pd.DataFrame
            Copy with ``health_state`` and ``health_state_name`` columns added.
        """
        if not self._is_calibrated:
            self.calibrate(shi_df, shi_col=shi_col)

        out = shi_df.copy()
        out["health_state"] = np.nan
        out["health_state_name"] = ""

        grp = out.sort_values([self.unit_col, self.cycle_col]).groupby(self.unit_col)

        def _classify_unit(group: pd.DataFrame) -> pd.DataFrame:
            """Apply persistence-based state classification for one unit."""
            n = len(group)
            shi_vals = group[shi_col].values
            states = np.empty(n, dtype=np.int8)
            state_names = [""] * n

            # First cycle: initial state
            current_raw = self._shi_to_raw_state(shi_vals[0])
            current_state = current_raw
            states[0] = current_state
            state_names[0] = current_state.name

            # Persistence tracking
            pending_state: HealthState | None = None
            pending_count = 0

            for i in range(1, n):
                raw_state = self._shi_to_raw_state(shi_vals[i])

                if raw_state == current_state:
                    # Same state: reset pending
                    pending_state = None
                    pending_count = 0
                elif raw_state == pending_state:
                    # Same pending state: increment counter
                    pending_count += 1
                    if pending_count >= self.persistence:
                        # Commit transition
                        current_state = raw_state
                        pending_state = None
                        pending_count = 0
                else:
                    # New different state: start tracking
                    pending_state = raw_state
                    pending_count = 1

                states[i] = current_state
                state_names[i] = current_state.name

            group = group.copy()
            group["health_state"] = states
            group["health_state_name"] = state_names
            return group

        result = grp.apply(_classify_unit, include_groups=False)
        # Restore original index if needed
        if isinstance(result.index, pd.MultiIndex):
            result = result.reset_index(level=0, drop=True)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_states(
    shi_df: pd.DataFrame,
    method: Literal["calibrated", "fixed_grid"] = "calibrated",
    fixed_grid: list[float] | None = None,
    persistence: int = 3,
    rul_df: pd.DataFrame | None = None,
    shi_col: str = "shi",
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> pd.DataFrame:
    """Classify SHI values into health states with persistence.

    This is a convenience wrapper around :class:`StateClassifier`.

    Parameters
    ----------
    shi_df:
        DataFrame with SHI values (must contain unit_col, cycle_col, shi_col).
    method:
        Threshold method: ``"calibrated"`` or ``"fixed_grid"``.
    fixed_grid:
        Thresholds for fixed_grid method.
    persistence:
        Number of consecutive cycles a new state must persist.
    rul_df:
        Optional DataFrame with RUL values for calibration.
    shi_col:
        Column name for SHI values.
    unit_col:
        Unit identifier column.
    cycle_col:
        Cycle identifier column.

    Returns
    -------
    pd.DataFrame
        Copy with ``health_state`` and ``health_state_name`` columns added.
    """
    if fixed_grid is None:
        fixed_grid = [80.0, 60.0, 40.0, 20.0]

    classifier = StateClassifier(
        method=method,
        fixed_grid=fixed_grid,
        persistence=persistence,
        unit_col=unit_col,
        cycle_col=cycle_col,
    )
    return classifier.classify(shi_df, shi_col=shi_col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def state_transition_matrix(
    states_df: pd.DataFrame,
    unit_col: str = _UNIT_COL,
    state_col: str = "health_state",
) -> pd.DataFrame:
    """Compute state transition counts.

    Parameters
    ----------
    states_df:
        DataFrame with state classifications.
    unit_col:
        Unit identifier column.
    state_col:
        Column with integer state values.

    Returns
    -------
    pd.DataFrame
        Transition count matrix (rows=from, cols=to).
    """
    all_states = [s.value for s in HealthState]
    n_states = len(all_states)
    trans_matrix = np.zeros((n_states, n_states), dtype=np.int64)

    for _, group in states_df.groupby(unit_col):
        states = group[state_col].values
        for i in range(len(states) - 1):
            from_idx = all_states.index(states[i])
            to_idx = all_states.index(states[i + 1])
            trans_matrix[from_idx, to_idx] += 1

    state_names = [s.name for s in HealthState]
    return pd.DataFrame(trans_matrix, index=state_names, columns=state_names)


def state_duration_stats(
    states_df: pd.DataFrame,
    unit_col: str = _UNIT_COL,
    state_col: str = "health_state",
) -> pd.DataFrame:
    """Compute duration statistics for each state.

    Parameters
    ----------
    states_df:
        DataFrame with state classifications.
    unit_col:
        Unit identifier column.
    state_col:
        Column with integer state values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: state, mean_duration, std_duration,
        min_duration, max_duration, count.
    """
    durations = []

    for _, group in states_df.groupby(unit_col):
        states = group[state_col].values
        if len(states) == 0:
            continue

        current_state = states[0]
        current_duration = 1
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                durations.append({"state": current_state, "duration": current_duration})
                current_state = states[i]
                current_duration = 1
        durations.append({"state": current_state, "duration": current_duration})

    if not durations:
        return pd.DataFrame(columns=["state", "mean_duration", "std_duration",
                                      "min_duration", "max_duration", "count"])

    dur_df = pd.DataFrame(durations)
    stats = dur_df.groupby("state")["duration"].agg(["mean", "std", "min", "max", "count"])
    stats.columns = ["mean_duration", "std_duration", "min_duration", "max_duration", "count"]
    return stats.reset_index()
