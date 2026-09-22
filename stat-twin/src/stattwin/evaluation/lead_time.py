"""Lead-time analysis for predictive maintenance warnings.

A **warning** is triggered when the cumulative failure probability exceeds
a configurable threshold for a sustained number of consecutive cycles.

This module provides:

* Warning-rule evaluation with configurable warning horizon *H_w*,
  threshold *tau*, and persistence *p*.
* Lead-time statistics (time between first valid warning and true failure).
* False early warning and false alarm rate computation.
* Budget-matched threshold tuning: *tau* is selected on a validation set
  so that the false alarm rate respects a given ``far_budget``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "LeadTimeReport",
    "lead_time_analysis",
    "tune_tau_for_budget",
]


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class LeadTimeReport:
    """Results of a lead-time analysis on a single dataset.

    Attributes
    ----------
    warning_horizon:
        The cycle-ahead window *H_w* used for the warning rule.
    tau:
        Probability threshold.
    persistence:
        Consecutive cycles above *tau* required before issuing a warning.
    n_units:
        Number of distinct units (engines) in the evaluation set.
    n_failed:
        Number of units that actually failed.
    n_warned:
        Number of units that received a warning before failure.
    n_false_early:
        Units warned but then recovered (no failure within *H_w* of
        the warning).
    n_false_alarm:
        Units warned but never failed during the entire run.
    far:
        False alarm rate = n_false_alarm / n_units.
    lead_times:
        List of lead times (in cycles) for correctly warned units.
    mean_lead_time:
        Mean lead time across warned-and-failed units.
    median_lead_time:
        Median lead time.
    min_lead_time:
        Minimum lead time.
    max_lead_time:
        Maximum lead time.
    """

    warning_horizon: int
    tau: float
    persistence: int
    n_units: int = 0
    n_failed: int = 0
    n_warned: int = 0
    n_false_early: int = 0
    n_false_alarm: int = 0
    far: float = np.nan
    lead_times: List[float] = field(default_factory=list)
    mean_lead_time: float = np.nan
    median_lead_time: float = np.nan
    min_lead_time: float = np.nan
    max_lead_time: float = np.nan


# ---------------------------------------------------------------------------
# Warning detection
# ---------------------------------------------------------------------------

def _detect_warnings_for_unit(
    proba_series: np.ndarray,
    failure_cycle: Optional[int],
    total_cycles: int,
    tau: float,
    persistence: int,
    warning_horizon: int,
) -> Tuple[Optional[int], bool, bool]:
    """Detect warnings for a single unit's probability time-series.

    Parameters
    ----------
    proba_series:
        Per-cycle failure probabilities (length = total_cycles).
    failure_cycle:
        The absolute cycle at which failure occurs (or *None* if the
        unit does not fail).
    total_cycles:
        Total number of cycles for this unit.
    tau:
        Probability threshold.
    persistence:
        Consecutive cycles above *tau* before a warning is triggered.
    warning_horizon:
        Cycles-ahead window *H_w*.

    Returns
    -------
    warning_cycle:
        The cycle at which the warning is first valid, or *None*.
    is_false_early:
        True if warning was issued but no failure followed within *H_w*
        cycles.
    is_false_alarm:
        True if warning was issued but the unit never failed.
    """
    above_threshold = proba_series >= tau

    # Find runs of consecutive True values
    consecutive = 0
    first_above: Optional[int] = None
    warning_cycle: Optional[int] = None

    for i in range(len(above_threshold)):
        if above_threshold[i]:
            if first_above is None:
                first_above = i
            consecutive += 1
            if consecutive >= persistence and warning_cycle is None:
                warning_cycle = i - persistence + 1  # start of the streak
        else:
            consecutive = 0
            first_above = None

    if warning_cycle is None:
        return None, False, False

    # Classify the warning
    is_false_alarm = failure_cycle is None

    is_false_early = False
    if failure_cycle is not None:
        # Warning is false-early if failure occurs more than H_w cycles
        # AFTER the warning cycle.
        if failure_cycle > warning_cycle + warning_horizon:
            is_false_early = True

    return warning_cycle, is_false_early, is_false_alarm


# ---------------------------------------------------------------------------
# Lead-time analysis
# ---------------------------------------------------------------------------

def lead_time_analysis(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    proba_col: str = "fail_prob",
    failure_cycle_col: str = "failure_cycle",
    warning_horizon: int = 20,
    tau: float = 0.5,
    persistence: int = 3,
) -> LeadTimeReport:
    """Run a full lead-time analysis on a dataset of per-cycle predictions.

    Parameters
    ----------
    df:
        DataFrame with columns for unit ID, cycle, predicted probability,
        and (optionally) the true failure cycle for each unit.
    unit_col:
        Column name for the unit identifier.
    cycle_col:
        Column name for the cycle number.
    proba_col:
        Column name for the predicted failure probability.
    failure_cycle_col:
        Column name for the true failure cycle.  Units that do not fail
        should have *NaN* in this column.
    warning_horizon:
        Cycles-ahead window *H_w* for the warning rule.
    tau:
        Probability threshold.
    persistence:
        Consecutive cycles above *tau* required before issuing a warning.

    Returns
    -------
    LeadTimeReport
    """
    units = df[unit_col].unique()
    n_units = len(units)

    failed_units: set = set()
    warned_units: set = set()
    false_early_units: set = set()
    false_alarm_units: set = set()
    lead_times_list: List[float] = []

    # Build per-unit failure cycle lookup
    failure_map: Dict = {}
    for uid, grp in df.groupby(unit_col):
        fc_vals = grp[failure_cycle_col].dropna()
        if len(fc_vals) > 0:
            failure_map[uid] = int(fc_vals.iloc[0])
            failed_units.add(uid)

    for uid, grp in df.groupby(unit_col):
        grp_sorted = grp.sort_values(cycle_col)
        proba_arr = grp_sorted[proba_col].values.astype(float)
        total_cycles = len(grp_sorted)

        failure_cycle = failure_map.get(uid)

        warning_cycle, is_false_early, is_false_alarm = _detect_warnings_for_unit(
            proba_arr, failure_cycle, total_cycles, tau, persistence, warning_horizon
        )

        if warning_cycle is not None:
            warned_units.add(uid)
            if is_false_early:
                false_early_units.add(uid)
            if is_false_alarm:
                false_alarm_units.add(uid)

            # Compute lead time: cycles between warning and failure
            if failure_cycle is not None:
                first_valid_warning_cycle = int(grp_sorted[cycle_col].iloc[warning_cycle])
                lead = failure_cycle - first_valid_warning_cycle
                if lead >= 0:
                    lead_times_list.append(float(lead))

    n_warned = len(warned_units)
    n_false_early = len(false_early_units & warned_units)
    n_false_alarm = len(false_alarm_units)
    far = n_false_alarm / n_units if n_units > 0 else np.nan

    return LeadTimeReport(
        warning_horizon=warning_horizon,
        tau=tau,
        persistence=persistence,
        n_units=n_units,
        n_failed=len(failed_units),
        n_warned=n_warned,
        n_false_early=n_false_early,
        n_false_alarm=n_false_alarm,
        far=float(far) if not np.isnan(far) else np.nan,
        lead_times=lead_times_list,
        mean_lead_time=float(np.mean(lead_times_list)) if lead_times_list else np.nan,
        median_lead_time=float(np.median(lead_times_list)) if lead_times_list else np.nan,
        min_lead_time=float(np.min(lead_times_list)) if lead_times_list else np.nan,
        max_lead_time=float(np.max(lead_times_list)) if lead_times_list else np.nan,
    )


# ---------------------------------------------------------------------------
# Budget-matched threshold tuning
# ---------------------------------------------------------------------------

def tune_tau_for_budget(
    df_val: pd.DataFrame,
    far_budget: float = 0.05,
    tau_grid: Optional[np.ndarray] = None,
    warning_horizon: int = 20,
    persistence: int = 3,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    proba_col: str = "fail_prob",
    failure_cycle_col: str = "failure_cycle",
) -> Tuple[float, LeadTimeReport]:
    """Tune the warning threshold *tau* on a validation set to meet a FAR budget.

    Performs a grid search over candidate *tau* values and returns the
    smallest *tau* (i.e. most sensitive warning) that keeps the false alarm
    rate at or below ``far_budget``.

    Parameters
    ----------
    df_val:
        Validation dataset.
    far_budget:
        Maximum acceptable false alarm rate.
    tau_grid:
        Candidate threshold values.  Defaults to ``np.linspace(0.05, 0.95, 19)``.
    warning_horizon, persistence, unit_col, cycle_col, proba_col, failure_cycle_col:
        Forwarded to ``lead_time_analysis``.

    Returns
    -------
    (best_tau, report)
        The selected threshold and the corresponding lead-time report.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.05, 0.95, 19)

    best_tau = float(tau_grid[0])
    best_report = lead_time_analysis(
        df_val, unit_col, cycle_col, proba_col, failure_cycle_col,
        warning_horizon, best_tau, persistence,
    )

    for tau_candidate in tau_grid:
        report = lead_time_analysis(
            df_val, unit_col, cycle_col, proba_col, failure_cycle_col,
            warning_horizon, float(tau_candidate), persistence,
        )
        far = report.far if not np.isnan(report.far) else 1.0
        if far <= far_budget:
            best_tau = float(tau_candidate)
            best_report = report
            break  # first (smallest) tau that satisfies the budget

    return best_tau, best_report
