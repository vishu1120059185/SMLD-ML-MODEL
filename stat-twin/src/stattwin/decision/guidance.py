"""Maintenance decision guidance for STAT-TWIN.

This module translates model predictions into actionable risk tiers and
maintenance recommendations.  It is explicitly **non-prescriptive**: all
outputs are decision-support information that must be reviewed by
qualified engineers.

Risk tiers
----------
* **Low** – No action required.  Continue routine monitoring.
* **Medium** – Schedule inspection at next planned maintenance window.
* **High** – Prioritise inspection; consider reducing operating load.
* **Critical** – Immediate inspection recommended; plan for potential
  intervention.

Tier assignment is based on:
1. ``P(+30)`` – the probability of failure within 30 cycles.
2. Health state from the SHI classifier.
3. Data-quality status (degraded DQ lowers displayed confidence).

Rules are configurable and should be calibrated on validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

__all__ = [
    "MaintenanceGuidance",
    "RiskTier",
    "generate_maintenance_guidance",
]


# ---------------------------------------------------------------------------
# Risk tier enum
# ---------------------------------------------------------------------------

class RiskTier(StrEnum):
    """Ordered risk tiers (lower = safer)."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

    @property
    def severity_rank(self) -> int:
        """Numeric rank: Low=0, Medium=1, High=2, Critical=3."""
        return list(RiskTier).index(self)


# ---------------------------------------------------------------------------
# Default configurable thresholds
# ---------------------------------------------------------------------------

@dataclass
class RiskThresholds:
    """Configurable thresholds for risk-tier assignment.

    Thresholds are expressed in terms of ``P(+30)`` (probability of
    failure within 30 cycles) and are meant to be calibrated on
    validation data.

    Attributes
    ----------
    p30_low:
        Above this threshold, risk becomes Medium.
    p30_medium:
        Above this threshold, risk becomes High.
    p30_critical:
        Above this threshold, risk becomes Critical.
    state_override_critical:
        If the health state is in this set, force at least Critical tier.
    state_override_high:
        If the health state is in this set, force at least High tier.
    """

    p30_low: float = 0.10
    p30_medium: float = 0.30
    p30_critical: float = 0.70
    state_override_critical: set[str] = field(
        default_factory=lambda: {"FAILURE_LIKELY", "CRITICAL"}
    )
    state_override_high: set[str] = field(
        default_factory=lambda: {"DEGRADING"}
    )


# ---------------------------------------------------------------------------
# Guidance result container
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceGuidance:
    """Maintenance decision support output.

    Attributes
    ----------
    unit_id:
        Unit identifier.
    cycle:
        Cycle number.
    risk_tier:
        Assigned risk tier.
    p30:
        Probability of failure within 30 cycles.
    p30_confidence_interval:
        ``(lower, upper)`` interval for P(+30) if available.
    health_state:
        Current health state name.
    shi:
        Statistical Health Index value.
    evidence_cards:
        List of evidence card dictionaries (top contributing sensors).
    dq_status:
        Data-quality status (``"OK"``, ``"DEGRADED"``, ``"POOR"``).
    confidence_level:
        Displayed confidence level, adjusted for DQ.
    recommendation:
        Human-readable recommendation string.
    rule_triggered:
        Description of which rule produced this tier.
    horizon_used:
        The failure horizon used for the primary risk assessment.
    """

    unit_id: Any
    cycle: int
    risk_tier: RiskTier
    p30: float
    p30_confidence_interval: tuple[float, float] | None = None
    health_state: str = ""
    shi: float = float("nan")
    evidence_cards: list[dict[str, Any]] = field(default_factory=list)
    dq_status: str = "OK"
    confidence_level: str = "Normal"
    recommendation: str = ""
    rule_triggered: str = ""
    horizon_used: int = 30

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            "risk_tier": self.risk_tier.value,
            "p30": self.p30,
            "p30_confidence_interval": self.p30_confidence_interval,
            "health_state": self.health_state,
            "shi": self.shi,
            "evidence_cards": self.evidence_cards,
            "dq_status": self.dq_status,
            "confidence_level": self.confidence_level,
            "recommendation": self.recommendation,
            "rule_triggered": self.rule_triggered,
            "horizon_used": self.horizon_used,
        }

    def __repr__(self) -> str:
        return (
            f"MaintenanceGuidance(unit={self.unit_id}, cycle={self.cycle}, "
            f"tier={self.risk_tier.value}, p30={self.p30:.3f})"
        )


# ---------------------------------------------------------------------------
# DQ-adjusted confidence
# ---------------------------------------------------------------------------

def _adjust_confidence(
    base_tier: RiskTier,
    dq_status: str,
) -> tuple[RiskTier, str]:
    """Lower displayed confidence if DQ is degraded.

    Returns ``(adjusted_tier, confidence_label)``.

    If DQ is degraded, we keep the same tier but flag reduced confidence.
    We never *upgrade* a tier due to poor DQ – only flag it.
    """
    if dq_status == "OK":
        return base_tier, "Normal"
    elif dq_status == "DEGRADED":
        return base_tier, "Reduced – data quality degraded"
    else:  # POOR
        return base_tier, "Low – data quality poor; results may be unreliable"


# ---------------------------------------------------------------------------
# Recommendation text
# ---------------------------------------------------------------------------

def _recommendation_for_tier(tier: RiskTier) -> str:
    """Return a standard recommendation string for each tier."""
    recs = {
        RiskTier.LOW: (
            "No immediate action required. Continue routine monitoring "
            "and scheduled maintenance."
        ),
        RiskTier.MEDIUM: (
            "Schedule inspection at the next planned maintenance window. "
            "Monitor trend closely for acceleration."
        ),
        RiskTier.HIGH: (
            "Prioritise inspection at the earliest feasible opportunity. "
            "Consider reducing operational load if practical. "
            "Increase monitoring frequency."
        ),
        RiskTier.CRITICAL: (
            "Immediate inspection recommended. Plan for potential "
            "intervention. Assess operational risk and consider "
            "proactive shutdown if safety margins are thin."
        ),
    }
    return recs.get(tier, "Review required.")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def generate_maintenance_guidance(
    unit_id: Any,
    cycle: int,
    proba: pd.DataFrame,
    horizons: list[int] | None = None,
    health_state: str = "",
    shi: float = float("nan"),
    evidence_cards: list[dict[str, Any]] | None = None,
    dq_status: str = "OK",
    thresholds: RiskThresholds | None = None,
    p30_ci: tuple[float, float] | None = None,
) -> MaintenanceGuidance:
    """Generate maintenance decision guidance from model predictions.

    Parameters
    ----------
    unit_id:
        Unit identifier.
    cycle:
        Current cycle number.
    proba:
        DataFrame with ``fail_h{h}`` columns from the model.
    horizons:
        Available failure horizons.  If ``None``, derived from ``proba`` columns.
    health_state:
        Current health state name (e.g. ``"HEALTHY"``, ``"DEGRADING"``).
    shi:
        Current SHI value.
    evidence_cards:
        List of evidence card dicts from the explainability module.
    dq_status:
        Data-quality status: ``"OK"``, ``"DEGRADED"``, or ``"POOR"``.
    thresholds:
        Configurable risk thresholds.  Uses defaults if ``None``.
    p30_ci:
        Optional confidence interval ``(lower, upper)`` for P(+30).

    Returns
    -------
    MaintenanceGuidance
        Decision support output with tier, recommendation, and evidence.
    """
    if thresholds is None:
        thresholds = RiskThresholds()

    if evidence_cards is None:
        evidence_cards = []

    # Determine available horizons
    if horizons is None:
        horizons = []
        for col in proba.columns:
            if col.startswith("fail_h"):
                try:
                    h = int(col.split("_h")[1])
                    horizons.append(h)
                except (IndexError, ValueError):
                    pass
        horizons = sorted(horizons)

    # Extract P(+30) or nearest available horizon
    horizon_used = 30
    if 30 in horizons:
        horizon_used = 30
    elif horizons:
        # Find closest to 30
        horizon_used = min(horizons, key=lambda h: abs(h - 30))
    else:
        horizon_used = 30

    label_col = f"fail_h{horizon_used}"
    p30 = (
        float(proba[label_col].iloc[0])
        if label_col in proba.columns
        else 0.0
    )

    # --- Tier assignment ---
    tier = RiskTier.LOW
    rule = ""

    # State-based overrides
    state_upper = health_state.upper() if health_state else ""
    state_forced = False
    if state_upper in thresholds.state_override_critical:
        tier = RiskTier.CRITICAL
        rule = f"Health state '{health_state}' triggers CRITICAL tier"
        state_forced = True
    elif state_upper in thresholds.state_override_high:
        if tier.severity_rank < RiskTier.HIGH.severity_rank:
            tier = RiskTier.HIGH
            rule = f"Health state '{health_state}' triggers HIGH tier"
            state_forced = True

    # P(+30)-based assignment (never downgrades a state-forced tier)
    if not state_forced or tier.severity_rank < RiskTier.CRITICAL.severity_rank:
        p30_tier = RiskTier.LOW
        p30_rule = f"P(+30)={p30:.3f} < {thresholds.p30_low} (Low threshold)"
        if p30 >= thresholds.p30_critical:
            p30_tier = RiskTier.CRITICAL
            p30_rule = (
                f"P(+30)={p30:.3f} >= {thresholds.p30_critical} (Critical threshold)"
            )
        elif p30 >= thresholds.p30_medium:
            p30_tier = RiskTier.HIGH
            p30_rule = f"P(+30)={p30:.3f} >= {thresholds.p30_medium} (High threshold)"
        elif p30 >= thresholds.p30_low:
            p30_tier = RiskTier.MEDIUM
            p30_rule = f"P(+30)={p30:.3f} >= {thresholds.p30_low} (Medium threshold)"

        if p30_tier.severity_rank > tier.severity_rank:
            tier = p30_tier
            rule = p30_rule
        elif not state_forced:
            rule = p30_rule

    # DQ adjustment
    tier_adj, confidence = _adjust_confidence(tier, dq_status)

    # Recommendation
    recommendation = _recommendation_for_tier(tier_adj)

    # Evidence summary
    cards_summary = evidence_cards if evidence_cards else []

    return MaintenanceGuidance(
        unit_id=unit_id,
        cycle=cycle,
        risk_tier=tier_adj,
        p30=p30,
        p30_confidence_interval=p30_ci,
        health_state=health_state,
        shi=shi,
        evidence_cards=cards_summary,
        dq_status=dq_status,
        confidence_level=confidence,
        recommendation=recommendation,
        rule_triggered=rule,
        horizon_used=horizon_used,
    )
