"""Decision support module for STAT-TWIN.

Non-prescriptive maintenance decision guidance:

* **Risk tiers** – Low, Medium, High, Critical based on predicted
  probability and health state.
* **Configurable rules** – calibrated on validation data.
* **Evidence display** – probability, confidence interval, top evidence
  cards, and data-quality status.

This module provides *decision support*, not prescriptive maintenance
instructions.  All recommendations should be reviewed by qualified
maintenance engineers.
"""

from stattwin.decision.guidance import (
    MaintenanceGuidance,
    RiskTier,
    generate_maintenance_guidance,
)

__all__ = [
    "MaintenanceGuidance",
    "RiskTier",
    "generate_maintenance_guidance",
]
