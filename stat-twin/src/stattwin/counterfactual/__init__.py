"""Counterfactual simulation module for STAT-TWIN.

Provides what-if analysis capabilities:

* **Simulator** – apply multiplicative, additive, or variance-scaling
  perturbations to recent sensor windows, recompute statistical features
  and SHI, and re-run the model to produce counterfactual predictions.

All simulation outputs carry a ``SIMULATION`` badge and a disclaimer
stating these are hypothetical scenarios, not real outcomes.
"""

from stattwin.counterfactual.simulator import (
    SimulationBadge,
    SimulationResult,
    WhatIfSimulator,
)

__all__ = [
    "SimulationBadge",
    "SimulationResult",
    "WhatIfSimulator",
]
