"""Health monitoring module for STAT-TWIN.

Phase 3: Statistical Health Index (SHI), health state classification,
and health index quality metrics.

* **SHI** – composite degradation score (0-100) from multiple evidence
  components, each normalised by train-fitted ECDF.
* **Health states** – discrete states with persistence (hysteresis) to
  avoid rapid oscillation between states.
* **Quality metrics** – monotonicity, trendability, prognosability, and
  Spearman correlation with true RUL.
"""

from stattwin.health.quality import (
    compute_quality_metrics,
    monotonicity,
    prognosability,
    spearman_rul_correlation,
    trendability,
)
from stattwin.health.shi import (
    EvidenceComponent,
    HealthIndex,
    compute_shi,
)
from stattwin.health.states import (
    HealthState,
    StateClassifier,
    classify_states,
)

__all__ = [
    "EvidenceComponent",
    "HealthIndex",
    "HealthState",
    "StateClassifier",
    "classify_states",
    "compute_quality_metrics",
    "compute_shi",
    "monotonicity",
    "prognosability",
    "spearman_rul_correlation",
    "trendability",
]
