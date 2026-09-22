"""Explainability module for STAT-TWIN.

Provides tools for understanding model predictions:

* **Evidence cards** – top-k contributing sensors with rich diagnostic
  statistics (z-score, trend, EWMA deviation, variance ratio, PSI).
* **Attribution** – group occlusion, risk-change decomposition, and
  optional SHAP cross-check for tree-based models.

All explanations use the framing "contributed to the model's risk
estimate" rather than causal claims about failure.
"""

from stattwin.explainability.attribution import (
    AttributionResult,
    group_occlusion_attribution,
    risk_change_decomposition,
    shap_attribution,
)
from stattwin.explainability.cards import (
    EvidenceCard,
    EvidenceCardSet,
    build_evidence_cards,
    severity_label,
)

__all__ = [
    "AttributionResult",
    "EvidenceCard",
    "EvidenceCardSet",
    "build_evidence_cards",
    "group_occlusion_attribution",
    "risk_change_decomposition",
    "severity_label",
    "shap_attribution",
]
