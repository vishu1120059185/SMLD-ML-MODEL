"""Evaluation module – metrics, lead-time analysis, and significance tests."""

from stattwin.evaluation.lead_time import (
    LeadTimeReport,
    lead_time_analysis,
    tune_tau_for_budget,
)
from stattwin.evaluation.metrics import (
    evaluate_classification,
    evaluate_rul,
    interval_metrics,
    probabilistic_metrics,
)
from stattwin.evaluation.significance import (
    paired_bootstrap_ci,
    paired_wilcoxon,
    holm_bonferroni,
)

__all__ = [
    "LeadTimeReport",
    "lead_time_analysis",
    "tune_tau_for_budget",
    "evaluate_classification",
    "evaluate_rul",
    "interval_metrics",
    "probabilistic_metrics",
    "paired_bootstrap_ci",
    "paired_wilcoxon",
    "holm_bonferroni",
]
