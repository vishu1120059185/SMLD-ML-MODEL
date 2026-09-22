"""Statistical significance tests for model comparison.

This module provides:

* **Paired Wilcoxon signed-rank test** – non-parametric test for
  differences between two paired sets of metric values.
* **Paired bootstrap confidence intervals** – resample-based CIs for
  the mean difference between two models.
* **Holm–Bonferroni correction** – controls family-wise error rate
  when performing multiple comparisons.
* **Effect sizes** – rank-biserial correlation (from Wilcoxon) and
  Cohen's d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import stats

__all__ = [
    "paired_wilcoxon",
    "paired_bootstrap_ci",
    "holm_bonferroni",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WilcoxonResult:
    """Outcome of a paired Wilcoxon signed-rank test.

    Attributes
    ----------
    statistic:
        Wilcoxon signed-rank statistic.
    p_value:
        Two-sided p-value.
    effect_size:
        Rank-biserial correlation (r = 1 - 2*W / (n*(n+1)/2)).
    n_pairs:
        Number of non-tied pairs used in the test.
    median_diff:
        Median of the paired differences.
    """

    statistic: float = np.nan
    p_value: float = np.nan
    effect_size: float = np.nan
    n_pairs: int = 0
    median_diff: float = np.nan


@dataclass
class BootstrapCI:
    """Paired bootstrap confidence interval for the mean difference.

    Attributes
    ----------
    mean_diff:
        Observed mean difference (model_a − model_b).
    ci_lower:
        Lower bound of the bootstrap CI.
    ci_upper:
        Upper bound of the bootstrap CI.
    se:
        Bootstrap standard error of the mean difference.
    n_resamples:
        Number of bootstrap resamples used.
    """

    mean_diff: float = np.nan
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    se: float = np.nan
    n_resamples: int = 0


@dataclass
class HolmBonferroniResult:
    """Outcome of Holm–Bonferroni correction for multiple comparisons.

    Attributes
    ----------
    adjusted_p_values:
        Mapping from hypothesis key to adjusted p-value.
    rejected:
        Mapping from hypothesis key to ``True`` (significant) or ``False``.
    alpha:
        Family-wise significance level.
    """

    adjusted_p_values: Dict[str, float] = field(default_factory=dict)
    rejected: Dict[str, bool] = field(default_factory=dict)
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# Paired Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def paired_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test for two sets of metric scores.

    Tests the null hypothesis that the distribution of paired differences
    ``scores_a − scores_b`` is symmetric about zero.

    Parameters
    ----------
    scores_a, scores_b:
        Paired metric values (e.g. per-unit MAE from two models).
    alternative:
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    WilcoxonResult
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    diff = scores_a - scores_b

    # Remove exact zeros (ties)
    non_zero = diff != 0
    diff_nz = diff[non_zero]
    n_pairs = len(diff_nz)

    if n_pairs == 0:
        return WilcoxonResult(n_pairs=0, median_diff=float(np.median(diff)))

    stat, p = stats.wilcoxon(diff_nz, alternative=alternative)

    # Rank-biserial effect size: r = 1 - 2*W / (n*(n+1)/2)
    n = n_pairs
    w_max = n * (n + 1) / 2.0
    effect_size = 1.0 - (2.0 * stat) / w_max if w_max > 0 else 0.0

    return WilcoxonResult(
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(effect_size),
        n_pairs=n_pairs,
        median_diff=float(np.median(diff)),
    )


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapCI:
    """Bootstrap confidence interval for the mean paired difference.

    Parameters
    ----------
    scores_a, scores_b:
        Paired metric values.
    n_resamples:
        Number of bootstrap resamples.
    confidence_level:
        Confidence level (e.g. 0.95 for 95 % CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    BootstrapCI
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    rng = np.random.default_rng(seed)
    diff = scores_a - scores_b
    n = len(diff)
    mean_obs = float(np.mean(diff))

    # Resample
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diff[idx])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_means, ddof=1))

    return BootstrapCI(
        mean_diff=mean_obs,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Holm–Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: Dict[str, float],
    alpha: float = 0.05,
) -> HolmBonferroniResult:
    """Apply Holm–Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values:
        Mapping from hypothesis name to raw (uncorrected) p-value.
    alpha:
        Family-wise significance level.

    Returns
    -------
    HolmBonferroniResult
    """
    if not p_values:
        return HolmBonferroniResult(alpha=alpha)

    # Sort by p-value (ascending)
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)

    adjusted: Dict[str, float] = {}
    rejected: Dict[str, bool] = {}

    for rank, (name, p_val) in enumerate(sorted_items, start=1):
        adj_p = min(p_val * (m - rank + 1), 1.0)
        # Ensure monotonicity: adjusted p-values must be non-decreasing
        if rank > 1:
            prev_name = sorted_items[rank - 2][0]
            adj_p = max(adj_p, adjusted[prev_name])
        adjusted[name] = adj_p
        rejected[name] = adj_p <= alpha

    return HolmBonferroniResult(
        adjusted_p_values=adjusted,
        rejected=rejected,
        alpha=alpha,
    )
