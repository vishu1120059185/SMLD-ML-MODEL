"""Auto-generate research report tables and figures from results/*.json.

This module reads all ``results/eN/eN_results.json`` files and produces:
  * Markdown tables for each experiment
  * Combined summary tables
  * A ``README_results.md`` file
  * No hand-typed numbers – everything comes from JSON files.

Usage::

    python -m stattwin.reports.generate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["generate_report"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_ROOT / "results"
_OUTPUT_DIR = _PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def _load_results(experiment: str) -> dict[str, Any] | None:
    """Load results JSON for an experiment directory."""
    path = _RESULTS_DIR / experiment / f"{experiment}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(val: Any, decimals: int = 4) -> str:
    """Format a value for Markdown tables."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "–"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ---------------------------------------------------------------------------
# Per-experiment table generators
# ---------------------------------------------------------------------------

def _table_e0(data: dict) -> str:
    """E0: Data Audit summary table."""
    us = data.get("unit_summary", {})
    cld = data.get("cycle_length_distribution", {})
    lines = [
        "## E0: Data Audit",
        "",
        f"**Dataset:** {data.get('dataset', '?')}  ",
        f"**Units:** {us.get('n_units', '?')}  ",
        f"**Rows:** {us.get('n_rows', '?')}  ",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cycles/unit (mean ± std) | {cld.get('mean', 0):.1f} ± {cld.get('std', 0):.1f} |",
        f"| Cycles/unit (min–max) | {cld.get('min', 0)}–{cld.get('max', 0)} |",
        (
            "| RUL range | "
            f"{data.get('unit_summary', {}).get('rul_range', {}).get('overall_min', '?')}–"
            f"{data.get('unit_summary', {}).get('rul_range', {}).get('overall_max', '?')} |"
        ),
    ]

    const = data.get("constant_sensors", {})
    if const.get("constant_sensors"):
        lines.append(f"| Constant sensors | {', '.join(const['constant_sensors'])} |")

    oc = data.get("operating_conditions", {})
    lines.append(f"| Operating regimes | {oc.get('n_clusters', '?')} |")
    lines.append("")

    return "\n".join(lines)


def _table_e1(data: dict) -> str:
    """E1: SHI Health-Index Validation."""
    qm = data.get("quality_metrics", {})
    lines = [
        "## E1: SHI Health-Index Validation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Monotonicity (mean) | {_fmt(qm.get('monotonicity_mean'))} |",
        f"| Trendability | {_fmt(qm.get('trendability'))} |",
        f"| Prognosability | {_fmt(qm.get('prognosability'))} |",
        f"| Spearman ρ (mean) | {_fmt(qm.get('spearman_rho_mean'))} |",
        "",
    ]
    return "\n".join(lines)


def _table_e2(data: dict) -> str:
    """E2: Model Comparison."""
    models = data.get("models", [])
    if not models:
        return "## E2: Model Comparison\n\n_No results._\n"

    # Build header
    horizons = [10, 20, 30, 40, 50]
    header = "| Model |"
    sep = "|-------|"
    for h in horizons:
        header += f" ROC-AUC h{h} |"
        sep += "---------|"
    header += " RUL RMSE | RUL MAE | NASA |"
    sep += "----------|---------|------|"

    lines = ["## E2: Model Comparison", "", header, sep]
    for m in models:
        if "error" in m:
            lines.append(f"| {m['model']} | _error_ |")
            continue
        row = f"| {m['model']} |"
        for h in horizons:
            entry = next((c for c in m.get("classification", []) if c["horizon"] == h), None)
            val = entry["roc_auc"] if entry else None
            row += f" {_fmt(val)} |"
        rul = m.get("rul", {})
        row += f" {_fmt(rul.get('rmse'))} | {_fmt(rul.get('mae'))} | {_fmt(rul.get('nasa_score'))} |"  # noqa: E501
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def _table_e3(data: dict) -> str:
    """E3: Early Warning."""
    models = data.get("models", [])
    lines = [
        "## E3: Early Warning Benchmark",
        "",
        f"**Horizon:** {data.get('horizon', '?')}  ",
        f"**FAR budget:** {data.get('far_budget', '?')}  ",
        "",
        "| Model | Mean Lead Time | Median Lead Time | FAR |",
        "|-------|---------------|-----------------|-----|",
    ]
    for m in models:
        if "error" in m:
            lines.append(f"| {m['model']} | _error_ | | |")
            continue
        lines.append(
            f"| {m['model']} | {_fmt(m.get('mean_lead_time'))} | "
            f"{_fmt(m.get('median_lead_time'))} | {_fmt(m.get('mean_far'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _table_e4(data: dict) -> str:
    """E4: Ablation Study."""
    variants = data.get("variants", {})
    lines = [
        "## E4: Ablation Study",
        "",
        "| Variant | Features | Mean MAE | Std MAE |",
        "|---------|----------|----------|---------|",
    ]
    for name, v in variants.items():
        lines.append(
            f"| {name} | {v.get('n_features', '?')} | "
            f"{_fmt(v.get('mean_mae'))} | {_fmt(v.get('std_mae'))} |"
        )

    # Holm-Bonferroni summary
    holm = data.get("holm_bonferroni", {})
    if holm.get("rejected"):
        lines.append("")
        lines.append("**Holm–Bonferroni significant differences (vs E_full):**")
        for key, rejected in holm.get("rejected", {}).items():
            if rejected:
                p = holm.get("adjusted_p_values", {}).get(key, "?")
                lines.append(f"  - {key}: adjusted p = {_fmt(p)}")
    lines.append("")
    return "\n".join(lines)


def _table_e5(data: dict) -> str:
    """E5: Uncertainty & Calibration."""
    cal = data.get("calibration", {})
    conf = data.get("conformal", {})
    lines = [
        "## E5: Uncertainty & Calibration",
        "",
        f"**Mean Brier:** {_fmt(cal.get('mean_brier'))}  ",
        f"**Mean ECE (equal-width):** {_fmt(cal.get('mean_ece_ew'))}  ",
        "",
        "### Per-Horizon Calibration",
        "",
        "| Horizon | Brier | ECE (EW) | ECE (EM) |",
        "|---------|-------|----------|----------|",
    ]
    for h_key, h_data in sorted(cal.get("per_horizon", {}).items()):
        lines.append(
            f"| {h_key} | {_fmt(h_data.get('brier'))} | "
            f"{_fmt(h_data.get('ece_equal_width'))} | {_fmt(h_data.get('ece_equal_mass'))} |"
        )

    lines.append("")
    lines.append("### Conformal Intervals")
    lines.append("")
    lines.append(f"- Alpha: {conf.get('alpha', '?')}")
    lines.append(f"- Quantile q: {_fmt(conf.get('quantile_q'))}")
    lines.append(f"- Coverage: {_fmt(conf.get('coverage'))}")
    lines.append(f"- Mean width: {_fmt(conf.get('mean_width'))}")
    lines.append("")
    return "\n".join(lines)


def _table_e6(data: dict) -> str:
    """E6: Cross-Dataset Generalization."""
    datasets = data.get("datasets", [])
    auc_matrix = data.get("auc_matrix", [])
    rmse_matrix = data.get("rmse_matrix", [])

    if not datasets:
        return "## E6: Cross-Dataset Generalization\n\n_No results._\n"

    lines = [
        "## E6: Cross-Dataset Generalization",
        "",
        "### ROC-AUC Heatmap (Train → Test)",
        "",
    ]

    # AUC table
    header = "| Train \\ Test |" + " | ".join(datasets) + " |"
    sep = "|--------------|" + " | ".join(["------"] * len(datasets)) + " |"
    lines.append(header)
    lines.append(sep)
    for i, train in enumerate(datasets):
        row = f"| {train} |"
        for j in range(len(datasets)):
            val = auc_matrix[i][j] if i < len(auc_matrix) and j < len(auc_matrix[i]) else None
            row += f" {_fmt(val)} |"
        lines.append(row)

    lines.append("")
    lines.append("### RUL RMSE Heatmap (Train → Test)")
    lines.append("")

    header = "| Train \\ Test |" + " | ".join(datasets) + " |"
    sep = "|--------------|" + " | ".join(["------"] * len(datasets)) + " |"
    lines.append(header)
    lines.append(sep)
    for i, train in enumerate(datasets):
        row = f"| {train} |"
        for j in range(len(datasets)):
            val = rmse_matrix[i][j] if i < len(rmse_matrix) and j < len(rmse_matrix[i]) else None
            row += f" {_fmt(val)} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _table_e7(data: dict) -> str:
    """E7: Operating-Condition Normalization."""
    comp = data.get("normalization_comparison", {})
    no_norm = comp.get("without_normalization", {})
    with_norm = comp.get("with_normalization", {})
    lines = [
        "## E7: Per-Condition Normalization",
        "",
        "| Setting | ROC-AUC (h30) | RUL RMSE |",
        "|----------|---------------|----------|",
        (
            f"| Without normalization | {_fmt(no_norm.get('mean_roc_auc_h30'))} "
            f"| {_fmt(no_norm.get('mean_rul_rmse'))} |"
        ),
        (
            f"| With normalization | {_fmt(with_norm.get('mean_roc_auc_h30'))} "
            f"| {_fmt(with_norm.get('mean_rul_rmse'))} |"
        ),
        "",
    ]
    return "\n".join(lines)


def _table_e8(data: dict) -> str:
    """E8: Fault Injection."""
    fi = data.get("fault_injection", {})
    lines = [
        "## E8: Fault Injection",
        "",
        "| Fault Type | Precision | Recall | F1 |",
        "|------------|-----------|--------|-----|",
    ]
    for ft in ["spike", "stuck"]:
        if ft in fi and "precision" in fi[ft]:
            r = fi[ft]
            lines.append(f"| {ft} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('f1'))} |")  # noqa: E501
    lines.append("")
    return "\n".join(lines)


def _table_e9(data: dict) -> str:
    """E9: Uncertainty Method Comparison."""
    comp = data.get("uncertainty_comparison", {})
    summary = comp.get("summary", {})
    lines = [
        "## E9: Uncertainty Method Comparison",
        "",
        f"**Alpha:** {data.get('alpha', '?')}  ",
        "",
        "| Method | PICP | Mean Width | Winkler |",
        "|--------|------|------------|---------|",
    ]
    for method, stats in summary.items():
        if "mean_picp" in stats:
            lines.append(
                f"| {method} | {_fmt(stats.get('mean_picp'))} | "
                f"{_fmt(stats.get('mean_width'))} | {_fmt(stats.get('mean_winkler'))} |"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

_TABLE_GENERATORS = {
    "e0_data_audit": _table_e0,
    "e1_health_index": _table_e1,
    "e2_model_comparison": _table_e2,
    "e3_early_warning": _table_e3,
    "e4_ablation": _table_e4,
    "e5_uncertainty": _table_e5,
    "e6_generalization": _table_e6,
    "e7_operating_conditions": _table_e7,
    "e8_fault_injection": _table_e8,
    "e9_uncertainty_comparison": _table_e9,
}


def generate_report(output_path: Path | None = None) -> Path:
    """Generate the full research report from results/*.json.

    Parameters
    ----------
    output_path:
        Where to write the Markdown report.  Defaults to
        ``reports/README_results.md``.

    Returns
    -------
    Path
        Path to the written report.
    """
    if output_path is None:
        output_path = _OUTPUT_DIR / "README_results.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        "# STAT-TWIN Research Results",
        "",
        "_Auto-generated from `results/*/e*_results.json`. Do not hand-edit._",
        "",
    ]

    for exp_name, gen_fn in _TABLE_GENERATORS.items():
        data = _load_results(exp_name)
        if data is None:
            sections.append(f"## {exp_name}\n\n_No results file found._\n")
            continue
        sections.append(gen_fn(data))

    report = "\n".join(sections)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")
    return output_path


def main() -> None:
    """CLI entry-point."""
    generate_report()


if __name__ == "__main__":
    main()
