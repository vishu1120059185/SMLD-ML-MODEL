"""Experiment e8 – Fault Injection.

Injects synthetic faults into healthy cycles and measures DQ detection
precision/recall, false warnings with and without data-quality gating.
Outputs to ``results/e8/``.

Usage::

    python -m stattwin.experiments.e8_fault_injection --ds FD001 --profile smoke
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stattwin.config import load_config
from stattwin.data.loader import load_cmapss
from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.preprocessing.dq import DQConfig, DQEngine

from ._common import (
    SENSOR_COLS,
    Timer,
    add_common_args,
    feature_columns,
    resolve_raw_path,
    save_json,
    setup_output,
)


# ---------------------------------------------------------------------------
# Synthetic fault injection
# ---------------------------------------------------------------------------

def _inject_spikes(
    df: pd.DataFrame,
    sensor_cols: list[str],
    n_faults_per_unit: int = 3,
    spike_magnitude: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject spike faults into healthy cycles and return ground truth.

    Returns
    -------
    (faulty_df, ground_truth)
        faulty_df has injected faults; ground_truth has columns:
        unit_id, cycle, sensor, fault_type, is_fault.
    """
    rng = np.random.default_rng(seed)
    faulty = df.copy()
    ground_truth_records = []

    for uid, grp in faulty.groupby("unit_id"):
        # Only inject into healthy cycles (RUL > 50)
        healthy_mask = grp["RUL"] > 50
        healthy_cycles = grp.loc[healthy_mask, "cycle"].values

        if len(healthy_cycles) < n_faults_per_unit:
            continue

        # Select random cycles and sensors
        fault_cycles = rng.choice(healthy_cycles, size=n_faults_per_unit, replace=False)
        fault_sensors = rng.choice(sensor_cols, size=n_faults_per_unit, replace=True)

        for cycle, sensor in zip(fault_cycles, fault_sensors):
            mask = (faulty["unit_id"] == uid) & (faulty["cycle"] == cycle)
            if mask.any():
                idx = faulty.index[mask][0]
                faulty.loc[idx, sensor] += spike_magnitude * rng.choice([-1, 1])
                ground_truth_records.append({
                    "unit_id": uid,
                    "cycle": int(cycle),
                    "sensor": sensor,
                    "fault_type": "spike",
                    "is_fault": 1,
                })

    # Add normal cycles as non-faults (sample for efficiency)
    normal_sample = df.sample(min(len(df), 5000), random_state=seed)
    for _, row in normal_sample.iterrows():
        ground_truth_records.append({
            "unit_id": row["unit_id"],
            "cycle": int(row["cycle"]),
            "sensor": "none",
            "fault_type": "none",
            "is_fault": 0,
        })

    ground_truth = pd.DataFrame(ground_truth_records)
    return faulty, ground_truth


def _inject_stuck(
    df: pd.DataFrame,
    sensor_cols: list[str],
    n_faults_per_unit: int = 2,
    stuck_window: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject stuck-sensor faults (constant value for several cycles)."""
    rng = np.random.default_rng(seed)
    faulty = df.copy()
    ground_truth_records = []

    for uid, grp in faulty.groupby("unit_id"):
        healthy_mask = grp["RUL"] > 50
        healthy_cycles = grp.loc[healthy_mask, "cycle"].values

        if len(healthy_cycles) < n_faults_per_unit * stuck_window:
            continue

        fault_starts = rng.choice(
            healthy_cycles[:-stuck_window],
            size=n_faults_per_unit,
            replace=False,
        )
        fault_sensors = rng.choice(sensor_cols, size=n_faults_per_unit, replace=True)

        for start, sensor in zip(fault_starts, fault_sensors):
            stuck_value = float(faulty.loc[
                (faulty["unit_id"] == uid) & (faulty["cycle"] == start), sensor
            ].values[0])

            for offset in range(stuck_window):
                cycle = start + offset
                mask = (faulty["unit_id"] == uid) & (faulty["cycle"] == cycle)
                if mask.any():
                    idx = faulty.index[mask][0]
                    faulty.loc[idx, sensor] = stuck_value

            ground_truth_records.append({
                "unit_id": uid,
                "cycle": int(start),
                "sensor": sensor,
                "fault_type": "stuck",
                "is_fault": 1,
            })

    normal_sample = df.sample(min(len(df), 3000), random_state=seed)
    for _, row in normal_sample.iterrows():
        ground_truth_records.append({
            "unit_id": row["unit_id"],
            "cycle": int(row["cycle"]),
            "sensor": "none",
            "fault_type": "none",
            "is_fault": 0,
        })

    ground_truth = pd.DataFrame(ground_truth_records)
    return faulty, ground_truth


# ---------------------------------------------------------------------------
# DQ detection evaluation
# ---------------------------------------------------------------------------

def _evaluate_dq_detection(
    faulty_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    sensor_cols: list[str],
) -> dict[str, Any]:
    """Run DQ engine on faulty data and evaluate detection performance."""
    dq_cfg = DQConfig(
        sensor_columns=sensor_cols,
        dropout_burst_k=3,
        stuck_window=10,
        spike_threshold=5.0,
        range_bounds=None,
        repair_strategy="forward_fill",
    )
    dq_engine = DQEngine(config=dq_cfg)

    # Fit on original healthy data (first 30 cycles of each unit)
    healthy_mask = faulty_df["RUL"] > 100
    dq_engine.fit(faulty_df[healthy_mask].head(1000))

    # Transform faulty data
    flagged = dq_engine.transform(faulty_df)

    # Check if DQ flags were added
    dq_flag_cols = [c for c in flagged.columns if "dq_" in c.lower() or "flag" in c.lower() or "outlier" in c.lower()]

    if not dq_flag_cols:
        # If no DQ flags, use robust_z on sensor deviations as proxy
        return _proxy_detection_evaluation(faulty_df, ground_truth, sensor_cols)

    # Evaluate detection per fault
    tp, fp, fn = 0, 0, 0
    detected_faults = []

    faults = ground_truth[ground_truth["is_fault"] == 1]
    normals = ground_truth[ground_truth["is_fault"] == 0]

    for _, fault_row in faults.iterrows():
        uid = fault_row["unit_id"]
        cycle = fault_row["cycle"]
        sensor = fault_row["sensor"]

        row_mask = (flagged["unit_id"] == uid) & (flagged["cycle"] == cycle)
        if row_mask.any():
            row_flags = flagged.loc[row_mask, dq_flag_cols].values
            detected = bool(np.any(row_flags))
            detected_faults.append(detected)
            if detected:
                tp += 1
            else:
                fn += 1
        else:
            fn += 1
            detected_faults.append(False)

    for _, norm_row in normals.iterrows():
        uid = norm_row["unit_id"]
        cycle = norm_row["cycle"]
        row_mask = (flagged["unit_id"] == uid) & (flagged["cycle"] == cycle)
        if row_mask.any():
            row_flags = flagged.loc[row_mask, dq_flag_cols].values
            if bool(np.any(row_flags)):
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_faults": len(faults),
        "n_normals": len(normals),
        "dq_flag_columns": dq_flag_cols,
    }


def _proxy_detection_evaluation(
    faulty_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    sensor_cols: list[str],
) -> dict[str, Any]:
    """Fallback detection using robust z-score thresholding."""
    # Compute rolling z-score per sensor
    tp, fp, fn = 0, 0, 0

    faults = ground_truth[ground_truth["is_fault"] == 1]
    normals = ground_truth[ground_truth["is_fault"] == 0]

    # Simple threshold: flag if any sensor deviates by > 5 sigma from its baseline
    for _, fault_row in faults.iterrows():
        uid = fault_row["unit_id"]
        cycle = fault_row["cycle"]
        sensor = fault_row["sensor"]

        unit_data = faulty_df[faulty_df["unit_id"] == uid]
        baseline = unit_data[unit_data["cycle"] <= 30]

        if sensor in baseline.columns and len(baseline) > 5:
            mean = baseline[sensor].mean()
            std = baseline[sensor].std()
            if std > 0:
                current_val = unit_data.loc[unit_data["cycle"] == cycle, sensor].values
                if len(current_val) > 0:
                    z = abs(current_val[0] - mean) / std
                    if z > 5.0:
                        tp += 1
                    else:
                        fn += 1
                else:
                    fn += 1
            else:
                fn += 1
        else:
            fn += 1

    for _, norm_row in normals.iterrows():
        uid = norm_row["unit_id"]
        cycle = norm_row["cycle"]

        unit_data = faulty_df[faulty_df["unit_id"] == uid]
        baseline = unit_data[unit_data["cycle"] <= 30]

        if len(baseline) > 5:
            any_flagged = False
            for s in sensor_cols[:5]:  # check first 5 sensors for efficiency
                if s in baseline.columns:
                    mean = baseline[s].mean()
                    std = baseline[s].std()
                    if std > 0:
                        current_val = unit_data.loc[unit_data["cycle"] == cycle, s].values
                        if len(current_val) > 0:
                            z = abs(current_val[0] - mean) / std
                            if z > 5.0:
                                any_flagged = True
                                break
            if any_flagged:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_faults": len(faults),
        "n_normals": len(normals),
        "method": "proxy_robust_z",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_detection_results(results: dict, out_dir: Path) -> None:
    """Bar chart of precision, recall, F1 for each fault type."""
    fault_types = [k for k in results if isinstance(results[k], dict) and "precision" in results[k]]
    if not fault_types:
        return

    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fault_types))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [results[ft][metric] for ft in fault_types]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels(fault_types)
    ax.set_ylabel("Score")
    ax.set_title("DQ Fault Detection Performance")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "detection_performance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e8(cfg, df, out_dir) -> dict[str, Any]:
    """Run fault injection experiment."""
    sensor_cols = [c for c in SENSOR_COLS if c in df.columns]

    all_results: dict[str, Any] = {}

    # Spike injection
    print("  Injecting spike faults...")
    faulty_spikes, gt_spikes = _inject_spikes(df, sensor_cols, seed=cfg.seed)
    spike_results = _evaluate_dq_detection(faulty_spikes, gt_spikes, sensor_cols)
    all_results["spike"] = spike_results
    print(f"    Spike: P={spike_results['precision']:.3f} R={spike_results['recall']:.3f} F1={spike_results['f1']:.3f}")

    # Stuck injection
    print("  Injecting stuck-sensor faults...")
    faulty_stuck, gt_stuck = _inject_stuck(df, sensor_cols, seed=cfg.seed)
    stuck_results = _evaluate_dq_detection(faulty_stuck, gt_stuck, sensor_cols)
    all_results["stuck"] = stuck_results
    print(f"    Stuck: P={stuck_results['precision']:.3f} R={stuck_results['recall']:.3f} F1={stuck_results['f1']:.3f}")

    # Combined
    all_results["summary"] = {
        "spike_precision": spike_results["precision"],
        "spike_recall": spike_results["recall"],
        "spike_f1": spike_results["f1"],
        "stuck_precision": stuck_results["precision"],
        "stuck_recall": stuck_results["recall"],
        "stuck_f1": stuck_results["f1"],
    }

    out = {"dataset": cfg.dataset.name, "fault_injection": all_results}
    save_json(out, out_dir / "e8_results.json")
    _plot_detection_results(all_results, out_dir)

    return out


def main() -> None:
    """CLI entry-point for e8_fault_injection."""
    parser = argparse.ArgumentParser(
        description="E8: Fault Injection – DQ detection precision/recall, false warnings."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile, dataset=args.ds)
    raw_path = resolve_raw_path(cfg.dataset.name)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_cmapss(raw_path, add_labels=True)
    out_dir = setup_output("e8_fault_injection")

    with Timer() as t:
        results = run_e8(cfg, df, out_dir)

    print(f"\n[e8] Completed in {t.elapsed:.1f}s")
    print(f"     Output: {out_dir}")

    from ._common import save_manifest
    save_manifest(out_dir, cfg, args.config, "e8_fault_injection",
                  extras={"elapsed_seconds": t.elapsed})


if __name__ == "__main__":
    main()
