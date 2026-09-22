"""Run all experiments on FD001 with fast profile."""
import warnings
warnings.filterwarnings('ignore')

import time
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from stattwin.config import load_config
from stattwin.data.loader import load_cmapss
from stattwin.health.shi import compute_shi
from stattwin.health.states import classify_states
from stattwin.health.quality import compute_quality_metrics
from stattwin.experiments._common import resolve_raw_path, SENSOR_COLS, Timer, save_json, setup_output, save_manifest

def run_e1():
    cfg = load_config('configs/base.yaml', profile='fast', dataset='FD001')
    df = load_cmapss(resolve_raw_path('FD001'), add_labels=True)
    sensor_cols = [c for c in SENSOR_COLS if c in df.columns]
    out_dir = setup_output('e1_health_index')

    with Timer() as t:
        print('[e1] Computing SHI...')
        hi = compute_shi(df, sensor_cols=sensor_cols, baseline_cycles=30)
        shi_df = hi.shi_values
        shi_min = float(shi_df['shi'].min())
        shi_max = float(shi_df['shi'].max())
        print(f'[e1] SHI done in {t.elapsed:.1f}s, range: [{shi_min:.1f}, {shi_max:.1f}]')

        print('[e1] Computing quality metrics...')
        quality = compute_quality_metrics(shi_df, rul_df=df)
        qm = quality.summary
        for k, v in qm.items():
            print(f'  {k}: {v}')

        print('[e1] Classifying states...')
        states_df = classify_states(shi_df, method='fixed_grid', rul_df=df)

        results = {
            'shi_range': [shi_min, shi_max],
            'quality_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in qm.items()},
            'threshold_sensitivity': [{'grid': [80,60,40,20], 'note': 'default fixed grid'}],
            'elapsed_seconds': round(t.elapsed, 2),
        }
        save_json(results, out_dir / 'e1_results.json')
        save_manifest(out_dir, cfg, 'configs/base.yaml', 'e1_health_index', extras={'elapsed_seconds': t.elapsed})

    print(f'[e1] COMPLETED in {t.elapsed:.1f}s -> {out_dir}')
    return results

if __name__ == "__main__":
    run_e1()
