"""Quick smoke test for the full pipeline."""
import warnings
warnings.filterwarnings('ignore')

import time
import sys

def main():
    from stattwin.config import load_config
    from stattwin.data.loader import load_cmapss
    from stattwin.health.shi import compute_shi
    from stattwin.health.states import classify_states
    from stattwin.experiments._common import resolve_raw_path, SENSOR_COLS

    print("Loading config...")
    cfg = load_config('configs/base.yaml', profile='fast', dataset='FD001')

    print("Loading FD001...")
    df = load_cmapss(resolve_raw_path('FD001'), add_labels=True)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} cols, {df['unit_id'].nunique()} units")

    sensor_cols = [c for c in SENSOR_COLS if c in df.columns]
    print(f"  Sensor cols: {len(sensor_cols)}")

    t0 = time.time()
    print("Computing SHI...")
    shi = compute_shi(df, sensor_cols=sensor_cols, baseline_cycles=30)
    t1 = time.time()
    print(f"  SHI done in {t1-t0:.1f}s")
    print(f"  SHI range: [{shi.shi_values['shi'].min():.1f}, {shi.shi_values['shi'].max():.1f}]")

    print("Classifying states...")
    states = classify_states(shi.shi_values, method='fixed_grid', rul_df=df)
    t2 = time.time()
    print(f"  States done in {t2-t1:.1f}s")

    print(f"\nTotal time: {t2-t0:.1f}s")
    print("SUCCESS")

if __name__ == "__main__":
    main()
