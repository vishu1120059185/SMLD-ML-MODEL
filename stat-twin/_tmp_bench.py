"""One-off: verify e2 speedup + NaN fix on real FD001 data."""

import time

import numpy as np

from stattwin.config import load_config
from stattwin.data.loader import load_cmapss
from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.data.splitter import make_group_kfold_splits
from stattwin.experiments._common import (
    feature_columns,
    impute_causal,
    screen_features,
)
from stattwin.experiments.e2_model_comparison import _engineer_features
from stattwin.models import LogisticModel, RandomForestModel, XGBoostModel

cfg = load_config("configs/base.yaml", profile="fast", dataset="FD001")
df = load_cmapss(
    "data/raw/CMAPSS/train_FD001.txt", add_labels=True, rul_clip=cfg.dataset.rul_clip
)

t = time.time()
feats = _engineer_features(df, cfg)
print("engineer:", round(time.time() - t, 1), "s", feats.shape)

cols = feature_columns(feats, include_health=True)
y = df[[label_col_for(h) for h in FAILURE_HORIZONS]]
nan_cols = int(feats[cols].isna().any().sum())
print("library cols:", len(cols), "| cols containing NaN:", nan_cols)

splits = make_group_kfold_splits(df, n_splits=3, seed=42)
s = splits[0]
Xtr = feats[feats.unit_id.isin(s["train_units"])].copy()
Xva = feats[feats.unit_id.isin(s["val_units"])].copy()
ytr = y.loc[Xtr.index]

t = time.time()
keep = screen_features(Xtr, ytr, cols, top_k=300)
print("screen:", round(time.time() - t, 1), "s ->", len(keep), "of", len(cols))
print("kept sample:", keep[:6])

t = time.time()
impute_causal([Xtr, Xva], keep)
print("impute:", round(time.time() - t, 1), "s | NaN left:", int(Xtr[keep].isna().any().sum()))

fit = Xtr[keep + ["unit_id", "cycle", "RUL"]]
pred = Xva[keep + ["unit_id", "cycle"]]
truth = Xva["RUL"].to_numpy()

for name, model in [
    ("XGB", XGBoostModel(horizons=FAILURE_HORIZONS)),
    ("RF", RandomForestModel(horizons=FAILURE_HORIZONS)),
    ("Logistic", LogisticModel(horizons=FAILURE_HORIZONS)),
]:
    t = time.time()
    try:
        model.fit(fit, ytr)
    except Exception as exc:  # noqa: BLE001
        print(f"{name:9s} FAILED: {type(exc).__name__}: {exc}")
        continue
    dt = time.time() - t
    rul = model.predict_rul(pred)
    mae = float(np.mean(np.abs(rul.to_numpy() - truth)))
    proba = model.predict_proba(pred)
    auc = float(np.mean([proba[f"fail_h{h}"].mean() for h in FAILURE_HORIZONS]))
    print(f"{name:9s} fit {dt:6.1f}s | RUL MAE {mae:6.2f} | mean p(fail) {auc:.3f}")
