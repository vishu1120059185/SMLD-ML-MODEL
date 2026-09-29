"""Print the current e2 results table from the checkpoint file."""

import json
import pathlib

d = json.loads(
    pathlib.Path("results/e2_model_comparison/e2_results.json").read_text(
        encoding="utf-8"
    )
)
print("partial:", d.get("partial"), "| features:", d.get("features"))
print("models:", [m["model"] for m in d["models"]])
for m in d["models"]:
    if "error" in m:
        print("  ERR", m["model"], str(m["error"])[:120])
        continue
    auc30 = next(
        (c["roc_auc"] for c in m["classification"] if c["horizon"] == 30), float("nan")
    )
    print(
        f"  {m['model']:16s} MAE={m['rul']['mae']:7.2f} RMSE={m['rul']['rmse']:7.2f} "
        f"NASA={m['rul']['nasa_score']:10.2f} AUC_h30={auc30:.4f} "
        f"folds={len(m.get('folds', []))}"
    )
