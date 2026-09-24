"""Leakage Guard: Training on shuffled labels yields ROC-AUC ~ 0.5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for


class TestShuffledLabelSanity:
    def test_shuffled_labels_random_auc(self, synthetic_dataset):
        """Training with shuffled labels should produce ROC-AUC ~ 0.5 (random)."""
        from stattwin.models.xgboost_model import XGBoostModel

        df = synthetic_dataset.copy()
        [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0][:3]
        exclude = {"unit_id", "cycle", "RUL"}
        exclude.update(label_col_for(h) for h in FAILURE_HORIZONS)
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and c not in {"op_setting_1", "op_setting_2", "op_setting_3"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Use only one horizon for speed
        h = 30
        label_col = label_col_for(h)

        model = XGBoostModel(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            horizons=[h],
        )

        X = df[feature_cols]
        y_orig = df[label_col].copy()
        y_shuffled = y_orig.sample(frac=1, random_state=42).reset_index(drop=True)

        # Fit on shuffled labels
        y_df = pd.DataFrame({label_col: y_shuffled})
        model.fit(X, y_df)

        # Predict on original features
        proba = model.predict_proba(X)
        prob_vals = proba[label_col].values
        y_true = y_orig.values

        # ROC-AUC should be near 0.5 for shuffled labels (if model didn't overfit)
        from sklearn.metrics import roc_auc_score
        # Only compute if both classes present
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, prob_vals)
            # With very few estimators and shuffled labels, AUC should be close to 0.5
            assert 0.3 < auc < 0.7, f"AUC={auc} with shuffled labels, expected ~0.5"
