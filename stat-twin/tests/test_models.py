"""Tests for model interface (fit/predict_proba/predict_rul) and basic training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.data.synthetic import make_synthetic_dataset
from stattwin.models.base import BaseModel, ModelResult


# ---------------------------------------------------------------------------
# Minimal concrete model for testing the ABC
# ---------------------------------------------------------------------------


class DummyModel(BaseModel):
    """Trivial model that returns constant predictions."""

    def __init__(self, horizons=None):
        super().__init__(horizons=horizons, name="DummyModel")

    def fit(self, X_train, y_train, groups=None):
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        n = len(X)
        data = {label_col_for(h): np.full(n, 0.5) for h in self.horizons}
        return pd.DataFrame(data, index=X.index)

    def predict_rul(self, X):
        return pd.Series(np.full(len(X), 60.0), index=X.index, name="RUL")

    def score_raw(self, X):
        return pd.Series(np.full(len(X), 0.5), index=X.index, name="raw_score")


# ---------------------------------------------------------------------------
# BaseModel ABC tests
# ---------------------------------------------------------------------------


class TestBaseModel:
    def test_dummy_model_fit(self):
        model = DummyModel()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.DataFrame({"fail_h30": [0, 1]})
        model.fit(X, y)
        assert model.is_fitted

    def test_dummy_model_predict_proba(self):
        model = DummyModel()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.DataFrame({label_col_for(h): [0, 1] for h in FAILURE_HORIZONS})
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert isinstance(proba, pd.DataFrame)
        for h in FAILURE_HORIZONS:
            assert label_col_for(h) in proba.columns
        assert (proba >= 0).all().all()
        assert (proba <= 1).all().all()

    def test_dummy_model_predict_rul(self):
        model = DummyModel()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.DataFrame({label_col_for(h): [0, 1] for h in FAILURE_HORIZONS})
        model.fit(X, y)

        rul = model.predict_rul(X)
        assert isinstance(rul, pd.Series)
        assert (rul >= 0).all()

    def test_dummy_model_score_raw(self):
        model = DummyModel()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.DataFrame({label_col_for(h): [0, 1] for h in FAILURE_HORIZONS})
        model.fit(X, y)

        scores = model.score_raw(X)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(X)

    def test_predict_result(self):
        model = DummyModel()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y = pd.DataFrame({label_col_for(h): [0, 1] for h in FAILURE_HORIZONS})
        model.fit(X, y)

        result = model.predict_result(X, fold=0)
        assert isinstance(result, ModelResult)
        assert result.fold == 0
        assert result.model_name == "DummyModel"

    def test_label_cols(self):
        model = DummyModel(horizons=[10, 30])
        assert model.label_cols == ["fail_h10", "fail_h30"]

    def test_repr(self):
        model = DummyModel()
        assert "not fitted" in repr(model)
        model.is_fitted = True
        assert "fitted" in repr(model)


# ---------------------------------------------------------------------------
# XGBoost model integration test
# ---------------------------------------------------------------------------


class TestXGBoostModel:
    @pytest.fixture()
    def training_data(self):
        df = make_synthetic_dataset(n_units=8, n_cycles_range=(80, 150), seed=42)
        sensor_cols = [c for c in df.columns if c.startswith("sensor_") and df[c].sum() != 0][:3]
        exclude = {"unit_id", "cycle", "RUL"}
        exclude.update(label_col_for(h) for h in FAILURE_HORIZONS)
        feature_cols = [
            c for c in df.columns
            if c not in exclude and c not in {"op_setting_1", "op_setting_2", "op_setting_3"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]
        return df, feature_cols, label_cols, sensor_cols

    def test_xgboost_fit_predict(self, training_data):
        from stattwin.models.xgboost_model import XGBoostModel

        df, feature_cols, label_cols, sensor_cols = training_data
        model = XGBoostModel(
            n_estimators=10,
            max_depth=3,
            random_state=42,
        )
        X = df[feature_cols]
        y = df[label_cols]
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape[0] == len(X)
        for h in FAILURE_HORIZONS:
            assert label_col_for(h) in proba.columns

        rul = model.predict_rul(X)
        assert len(rul) == len(X)
        assert (rul >= 0).all()

        scores = model.score_raw(X)
        assert len(scores) == len(X)
