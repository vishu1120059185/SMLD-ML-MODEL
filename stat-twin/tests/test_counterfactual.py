"""Tests for counterfactual identity: zero perturbation reproduces original prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.counterfactual.simulator import (
    SimulationBadge,
    SimulationResult,
    WhatIfSimulator,
)

# ---------------------------------------------------------------------------
# Simple model for counterfactual tests
# ---------------------------------------------------------------------------


class SimpleModel:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        vals = X.values.mean(axis=1) * 0.01 if len(X.columns) > 0 else np.full(n, 0.5)
        vals = np.clip(vals, 0.0, 1.0)
        return pd.DataFrame(
            {f"fail_h{h}": vals for h in [10, 20, 30, 40, 50]},
            index=X.index,
        )

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(np.full(len(X), 60.0), index=X.index, name="RUL")


class TestIdentityTest:
    @pytest.fixture()
    def df(self):
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame(
            {
                "unit_id": np.ones(n, dtype=int),
                "cycle": np.arange(1, n + 1),
                "sensor_1": 10.0 + rng.normal(0, 1, n),
                "sensor_2": 20.0 + rng.normal(0, 2, n),
            }
        )

    def test_identity_zero_perturbation(self, df):
        model = SimpleModel()
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = ["sensor_1", "sensor_2"]

        sim = WhatIfSimulator(
            model=model,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
        )

        result = sim.identity_test(df, unit_id=1, cycle=50)
        assert isinstance(result, SimulationResult)
        assert result.identity_verified

    def test_identity_probas_match(self, df):
        model = SimpleModel()
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = ["sensor_1", "sensor_2"]

        sim = WhatIfSimulator(
            model=model,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
        )

        result = sim.identity_test(df, unit_id=1, cycle=50)
        # Original and simulated probabilities should be very close
        orig_vals = result.original_proba.values
        sim_vals = result.simulated_proba.values
        np.testing.assert_allclose(orig_vals, sim_vals, atol=1e-6)

    def test_identity_rul_match(self, df):
        model = SimpleModel()
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = ["sensor_1", "sensor_2"]

        sim = WhatIfSimulator(
            model=model,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
        )

        result = sim.identity_test(df, unit_id=1, cycle=50)
        assert result.original_rul == pytest.approx(result.simulated_rul, abs=1e-6)

    def test_simulation_badge(self):
        badge = SimulationBadge()
        assert badge.is_simulation is True
        assert "SIMULATION" in badge.disclaimer
        assert badge.timestamp != ""


class TestWhatIfSimulatorRun:
    @pytest.fixture()
    def df(self):
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame(
            {
                "unit_id": np.ones(n, dtype=int),
                "cycle": np.arange(1, n + 1),
                "sensor_1": 10.0 + rng.normal(0, 1, n),
                "sensor_2": 20.0 + rng.normal(0, 2, n),
            }
        )

    def test_multiplicative_perturbation(self, df):
        model = SimpleModel()
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = ["sensor_1", "sensor_2"]

        sim = WhatIfSimulator(
            model=model,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
        )

        result = sim.run(
            df, unit_id=1, cycle=50,
            sensor_perturbations={
                "sensor_1": {
                    "type": "multiplicative",
                    "factor": 0.5,
                    "start_cycle": 50,
                    "end_cycle": 60,
                },
            },
        )
        assert isinstance(result, SimulationResult)
        assert result.perturbation_type == "mixed"
        assert result.badge.is_simulation

    def test_additive_perturbation(self, df):
        model = SimpleModel()
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = ["sensor_1", "sensor_2"]

        sim = WhatIfSimulator(
            model=model,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
        )

        result = sim.run(
            df, unit_id=1, cycle=50,
            sensor_perturbations={
                "sensor_1": {"type": "additive", "offset": 5.0, "start_cycle": 50, "end_cycle": 60},
            },
        )
        assert isinstance(result, SimulationResult)
        assert "sensor_1" in result.perturbed_sensors

    def test_simulation_result_summary(self, df):
        model = SimpleModel()
        sensor_cols = ["sensor_1", "sensor_2"]
        feature_cols = ["sensor_1", "sensor_2"]

        sim = WhatIfSimulator(
            model=model,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
        )

        result = sim.run(
            df, unit_id=1, cycle=50,
            sensor_perturbations={
                "sensor_1": {"type": "additive", "offset": 5.0, "start_cycle": 50, "end_cycle": 60},
            },
        )
        summary = result.summary()
        assert "badge" in summary
        assert "rul_delta" in summary
        assert "shi_delta" in summary
