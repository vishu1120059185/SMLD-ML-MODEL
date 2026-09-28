"""Regression tests for the correctness + quality overhaul.

Covers:
  * ``EnsembleModel`` learned/uniform weights and uncertainty signal
  * ``HybridModel`` tabular-fallback (all horizons trained) + monotone
    probability envelope
  * ``_nasa_score`` asymmetry direction (late vs early, same |error|)
  * ``rul_clip`` in :func:`load_cmapss` / ``_compute_rul``
  * SHI ``corr_shift`` routing (correlation, not variance)
  * ``GRUModel`` feature normalisation, RUL target scaling, and
    row-index alignment under shuffled input
  * ``LogisticModel`` RUL tail extrapolation past the last horizon
  * e9 uncertainty helpers (ensemble variance, quantile offsets,
    residual-augmented bootstrap)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.evaluation.metrics import _nasa_score
from stattwin.models import (
    EnsembleModel,
    GRUModel,
    HybridModel,
    LogisticModel,
    build_member,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panel(n_units: int = 10, n_cycles: int = 70, seed: int = 0) -> pd.DataFrame:
    """Small panel with raw sensors, engineered cols, RUL, and labels.

    ``n_cycles`` must exceed the largest failure horizon (50) so that
    every ``fail_h{h}`` column contains both classes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            rul = n_cycles - c
            row = {"unit_id": u, "cycle": c, "RUL": float(rul)}
            row["sensor_1"] = 100.0 + u + 0.5 * c + rng.normal(0, 1)
            row["sensor_2"] = 200.0 + 0.3 * c + rng.normal(0, 1)
            row["op_setting_1"] = 1.0
            # engineered-looking column
            row["rolling_mean_sensor_1_5"] = 100.0 + u + 0.25 * c
            for h in FAILURE_HORIZONS:
                row[label_col_for(h)] = int(rul <= h)
            rows.append(row)
    return pd.DataFrame(rows)


def _labels(df: pd.DataFrame) -> pd.DataFrame:
    return df[[label_col_for(h) for h in FAILURE_HORIZONS]].copy()


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------


class TestEnsembleModel:
    def test_learned_weights_sum_to_one(self):
        df = _panel(n_units=8, n_cycles=70)
        ens = EnsembleModel(
            members=[
                (name, build_member(name, _dummy_cfg(), FAILURE_HORIZONS))
                for name in ("logistic", "random_forest")
            ],
            weight_mode="learned",
            val_fraction=0.3,
            horizons=FAILURE_HORIZONS,
        )
        ens.fit(df, _labels(df))
        for weights in (ens.prob_weights_, ens.rul_weights_):
            assert set(weights) == {"logistic", "random_forest"}
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_weights_when_single_member(self):
        df = _panel(n_units=6, n_cycles=70)
        ens = EnsembleModel(
            members=[("logistic", build_member("logistic", _dummy_cfg(), FAILURE_HORIZONS))],
            weight_mode="learned",
            horizons=FAILURE_HORIZONS,
        )
        ens.fit(df, _labels(df))
        assert ens.prob_weights_["logistic"] == pytest.approx(1.0)

    def test_predict_and_uncertainty_shapes(self):
        df = _panel(n_units=6, n_cycles=70)
        ens = EnsembleModel(
            members=[
                (name, build_member(name, _dummy_cfg(), FAILURE_HORIZONS))
                for name in ("logistic", "random_forest", "xgboost")
            ],
            weight_mode="uniform",
            horizons=FAILURE_HORIZONS,
        )
        ens.fit(df, _labels(df))
        proba = ens.predict_proba(df)
        assert proba.shape == (len(df), len(FAILURE_HORIZONS))
        assert ((proba >= 0) & (proba <= 1)).all().all()
        unc = ens.predict_uncertainty(df)
        assert len(unc) == len(df)
        assert (unc >= 0).all()


def _dummy_cfg():
    """Minimal config object for ``build_member`` (only ``.model`` is read)."""
    from stattwin.config import STATTWINConfig

    return STATTWINConfig()


# ---------------------------------------------------------------------------
# HybridModel
# ---------------------------------------------------------------------------


class TestHybridModel:
    def test_hybrid_proba_live_for_every_horizon(self):
        """Regression: the tabular-fallback loop body was dedented, so only
        the last horizon was actually fitted and the rest returned constants.
        """
        df = _panel(n_units=8, n_cycles=70, seed=1)
        model = HybridModel(
            hidden=8, layers=1, epochs=1, patience=1, batch_size=64,
            ensemble_size=1, use_tabular_fallback=True,
            horizons=FAILURE_HORIZONS,
        )
        model.fit(df, _labels(df))
        proba = model.predict_proba(df)
        for h in FAILURE_HORIZONS:
            col = label_col_for(h)
            assert col in proba.columns
            assert proba[col].std() > 0, f"{col} constant -> horizon not trained"

    def test_enforce_monotone_is_cumulative_max(self):
        from stattwin.models.hybrid import _enforce_monotone

        rng = np.random.default_rng(3)
        raw = pd.DataFrame(
            rng.uniform(0, 1, size=(20, len(FAILURE_HORIZONS))),
            columns=[label_col_for(h) for h in FAILURE_HORIZONS],
        )
        out = _enforce_monotone(raw, list(FAILURE_HORIZONS))
        arr = out.to_numpy()
        # monotone non-decreasing across horizons (np.maximum.accumulate)
        assert np.all(np.diff(arr, axis=1) >= -1e-12)


# ---------------------------------------------------------------------------
# NASA score
# ---------------------------------------------------------------------------


class TestNasaScore:
    def test_late_error_penalised_more_than_early(self):
        """Over-prediction (late) must cost more than under-prediction."""
        y_true = np.array([100.0, 100.0])
        late = _nasa_score(y_true, np.array([115.0, 100.0]))  # d = +15
        early = _nasa_score(y_true, np.array([85.0, 100.0]))  # d = -15
        assert late > early, (late, early)

    def test_perfect_prediction_is_zero(self):
        y_true = np.array([50.0, 120.0])
        assert _nasa_score(y_true, y_true) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RUL clip
# ---------------------------------------------------------------------------


class TestRulClip:
    def test_compute_rul_clips_to_cap(self):
        from stattwin.data.loader import _compute_rul

        # A unit with 200 cycles would otherwise yield RUL up to 199
        df = pd.DataFrame({"unit_id": 1, "cycle": np.arange(1, 201)})
        out_unclipped = _compute_rul(df, rul_clip=None)
        out_clipped = _compute_rul(df, rul_clip=125)
        rul_unclipped = out_unclipped["RUL"]
        rul_clipped = out_clipped["RUL"]
        assert rul_unclipped.max() == 199
        assert rul_clipped.max() == 125
        # short RUL (near failure) is unaffected by the cap
        short_mask = rul_unclipped < 125
        assert np.allclose(
            rul_unclipped[short_mask].to_numpy(),
            rul_clipped[short_mask].to_numpy(),
        )


# ---------------------------------------------------------------------------
# SHI corr_shift routing
# ---------------------------------------------------------------------------


class TestShiCorrShift:
    def test_corr_shift_differs_from_variance_kernel(self):
        """Regression: corr_shift was mistakenly routed to the variance kernel."""
        from stattwin.health.shi import _compute_corr_shift, _compute_variance

        values = np.linspace(100, 140, 40)  # strong linear drift
        baseline_values = values[:20]
        corr = _compute_corr_shift(values, baseline_values, window=10)
        var = _compute_variance(values, window=10)
        assert corr.shape == var.shape
        assert not np.allclose(corr, var), "corr_shift returned variance output"

    def test_corr_shift_evidence_high_for_monotonic_drift(self):
        from stattwin.health.shi import _compute_corr_shift

        values = np.linspace(100, 140, 40)
        baseline_values = values[:20]
        corr = _compute_corr_shift(values, baseline_values, window=10)
        # late-window correlation with a monotonic sensor is high
        assert corr[-1] > 0.5, corr[-1]

    def test_compute_shi_uses_corr_shift_kernel(self, monkeypatch):
        """``compute_shi`` must call ``_compute_corr_shift`` for the corr_shift term."""
        import stattwin.health.shi as shi_mod

        called = {"n": 0}
        original = shi_mod._compute_corr_shift

        def _spy(values, baseline_values, window=10):
            called["n"] += 1
            return original(values, baseline_values, window=window)

        monkeypatch.setattr(shi_mod, "_compute_corr_shift", _spy)
        df = pd.DataFrame(
            {
                "unit_id": np.repeat(1, 40),
                "cycle": np.arange(1, 41),
                "sensor_1": np.linspace(100, 140, 40),
                "sensor_2": np.linspace(200, 240, 40),
                "RUL": np.linspace(40, 1, 40),
            }
        )
        result = shi_mod.compute_shi(
            df, sensor_cols=["sensor_1", "sensor_2"], baseline_cycles=20
        )
        assert called["n"] > 0, "corr_shift kernel never invoked"
        assert "shi" in result.shi_values.columns


# ---------------------------------------------------------------------------
# GRUModel
# ---------------------------------------------------------------------------


class TestGRUModel:
    def test_predict_aligned_to_shuffled_index(self):
        """Regression: predictions were indexed by position, not by row label."""
        df = _panel(n_units=6, n_cycles=70, seed=2)
        model = GRUModel(
            epochs=1, patience=1, batch_size=64, hidden=8, layers=1,
            normalize_features=True, val_fraction=0.2,
            horizons=FAILURE_HORIZONS,
        )
        model.fit(df, _labels(df))
        shuffled = df.sample(frac=1.0, random_state=5)
        rul = model.predict_rul(shuffled)
        assert list(rul.index) == list(shuffled.index)
        # predictions must be a permutation of the ordered predictions
        ordered = model.predict_rul(df)
        key_ord = df[["unit_id", "cycle"]].copy()
        key_ord["p"] = ordered.to_numpy()
        key_shuf = shuffled[["unit_id", "cycle"]].copy()
        key_shuf["p"] = rul.to_numpy()
        merged = key_ord.merge(key_shuf, on=["unit_id", "cycle"], suffixes=("_o", "_s"))
        assert np.allclose(merged["p_o"], merged["p_s"], atol=1e-4)

    def test_rul_targets_scaled_and_predictions_unbounded_below_zero(self):
        df = _panel(n_units=6, n_cycles=70, seed=3)
        model = GRUModel(
            epochs=1, patience=1, batch_size=64, hidden=8, layers=1,
            horizons=FAILURE_HORIZONS,
        )
        model.fit(df, _labels(df))
        assert model._rul_scale > 0
        rul = model.predict_rul(df)
        assert (rul >= 0).all()

    def test_raw_only_restricts_to_raw_sensors(self):
        df = _panel(n_units=4, n_cycles=20, seed=4)
        model = GRUModel(
            epochs=1, patience=1, batch_size=64, hidden=4, layers=1,
            raw_only=True, horizons=FAILURE_HORIZONS,
        )
        model.fit(df, _labels(df))
        assert all(
            c.startswith("sensor_") or c.startswith("op_setting_")
            for c in model._sensor_cols
        )
        assert "rolling_mean_sensor_1_5" not in model._sensor_cols


# ---------------------------------------------------------------------------
# Logistic RUL tail
# ---------------------------------------------------------------------------


class TestLogisticTail:
    def test_rul_can_exceed_last_horizon(self):
        """Regression: derived RUL was capped at the last horizon (50)."""
        df = _panel(n_units=8, n_cycles=70, seed=6)
        model = LogisticModel(horizons=FAILURE_HORIZONS)
        model.fit(df, _labels(df))
        rul = model.predict_rul(df)
        # earliest cycles (long true RUL) should extrapolate past 50
        early = rul[df["cycle"] <= 2]
        assert early.max() > 50.0, early.head()


# ---------------------------------------------------------------------------
# e9 uncertainty helpers
# ---------------------------------------------------------------------------


class TestUncertaintyHelpers:
    def test_ensemble_variance_with_resid_std_achieves_target_coverage(self):
        from stattwin.experiments.e9_uncertainty_comparison import _ensemble_variance

        rng = np.random.default_rng(0)
        truth = rng.normal(50, 15, 4000)
        shared_err = rng.normal(0, 14, 4000)
        members = [truth + shared_err + rng.normal(0, 2, 4000) for _ in range(5)]
        ens = _ensemble_variance(members, 0.10, resid_std=14.0)
        cov = np.mean((truth >= ens["lower"]) & (truth <= ens["upper"]))
        assert 0.85 <= cov <= 0.96, cov

    def test_epistemic_only_undercovers(self):
        from stattwin.experiments.e9_uncertainty_comparison import _ensemble_variance

        rng = np.random.default_rng(1)
        truth = rng.normal(50, 15, 4000)
        shared_err = rng.normal(0, 14, 4000)
        members = [truth + shared_err + rng.normal(0, 2, 4000) for _ in range(5)]
        ens = _ensemble_variance(members, 0.10, resid_std=None)
        cov = np.mean((truth >= ens["lower"]) & (truth <= ens["upper"]))
        assert cov < 0.5, cov  # honest: variance alone cannot cover aleatoric noise

    def test_quantile_offsets_from_calibration(self):
        from stattwin.experiments.e9_uncertainty_comparison import _quantile_regression

        rng = np.random.default_rng(2)
        resid = rng.normal(8, 12, 600)  # residual = true - pred
        q = _quantile_regression(50 + resid, np.full(600, 50.0), 0.10, n_bootstrap=30)
        assert q["lower_offset"] < 0 < q["upper_offset"]
        # applying to a fresh draw gives ~90 % coverage
        test_resid = rng.normal(8, 12, 2000)
        cov = np.mean((test_resid >= q["lower_offset"]) & (test_resid <= q["upper_offset"]))
        assert 0.85 <= cov <= 0.97, cov

    def test_bootstrap_residuals_widen_interval(self):
        from stattwin.experiments.e9_uncertainty_comparison import _bootstrap_pi
        from stattwin.models import XGBoostModel

        rng = np.random.default_rng(3)
        n_units, n_cycles = 6, 70
        rows = []
        for u in range(1, n_units + 1):
            for c in range(1, n_cycles + 1):
                rows.append(
                    {
                        "unit_id": u,
                        "cycle": c,
                        "RUL": float(n_cycles - c),
                        "sensor_1": 100.0 + u + c + rng.normal(0, 2),
                        "sensor_2": 200.0 + c + rng.normal(0, 2),
                    }
                )
        df = pd.DataFrame(rows)
        for h in FAILURE_HORIZONS:
            df[label_col_for(h)] = (df["RUL"] <= h).astype(int)
        y = _labels(df)
        cols = ["sensor_1", "sensor_2"]
        template = XGBoostModel(horizons=FAILURE_HORIZONS, n_estimators=10, max_depth=2)
        test = df.iloc[:30]

        with_resid = _bootstrap_pi(
            template, df, y, test, cols, n_bootstraps=6,
            residuals=rng.normal(0, 10, len(test)),
        )
        without = _bootstrap_pi(template, df, y, test, cols, n_bootstraps=6)

        assert with_resid["residual_augmented"] is True
        assert with_resid["bootstrap_unit"] == "cluster"
        w_with = float(np.mean(with_resid["upper"] - with_resid["lower"]))
        w_without = float(np.mean(without["upper"] - without["lower"]))
        assert w_with > w_without
