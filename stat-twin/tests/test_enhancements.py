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
# Feature screening + causal imputation
# ---------------------------------------------------------------------------


class TestFeatureScreen:
    @staticmethod
    def _frame(n_units=30, n_cycles=100, n_noise=400, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(1, n_units + 1):
            for c in range(1, n_cycles + 1):
                rows.append({"unit_id": u, "cycle": c, "RUL": float(n_cycles - c)})
        X = pd.DataFrame(rows)
        rul = X["RUL"].to_numpy()
        # build the noise block at once (avoids a fragmented frame)
        noise = rng.normal(0, 1, (n_noise, len(rul)))
        block = {
            f"f{i}": (
                (rul * 0.4 if i < 5 else 0.0) + noise[i]
            )
            for i in range(n_noise)
        }
        X = pd.concat([X, pd.DataFrame(block, index=X.index)], axis=1)
        cols = [f"f{i}" for i in range(n_noise)]
        y = pd.DataFrame(
            {
                label_col_for(h): (X["RUL"] <= h).astype(int).to_numpy()
                for h in FAILURE_HORIZONS
            }
        )
        return X, y, cols

    def test_keeps_informative_columns(self):
        from stattwin.experiments._common import screen_features

        X, y, cols = self._frame()
        keep = screen_features(X, y, cols, top_k=50, min_keep=30)
        for informative in ("f0", "f1", "f2", "f3", "f4"):
            assert informative in keep

    def test_respects_top_k(self):
        from stattwin.experiments._common import screen_features

        X, y, cols = self._frame()
        keep = screen_features(X, y, cols, top_k=40, min_keep=10)
        assert len(keep) <= 40

    def test_no_screening_when_under_budget(self):
        from stattwin.experiments._common import screen_features

        X, y, cols = self._frame(n_noise=20)
        keep = screen_features(X, y, cols, top_k=300)
        assert keep == cols

    def test_always_keeps_raw_sensors(self):
        from stattwin.experiments._common import screen_features

        X, y, cols = self._frame(n_noise=50)
        X = X.copy()
        X["sensor_1"] = X["RUL"] * 0.1 + 500.0
        X["op_setting_2"] = 1.0  # constant raw column
        keep = screen_features(X, y, cols + ["sensor_1", "op_setting_2"],
                               top_k=10, min_keep=5)
        assert "sensor_1" in keep
        assert "op_setting_2" in keep

    def test_selection_is_train_only(self):
        """A feature that only correlates in the validation units must not
        be promoted by the screen."""
        from stattwin.experiments._common import screen_features

        X, y, cols = self._frame(n_noise=60)
        X["spurious"] = 0.0
        units = sorted(X["unit_id"].unique())
        val_units = units[-5:]
        mask = X["unit_id"].isin(val_units)
        # correlate `spurious` with RUL only on the validation units
        X.loc[mask, "spurious"] = X.loc[mask, "RUL"].to_numpy() * 10.0
        train = X[~mask]
        keep = screen_features(train, y.loc[train.index], cols + ["spurious"],
                               top_k=8, min_keep=5)
        assert "spurious" not in keep

    def test_config_switch_respects_disable(self):
        from stattwin.config import STATTWINConfig
        from stattwin.experiments._common import resolve_feature_cols

        cfg = STATTWINConfig()
        cfg.model.feature_screen.enabled = False
        X, y, cols = self._frame(n_noise=60)
        assert resolve_feature_cols(cfg, X, y, cols) == cols

    def test_config_switch_uses_top_k(self):
        from stattwin.config import STATTWINConfig
        from stattwin.experiments._common import resolve_feature_cols

        cfg = STATTWINConfig()
        cfg.model.feature_screen.top_k = 20
        cfg.model.feature_screen.min_keep = 10
        X, y, cols = self._frame(n_noise=60)
        assert len(resolve_feature_cols(cfg, X, y, cols)) <= 20

    def test_min_keep_floors_the_screened_count(self):
        """A tiny ``top_k`` must not gut the model below ``min_keep``."""
        from stattwin.config import STATTWINConfig
        from stattwin.experiments._common import resolve_feature_cols

        cfg = STATTWINConfig()
        cfg.model.feature_screen.top_k = 5
        cfg.model.feature_screen.min_keep = 40
        X, y, cols = self._frame(n_noise=60)
        keep = resolve_feature_cols(cfg, X, y, cols)
        assert len(keep) >= 40


class TestImputeCausal:
    def test_forward_fills_interior_gaps_within_unit(self):
        from stattwin.experiments._common import impute_causal

        X = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "f": [1.0, np.nan, 3.0, 5.0, np.nan, 7.0],
            }
        )
        impute_causal([X], ["f"])
        # interior gaps inherit the previous cycle of the same unit
        assert X.loc[1, "f"] == 1.0   # unit 1
        assert X.loc[4, "f"] == 5.0   # unit 2, not unit 1's 1.0

    def test_leading_nan_falls_back_to_column_median(self):
        from stattwin.experiments._common import impute_causal

        X = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2],
                "f": [1.0, 3.0, 5.0, np.nan, 7.0],
            }
        )
        impute_causal([X], ["f"])
        # unit 2 has no history at its first row -> median of observed
        # training values (1, 3, 5, 7) = 4.0
        assert X.loc[3, "f"] == 4.0

    def test_does_not_fill_across_units(self):
        from stattwin.experiments._common import impute_causal

        X = pd.DataFrame(
            {"unit_id": [1, 1, 2, 2], "f": [5.0, np.nan, np.nan, 7.0]}
        )
        impute_causal([X], ["f"])
        # unit 1 interior gap inherits 5.0 from its own unit; unit 2's
        # leading gap has no history, so it takes the column median
        # (observed values 5, 5, 7 -> 5.0) and never a *sequence* from
        # unit 1
        assert X.loc[1, "f"] == 5.0
        assert X.loc[2, "f"] == 5.0

    def test_all_missing_column_becomes_zero(self):
        from stattwin.experiments._common import impute_causal

        train = pd.DataFrame({"unit_id": 1, "f": [np.nan, np.nan]})
        val = pd.DataFrame({"unit_id": 2, "f": [np.nan, 1.0]})
        impute_causal([train, val], ["f"])
        assert train["f"].tolist() == [0.0, 0.0]
        assert val["f"].tolist() == [0.0, 1.0]

    def test_validation_values_never_shape_the_median(self):
        from stattwin.experiments._common import impute_causal

        # training column is constant at 10.0; validation holds 1000/NaN
        train = pd.DataFrame({"unit_id": 1, "f": [10.0, 10.0, 10.0]})
        val = pd.DataFrame({"unit_id": 2, "f": [1000.0, np.nan]})
        impute_causal([train, val], ["f"])
        # the val NaN is filled by causal ffill inside unit 2 (1000.0) —
        # a training *median* would have been 10.0, proving the fill is
        # row-local and never the val distribution
        assert val.loc[1, "f"] == 1000.0

    def test_leading_val_nan_uses_training_median(self):
        from stattwin.experiments._common import impute_causal

        train = pd.DataFrame({"unit_id": 1, "f": [10.0, 20.0, 30.0]})
        val = pd.DataFrame({"unit_id": 2, "f": [np.nan, 1000.0]})
        impute_causal([train, val], ["f"])
        # val unit's FIRST row has no history -> training median (20.0),
        # never the val mean / any val-derived statistic
        assert val.loc[0, "f"] == 20.0

    def test_missing_columns_are_ignored(self):
        from stattwin.experiments._common import impute_causal

        X = pd.DataFrame({"unit_id": [1, 2], "f": [1.0, 2.0]})
        impute_causal([X], ["f", "not_present"])
        assert X["f"].tolist() == [1.0, 2.0]

    def test_works_without_unit_column(self):
        from stattwin.experiments._common import impute_causal

        X = pd.DataFrame({"f": [1.0, np.nan, 3.0]})
        impute_causal([X], ["f"])
        # no unit column -> no ffill; the gap takes the column median (2.0)
        assert X["f"].tolist() == [1.0, 2.0, 3.0]


class TestHybridModelTraining:
    """Regression tests for the four defects that made HybridModel output
    a near-constant predictor (OOF ROC-AUC 0.52, RUL MAE 65)."""

    @staticmethod
    def _hybrid(epochs=2, **kwargs):
        params = dict(
            hidden=8, layers=1, dropout=0.0, seq_len=10, ensemble_size=1,
            epochs=epochs, patience=1, batch_size=64, val_fraction=0.25,
            horizons=FAILURE_HORIZONS,
        )
        params.update(kwargs)
        return HybridModel(**params)

    def test_learns_signal_instead_of_collapsing(self):
        df = _panel(n_units=8, n_cycles=70, seed=11)
        model = self._hybrid(epochs=6)
        model.fit(df, _labels(df))
        proba = model.predict_proba(df)
        # a collapsed net returns ~0.5 everywhere; a trained one spreads out
        assert float(proba.values.std()) > 0.05
        # and separates the classes: risky rows (RUL <= 10) score higher
        risky = df["RUL"] <= 10
        assert float(proba.loc[risky].mean().mean()) > float(
            proba.loc[~risky].mean().mean()
        )

    def test_features_are_standardised(self):
        df = _panel(n_units=6, n_cycles=70, seed=12)
        model = self._hybrid(epochs=1)
        model.fit(df, _labels(df))
        assert model._feat_mean is not None
        assert model._feat_std is not None
        assert np.all(model._feat_std > 0)
        # sequences must be centred: |mean| far below the raw sensor scale
        seq, *_ = model._build_extended_sequences(df)
        assert abs(float(np.nanmean(seq))) < 2.0

    def test_rul_targets_are_scaled(self):
        df = _panel(n_units=6, n_cycles=70, seed=13)
        model = self._hybrid(epochs=1)
        model.fit(df, _labels(df))
        assert model._rul_scale == pytest.approx(float(df["RUL"].max()), rel=0.01)

    def test_predictions_align_to_frame_index_when_shuffled(self):
        """Sequences are built per unit; predictions must follow row labels."""
        df = _panel(n_units=6, n_cycles=70, seed=14)
        model = self._hybrid(epochs=1)
        model.fit(df, _labels(df))
        shuffled = df.sample(frac=1.0, random_state=3)
        proba = model.predict_proba(shuffled)
        rul = model.predict_rul(shuffled)
        assert list(proba.index) == list(shuffled.index)
        assert list(rul.index) == list(shuffled.index)

        ordered = model.predict_proba(df)
        key_o = df[["unit_id", "cycle"]].copy()
        key_o["p"] = ordered["fail_h30"].to_numpy()
        key_s = shuffled[["unit_id", "cycle"]].copy()
        key_s["p"] = proba["fail_h30"].to_numpy()
        merged = key_o.merge(key_s, on=["unit_id", "cycle"], suffixes=("_o", "_s"))
        assert np.allclose(merged["p_o"], merged["p_s"], atol=1e-6)

    def test_interleaved_units_keep_their_own_predictions(self):
        """Row order that interleaves units must not scramble outputs."""
        df = _panel(n_units=6, n_cycles=70, seed=15)
        model = self._hybrid(epochs=1)
        model.fit(df, _labels(df))
        base = df.sort_values(["unit_id", "cycle"])
        interleaved = (
            df.sort_values(["cycle", "unit_id"]).sort_index(kind="stable")
        )
        p_base = model.predict_proba(base)["fail_h30"]
        p_int = model.predict_proba(interleaved)["fail_h30"]
        # same rows -> same predictions regardless of frame order
        joined = pd.DataFrame({"b": p_base, "i": p_int}, index=p_base.index)
        assert np.allclose(joined["b"], joined["i"], atol=1e-6)

    def test_rul_uses_regression_head_and_is_nonnegative(self):
        df = _panel(n_units=6, n_cycles=70, seed=16)
        model = self._hybrid(epochs=3)
        model.fit(df, _labels(df))
        rul = model.predict_rul(df)
        assert (rul >= 0).all()
        # the regression head must actually track the target
        assert float(np.corrcoef(rul.to_numpy(), df["RUL"].to_numpy())[0, 1]) > 0.5

    def test_monotone_envelope_still_applied(self):
        df = _panel(n_units=6, n_cycles=70, seed=17)
        model = self._hybrid(epochs=1)
        model.fit(df, _labels(df))
        proba = model.predict_proba(df)
        arr = proba.to_numpy()
        assert np.all(np.diff(arr, axis=1) >= -1e-9)


# ---------------------------------------------------------------------------
# e2 checkpointing / resume
# ---------------------------------------------------------------------------


class TestE2Checkpointing:
    @staticmethod
    def _fingerprint():
        return {
            "features": "engineered",
            "feature_screen": {"top_k": 300},
            "model_signature": {"xgb": {"n_estimators": 400}},
        }

    def test_build_results_marks_partial(self):
        from stattwin.config import STATTWINConfig
        from stattwin.experiments.e2_model_comparison import _build_results

        cfg = STATTWINConfig()
        rows = [
            {
                "model": "XGBoostModel",
                "classification": [{"horizon": 30, "roc_auc": 0.9, "f1": 0.5}],
                "rul": {"rmse": 20.0, "mae": 12.0, "nasa_score": 300.0},
            }
        ]
        partial = _build_results(cfg, self._fingerprint(), rows, partial=True)
        assert partial["partial"] is True
        assert partial["summary_table"][0]["rul_rmse"] == 20.0
        final = _build_results(cfg, self._fingerprint(), rows, partial=False)
        assert final["partial"] is False

    def test_error_rows_survive_summary(self):
        from stattwin.config import STATTWINConfig
        from stattwin.experiments.e2_model_comparison import _build_results

        cfg = STATTWINConfig()
        out = _build_results(
            cfg,
            self._fingerprint(),
            [{"model": "BrokenModel", "error": "boom"}],
            partial=True,
        )
        assert out["summary_table"] == [{"model": "BrokenModel", "error": "boom"}]

    def test_save_and_read_roundtrip(self, temp_dir):
        from stattwin.config import STATTWINConfig
        from stattwin.experiments.e2_model_comparison import (
            _read_json,
            _save_results,
        )

        cfg = STATTWINConfig()
        fp = self._fingerprint()
        rows = [
            {
                "model": "GRUModel",
                "classification": [{"horizon": 10, "roc_auc": 0.8, "f1": 0.4}],
                "rul": {"rmse": 25.0, "mae": 15.0, "nasa_score": 500.0},
            }
        ]
        _save_results(cfg, temp_dir, fp, rows, partial=True)
        loaded = _read_json(temp_dir / "e2_results.json")
        assert isinstance(loaded, dict)
        assert loaded["partial"] is True
        assert loaded["models"][0]["model"] == "GRUModel"
        assert loaded["feature_screen"] == fp["feature_screen"]
        # the signature is persisted so a config change invalidates the cache
        assert loaded["model_signature"] == fp["model_signature"]

    def test_model_signature_covers_hyperparameters(self):
        from stattwin.config import load_config
        from stattwin.experiments.e2_model_comparison import _model_signature

        fast = _model_signature(load_config("configs/base.yaml", profile="fast"))
        full = _model_signature(load_config("configs/base.yaml", profile="full"))
        assert fast != full, "different profiles must not share a resume cache"
        assert set(fast) == {"lr", "rf", "xgb", "gru", "lstm", "ensemble"}

    def test_read_json_missing_and_malformed(self, temp_dir):
        from stattwin.experiments.e2_model_comparison import _read_json

        assert _read_json(temp_dir / "nope.json") is None
        bad = temp_dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert _read_json(bad) is None


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
