"""Model 8: Heterogeneous soft-vote ensemble.

Combines per-horizon classifiers and RUL regressors from several base
learners (logistic, random forest, XGBoost, …) into a single model with
explicit combination weights.

Two weight modes:

* ``"uniform"`` – equal weights over all members.
* ``"learned"``  – members are fitted on an inner training split; weights
  are optimised on a held-out inner validation split (unit-disjoint) by
  minimising mean binary log-loss across failure horizons for the
  probability heads, and mean absolute error for the RUL head.  Members
  are then refitted on the full training set with the frozen weights.

The ensemble exposes ``predict_uncertainty`` = disagreement (standard
deviation) across members, which downstream uncertainty code can use as
a genuine epistemic signal.

References
----------
* Wolpert (1992) – stacked generalisation.
* STAT-TWIN masterplan, Section 6.8.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from stattwin.data.schema import label_col_for
from stattwin.models.base import BaseModel

__all__ = ["EnsembleModel", "build_member"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


def build_member(name: str, cfg: Any, horizons: Sequence[int]) -> BaseModel:
    """Construct a fresh base learner by name from a ``ModelCfg``.

    Parameters
    ----------
    name:
        One of ``logistic``, ``random_forest``, ``xgboost``.
    cfg:
        The full ``STATTWINConfig`` (only ``cfg.model`` is read).
    horizons:
        Failure horizons passed to the constructor.
    """
    from stattwin.models.logistic import LogisticModel
    from stattwin.models.random_forest import RandomForestModel
    from stattwin.models.xgboost_model import XGBoostModel

    factories: dict[str, Callable[[], BaseModel]] = {
        "logistic": lambda: LogisticModel(
            C=cfg.model.lr.C,
            max_iter=cfg.model.lr.max_iter,
            horizons=list(horizons),
        ),
        "random_forest": lambda: RandomForestModel(
            n_estimators=cfg.model.rf.n_estimators,
            max_depth=cfg.model.rf.max_depth,
            horizons=list(horizons),
        ),
        "xgboost": lambda: XGBoostModel(
            n_estimators=cfg.model.xgb.n_estimators,
            max_depth=cfg.model.xgb.max_depth,
            learning_rate=cfg.model.xgb.learning_rate,
            horizons=list(horizons),
        ),
    }
    if name not in factories:
        raise ValueError(
            f"Unknown ensemble member '{name}'. Available: {sorted(factories)}"
        )
    return factories[name]()


class EnsembleModel(BaseModel):
    """Weighted soft-vote ensemble over heterogeneous base learners.

    Parameters
    ----------
    members:
        List of ``(name, unfitted_model)`` pairs.  If *None*, a caller
        must use :meth:`set_members` before :meth:`fit`.
    weight_mode:
        ``"learned"`` (default) fits members on an inner unit-disjoint
        split, optimises weights on the held-out inner validation units,
        then refits on the full training set.  ``"uniform"`` skips the
        inner split and uses equal weights.
    val_fraction:
        Fraction of training units held out for weight learning
        (only used when ``weight_mode="learned"``).
    random_state:
        Seed for the inner split.
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        members: list[tuple[str, BaseModel]] | None = None,
        weight_mode: Literal["learned", "uniform"] = "learned",
        val_fraction: float = 0.2,
        random_state: int = 42,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="EnsembleModel")
        self.members = members or []
        self.weight_mode = weight_mode
        self.val_fraction = val_fraction
        self.random_state = random_state

        self.prob_weights_: dict[str, float] = {}
        self.rul_weights_: dict[str, float] = {}
        self._fitted_members: list[tuple[str, BaseModel]] = []
        self._weight_diagnostics_: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_members(self, members: list[tuple[str, BaseModel]]) -> None:
        """Attach unfitted member models before calling :meth:`fit`."""
        self.members = members

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - x.max()
        e = np.exp(z)
        return e / e.sum()

    def _inner_split(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit-disjoint inner train/val index arrays."""
        units = np.array(sorted(X[_UNIT_COL].unique()))
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(units)
        n_val = max(1, int(round(self.val_fraction * len(units))))
        val_units = set(units[:n_val].tolist())
        val_mask = X[_UNIT_COL].isin(val_units).to_numpy()
        return X.index[~val_mask], X.index[val_mask]

    def _learn_weights(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        """Fit members on inner-train, score on inner-val, optimise weights.

        Returns ``(prob_weights, rul_weights, diagnostics)``.
        """
        train_idx, val_idx = self._inner_split(X)
        X_tr, X_va = X.loc[train_idx], X.loc[val_idx]
        y_tr = y.loc[train_idx]
        if _UNIT_COL not in X_va.columns and _UNIT_COL in X.columns:
            raise RuntimeError("unit_id missing from feature frame")

        # Fit scratch members on inner-train
        scratch: list[tuple[str, BaseModel]] = []
        for name, factory in self.members:
            m = copy.deepcopy(factory)
            fit_cols = [c for c in X_tr.columns]
            m.fit(X_tr[fit_cols], y_tr)
            scratch.append((name, m))

        proba_va = {n: m.predict_proba(X_va) for n, m in scratch}
        rul_va = {n: m.predict_rul(X_va) for n, m in scratch}
        rul_true = (
            X_va["RUL"].to_numpy(dtype=float)
            if "RUL" in X_va.columns
            else None
        )

        # --- probability weights: minimise mean log-loss over horizons ----
        label_cols = [label_col_for(h) for h in self.horizons if label_col_for(h) in y.columns]
        y_va = y.loc[val_idx, label_cols] if label_cols else None

        n_m = len(scratch)
        if y_va is not None and n_m > 0:
            names = [n for n, _ in scratch]
            # Coordinate ascent on a simplex: start uniform, greedy refine.
            best = np.full(n_m, 1.0 / n_m)

            def _logloss(w: np.ndarray) -> float:
                acc = np.zeros(len(y_va))
                for wi, name in zip(w, names, strict=True):
                    p = proba_va[name][label_cols].to_numpy()
                    acc = acc + wi * p
                acc = np.clip(acc, 1e-9, 1 - 1e-9)
                yt = y_va.to_numpy(dtype=float)
                return float(-np.mean(yt * np.log(acc) + (1 - yt) * np.log(1 - acc)))

            best_ll = _logloss(best)
            rng = np.random.default_rng(self.random_state)
            for _ in range(60):
                cand = self._softmax(rng.normal(0, 0.35, size=n_m))
                ll = _logloss(cand)
                if ll < best_ll:
                    best_ll = ll
                    best = cand
            prob_w = {n: float(w) for n, w in zip(names, best, strict=True)}
        else:
            prob_w = {n: 1.0 / max(n_m, 1) for n, _ in scratch}
            best_ll = float("nan")

        # --- RUL weights: minimise MAE on inner-val -----------------------
        if rul_true is not None and n_m > 0:
            names = [n for n, _ in scratch]
            preds = np.column_stack(
                [rul_va[n].reindex(X_va.index).to_numpy(dtype=float) for n in names]
            )
            best_rw = np.full(n_m, 1.0 / n_m)
            best_mae = float(
                np.mean(np.abs(rul_true - preds @ best_rw))
            )
            rng = np.random.default_rng(self.random_state + 1)
            for _ in range(60):
                cand = self._softmax(rng.normal(0, 0.35, size=n_m))
                mae = float(np.mean(np.abs(rul_true - preds @ cand)))
                if mae < best_mae:
                    best_mae = mae
                    best_rw = cand
            rul_w = {n: float(w) for n, w in zip(names, best_rw, strict=True)}
        else:
            rul_w = {n: 1.0 / max(n_m, 1) for n, _ in scratch}
            best_mae = float("nan")

        diag = {
            "inner_logloss": None if np.isnan(best_ll) else float(best_ll),
            "inner_rul_mae": None if np.isnan(best_mae) else float(best_mae),
            "n_inner_train_units": int(X_tr[_UNIT_COL].nunique()),
            "n_inner_val_units": int(X_va[_UNIT_COL].nunique()),
        }
        return prob_w, rul_w, diag

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> EnsembleModel:
        """Fit every member and (optionally) learn combination weights."""
        if not self.members:
            raise RuntimeError("EnsembleModel has no members; call set_members().")

        if self.weight_mode == "learned" and len(self.members) > 1:
            self.prob_weights_, self.rul_weights_, self._weight_diagnostics_ = (
                self._learn_weights(X_train, y_train)
            )
        else:
            w = 1.0 / len(self.members)
            self.prob_weights_ = {n: w for n, _ in self.members}
            self.rul_weights_ = {n: w for n, _ in self.members}
            self._weight_diagnostics_ = {"mode": "uniform"}

        # Refit every member on the full training set with frozen weights
        self._fitted_members = []
        for name, factory in self.members:
            m = copy.deepcopy(factory)
            m.fit(X_train, y_train)
            self._fitted_members.append((name, m))

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Weighted average of member per-horizon probabilities."""
        if not self._fitted_members:
            raise RuntimeError("Model not fitted.")

        total_w = sum(self.prob_weights_.get(n, 0.0) for n, _ in self._fitted_members)
        if total_w <= 0:
            total_w = float(len(self._fitted_members))
            self.prob_weights_ = {
                n: 1.0 for n, _ in self._fitted_members
            }

        acc: pd.DataFrame | None = None
        for name, m in self._fitted_members:
            w = self.prob_weights_.get(name, 0.0) / total_w
            p = m.predict_proba(X)
            acc = p * w if acc is None else acc + p * w
        assert acc is not None
        return acc.clip(0.0, 1.0)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Weighted average of member RUL predictions."""
        if not self._fitted_members:
            raise RuntimeError("Model not fitted.")

        total_w = sum(self.rul_weights_.get(n, 0.0) for n, _ in self._fitted_members)
        if total_w <= 0:
            total_w = float(len(self._fitted_members))
            self.rul_weights_ = {n: 1.0 for n, _ in self._fitted_members}

        acc = pd.Series(0.0, index=X.index, name="RUL")
        for name, m in self._fitted_members:
            w = self.rul_weights_.get(name, 0.0) / total_w
            acc = acc + w * m.predict_rul(X).reindex(X.index)
        return acc.clip(lower=0.0).rename("RUL")

    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Max of the weighted per-horizon probabilities."""
        proba = self.predict_proba(X)
        return proba.max(axis=1).rename("raw_score")

    def predict_uncertainty(self, X: pd.DataFrame) -> pd.Series:
        """Disagreement across members (std of probabilities and RUL).

        Returned value is the mean over horizons of the member-probability
        standard deviation, blended with the normalised RUL disagreement.
        """
        if not self._fitted_members:
            raise RuntimeError("Model not fitted.")

        probas = [m.predict_proba(X) for _, m in self._fitted_members]
        stacked = np.stack([p.to_numpy(dtype=float) for p in probas], axis=0)
        prob_std = stacked.std(axis=0).mean(axis=1)  # (N,)

        ruls = np.stack(
            [m.predict_rul(X).reindex(X.index).to_numpy(dtype=float)
             for _, m in self._fitted_members],
            axis=0,
        )
        rul_std = ruls.std(axis=0)
        rul_scale = max(float(np.mean(ruls)), 1.0)
        blended = 0.5 * prob_std + 0.5 * (rul_std / rul_scale)
        return pd.Series(blended, index=X.index, name="uncertainty")
