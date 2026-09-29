"""Model 7: STAT-TWIN Hybrid – GRU ensemble with health features.

The flagship model combines:

1. **GRU backbone** – processes raw sensor + statistical + health features
   as a sequence over the last ``seq_len`` cycles.
2. **Ensemble of M seeds** – independent GRU training runs with different
   random initialisations; mean prediction is the point estimate, std
   feeds the uncertainty normaliser.
3. **Monotone enforcement** – predicted probabilities are reordered so
   that ``P(fail_h10) <= P(fail_h20) <= ... <= P(fail_h50)``.
4. **Tabular fallback** – XGBoost on raw + statistical + temporal + health
   features when sequences are unavailable.

The ensemble mean is the prediction and ensemble std is the uncertainty
normaliser (higher std → less confident).

References
----------
*  STAT-TWIN masterplan, Section 6.7.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.models.base import BaseModel
from stattwin.models.gru import _GRUNetwork, _LSTMNetwork, _SequenceDataset

__all__ = ["HybridModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# -----------------------------------------------------------------------
# Tabular fallback: XGBoost on extended features
# -----------------------------------------------------------------------


class _TabularFallback:
    """XGBoost per-horizon classifier + RUL regressor on extended features."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._classifiers: dict[int, Any] = {}
        self._regressor: Any = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        rul: pd.Series | None = None,
    ) -> None:
        from xgboost import XGBClassifier, XGBRegressor

        self._feature_cols = [
            c for c in X.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X[c])
        ]
        X_arr = X[self._feature_cols].to_numpy(dtype=np.float64)

        for h in FAILURE_HORIZONS:
            col = label_col_for(h)
            if col not in y.columns:
                continue
            y_arr = y[col].to_numpy().astype(np.int32)
            n_pos = y_arr.sum()
            n_neg = len(y_arr) - n_pos
            spw = n_neg / max(n_pos, 1.0)

            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=min(spw, 50.0),
                random_state=self.random_state,
                eval_metric="aucpr",
                n_jobs=-1,
            )
            clf.fit(X_arr, y_arr)
            self._classifiers[h] = clf

        if rul is not None:
            self._regressor = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._regressor.fit(X_arr, rul.to_numpy(dtype=np.float64))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_arr = X[self._feature_cols].to_numpy(dtype=np.float64)
        proba_dict: dict[str, np.ndarray] = {}
        for h in FAILURE_HORIZONS:
            clf = self._classifiers.get(h)
            if clf is not None:
                proba_dict[label_col_for(h)] = clf.predict_proba(X_arr)[:, 1]
            else:
                proba_dict[label_col_for(h)] = np.full(len(X), 0.5)
        return pd.DataFrame(proba_dict, index=X.index)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        if self._regressor is not None:
            X_arr = X[self._feature_cols].to_numpy(dtype=np.float64)
            return pd.Series(
                np.maximum(self._regressor.predict(X_arr), 0.0),
                index=X.index,
                name="RUL",
            )
        return pd.Series(np.full(len(X), 60.0), index=X.index, name="RUL")


# -----------------------------------------------------------------------
# Monotone enforcement
# -----------------------------------------------------------------------


def _enforce_monotone(proba: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Enforce P(fail_h10) <= P(fail_h20) <= ... <= P(fail_h50) per row.

    Implemented as a vectorised cumulative maximum from left to right
    (lowest horizon first), which guarantees a non-decreasing sequence
    without collapsing rows to a constant.
    """
    cols = [label_col_for(h) for h in horizons]
    out = proba[cols].to_numpy(dtype=np.float64)
    out = np.maximum.accumulate(out, axis=1)
    return pd.DataFrame(out, columns=cols, index=proba.index)


# -----------------------------------------------------------------------
# Main hybrid model
# -----------------------------------------------------------------------


class HybridModel(BaseModel):
    """STAT-TWIN Hybrid: GRU ensemble with health features + tabular fallback.

    Parameters
    ----------
    backbone:
        ``"gru"`` or ``"lstm"`` (default ``"gru"``).
    hidden:
        GRU/LSTM hidden size (default 64).
    layers:
        Number of RNN layers (default 2).
    dropout:
        Dropout rate (default 0.2).
    seq_len:
        Sequence length in cycles (default 30).
    ensemble_size:
        Number of independently seeded GRU runs (default 3).
    enforce_monotone:
        Whether to enforce monotonic horizon probabilities (default True).
    use_tabular_fallback:
        Whether to use XGBoost on all features as fallback (default True).
    lr:
        Learning rate (default 1e-3).
    batch_size:
        Mini-batch size (default 64).
    epochs:
        Maximum training epochs (default 50).
    patience:
        Early-stopping patience (default 8).
    device:
``"cpu"`` or ``"cuda"``.
    horizons:
        Failure horizons.
    """

    def __init__(
        self,
        backbone: Literal["gru", "lstm"] = "gru",
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 30,
        ensemble_size: int = 3,
        enforce_monotone: bool = True,
        use_tabular_fallback: bool = True,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 8,
        val_fraction: float = 0.15,
        device: str | None = None,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name="HybridModel")
        self.backbone = backbone
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.ensemble_size = ensemble_size
        self.enforce_monotone = enforce_monotone
        self.use_tabular_fallback = use_tabular_fallback
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._ensemble: list[_GRUNetwork | _LSTMNetwork] = []
        self._tabular: _TabularFallback | None = None
        self._sensor_cols: list[str] = []
        self._stat_cols: list[str] = []
        self._health_cols: list[str] = []
        self._all_feature_cols: list[str] = []
        # fitted standardisation + RUL scale (set in fit)
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._rul_scale: float = 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """Separate raw sensor, statistical, and health feature columns."""
        exclude = {_UNIT_COL, _CYCLE_COL, "RUL"}
        all_numeric = [
            c for c in X.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(X[c])
        ]

        sensor_cols = [c for c in all_numeric if c.startswith("sensor_") or c.startswith("op_setting_")]  # noqa: E501
        health_cols = [c for c in all_numeric if "shi" in c.lower() or "health" in c.lower() or "evidence_" in c.lower()]  # noqa: E501
        stat_cols = [c for c in all_numeric if c not in sensor_cols and c not in health_cols]

        # If no health/stat columns detected, fall back to all numeric
        if not stat_cols and not health_cols:
            sensor_cols = all_numeric

        return sensor_cols, stat_cols, health_cols

    def _build_extended_sequences(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build standardised sequences from sensor + stat + health columns.

        Inputs are z-scored with statistics captured on the training frame
        (see :meth:`fit`); rows with missing values are zeroed *after*
        standardisation, which maps them to the column mean rather than
        to an arbitrary constant.
        """
        all_cols = self._all_feature_cols
        block = X[all_cols].to_numpy(dtype=np.float64)
        if self._feat_mean is not None and self._feat_std is not None:
            block = (block - self._feat_mean) / self._feat_std
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

        all_units = X[_UNIT_COL].unique()
        X_list: list[np.ndarray] = []
        idx_list: list[int] = []
        mask_list: list[np.ndarray] = []

        for unit in all_units:
            unit_df = X[X[_UNIT_COL] == unit].sort_values(_CYCLE_COL)
            rows = unit_df.index.to_numpy()
            vals = X.reindex(rows)[all_cols].to_numpy(dtype=np.float64)
            if self._feat_mean is not None and self._feat_std is not None:
                vals = (vals - self._feat_mean) / self._feat_std
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            if len(vals) == 0:
                continue
            # pad short sequences with THIS unit's earliest observed state,
            # so a row's prediction never depends on the caller's row order
            first_row = vals[0:1, :]

            for t in range(len(vals)):
                start = max(0, t - self.seq_len + 1)
                seq = vals[start : t + 1, :]
                actual_len = seq.shape[0]

                if actual_len < self.seq_len:
                    pad = np.tile(first_row, (self.seq_len - actual_len, 1))
                    seq = np.concatenate([pad, seq], axis=0)
                    mask = np.concatenate(
                        [np.zeros(self.seq_len - actual_len), np.ones(actual_len)]
                    )
                else:
                    mask = np.ones(self.seq_len)

                X_list.append(seq)
                idx_list.append(rows[t])
                mask_list.append(mask)

        return (
            np.array(X_list, dtype=np.float64),
            np.array(idx_list, dtype=np.int64),
            np.array(mask_list, dtype=np.float64),
            np.array(first_row, dtype=np.float64),
        )

    def _create_network(self, input_size: int) -> _GRUNetwork | _LSTMNetwork:
        if self.backbone == "lstm":
            return _LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden,
                num_layers=self.layers,
                dropout=self.dropout,
                num_horizons=len(self.horizons),
            )
        return _GRUNetwork(
            input_size=input_size,
            hidden_size=self.hidden,
            num_layers=self.layers,
            dropout=self.dropout,
            num_horizons=len(self.horizons),
        )

    def _train_single_seed(
        self,
        X_seq: np.ndarray,
        y_labels: np.ndarray,
        y_rul: np.ndarray,
        seed: int,
        is_val: np.ndarray | None = None,
    ) -> _GRUNetwork | _LSTMNetwork:
        """Train one GRU instance, early-stopping on held-out units.

        When *is_val* marks a unit-disjoint inner validation split, the
        reported loss and the early-stopping decision come from those
        rows only; otherwise (or with an empty split) the training rows
        are used and the caller gets the old in-sample behaviour.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        net = self._create_network(X_seq.shape[2]).to(self.device)
        is_val = np.zeros(len(X_seq), dtype=bool) if is_val is None else is_val
        has_val = bool(is_val.any()) and bool((~is_val).any())

        train_ds = _SequenceDataset(
            X_seq[~is_val], y_labels[~is_val], y_rul[~is_val]
        )
        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        if has_val:
            val_loader = DataLoader(
                _SequenceDataset(X_seq[is_val], y_labels[is_val], y_rul[is_val]),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = loader

        pos_weights = []
        for i in range(y_labels.shape[1]):
            n_pos = y_labels[~is_val, i].sum()
            n_neg = int((~is_val).sum()) - n_pos
            pw = n_neg / max(n_pos, 1.0)
            pos_weights.append(min(pw, 50.0))
        pos_weight = torch.tensor(pos_weights, dtype=torch.float32, device=self.device)

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for _epoch in range(self.epochs):
            net.train()
            for X_b, y_b, r_b in loader:
                X_b, y_b, r_b = X_b.to(self.device), y_b.to(self.device), r_b.to(self.device)
                optimizer.zero_grad()
                h_logits, r_pred = net(X_b)
                loss = self.horizon_weight * bce_loss(h_logits, y_b) + mse_loss(r_pred, r_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            # Evaluate on the held-out units (never on the training rows
            # when an inner split exists)
            net.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_b, y_b, r_b in val_loader:
                    X_b, y_b, r_b = (
                        X_b.to(self.device),
                        y_b.to(self.device),
                        r_b.to(self.device),
                    )
                    h_logits, r_pred = net(X_b)
                    batch_loss = (
                        bce_loss(h_logits, y_b).item() + mse_loss(r_pred, r_b).item()
                    )
                    val_loss += batch_loss
                    n_val += 1
            val_loss /= max(n_val, 1)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        return net.to(self.device)

    @property
    def horizon_weight(self) -> float:
        return 1.0

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> HybridModel:
        """Fit the hybrid model (GRU ensemble + optional tabular fallback).

        The sequence branch standardises its inputs and scales the RUL
        target, and early-stops on a **unit-disjoint inner split** — the
        same protocol as :class:`~stattwin.models.gru.GRUModel`.  Without
        those three steps the network sees unscaled sensors (magnitude
        ~500) next to operating settings (0/1) and an RUL target in
        cycles, whose MSE swamps the horizon BCE term, so it collapses to
        a near-constant predictor.

        Parameters
        ----------
        X_train:
            Feature matrix.
        y_train:
            Binary label DataFrame.
        groups:
            Ignored (the inner split is derived from ``unit_id``).
        """
        self._sensor_cols, self._stat_cols, self._health_cols = self._select_features(X_train)
        self._all_feature_cols = self._sensor_cols + self._stat_cols + self._health_cols

        if not self._all_feature_cols:
            raise ValueError("No numeric features found in X_train.")

        # Merge labels
        df = X_train.copy()
        for col in self.label_cols:
            if col not in df.columns and col in y_train.columns:
                df[col] = y_train[col].values

        # --- feature standardisation (fit on training rows only) ----------
        block = X_train[self._all_feature_cols]
        self._feat_mean = block.mean(axis=0).to_numpy(dtype=np.float64)
        std = block.std(axis=0).to_numpy(dtype=np.float64)
        self._feat_std = np.where(std < 1e-8, 1.0, std)

        # --- RUL target scale -------------------------------------------
        if "RUL" in X_train.columns:
            rul_vals = X_train["RUL"].to_numpy(dtype=np.float64)
            self._rul_scale = max(float(np.nanmax(rul_vals)), 1.0)
        else:
            self._rul_scale = 1.0

        # Build sequences
        X_seq, row_idx, mask, first_row = self._build_extended_sequences(df)
        y_labels = np.zeros((len(X_seq), len(self.horizons)), dtype=np.float64)
        y_rul = np.full(len(X_seq), np.nan, dtype=np.float64)

        # Fill labels/RUL from merged df (vectorised, no per-row loop)
        label_block = df[self.label_cols].reindex(row_idx)
        y_labels = label_block.to_numpy(dtype=np.float64)
        if "RUL" in df.columns:
            y_rul = df["RUL"].reindex(row_idx).to_numpy(dtype=np.float64)
        y_rul = np.where(np.isnan(y_rul), np.nanmean(y_rul) if np.any(
            ~np.isnan(y_rul)) else 60.0, y_rul)
        y_rul_scaled = y_rul / self._rul_scale

        # --- unit-disjoint inner split for early stopping -----------------
        unit_of_row = df[_UNIT_COL].reindex(row_idx).to_numpy()
        is_val = np.zeros(len(row_idx), dtype=bool)
        units = np.array(sorted(df[_UNIT_COL].unique()))
        if self.val_fraction > 0 and len(units) >= 4:
            n_val_units = max(1, int(round(self.val_fraction * len(units))))
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(units))
            is_val = np.isin(unit_of_row, units[perm[:n_val_units]])

        # Train ensemble of M seeds
        self._ensemble = []
        base_seed = 42
        for m in range(self.ensemble_size):
            net = self._train_single_seed(
                X_seq, y_labels, y_rul_scaled, seed=base_seed + m * 17,
                is_val=is_val,
            )
            self._ensemble.append(net)

        # Tabular fallback
        if self.use_tabular_fallback:
            self._tabular = _TabularFallback()
            rul_series = X_train["RUL"] if "RUL" in X_train.columns else None
            self._tabular.fit(X_train, y_train, rul=rul_series)

        self.is_fitted = True
        return self

    def _predict_arrays(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Run both branches once and return them aligned to ``X.index``.

        Sequences are built in ``(unit, cycle)`` order, which is not
        generally the caller's row order, so predictions are reindexed by
        the row labels recorded during sequence construction.
        """
        df = X.copy()
        for col in self.label_cols:
            if col not in df.columns:
                df[col] = 0

        X_seq, row_idx, _mask, _first = self._build_extended_sequences(df)
        seq_index = pd.Index(row_idx, name=X.index.name)

        if self._ensemble:
            mean_proba, _std = self._ensemble_predict_proba(X_seq)
            rul_arr = self._ensemble_predict_rul(X_seq)
        elif self._tabular is not None:
            tab = self._tabular.predict_proba(X)
            proba = tab.reindex(seq_index)
            rul = self._tabular.predict_rul(X).reindex(seq_index)
            rul_arr = rul.to_numpy(dtype=float)
            mean_proba = proba[
                [label_col_for(h) for h in self.horizons]
            ].to_numpy(dtype=float)
        else:
            raise RuntimeError("Model not fitted.")

        proba = pd.DataFrame(
            {
                label_col_for(h): mean_proba[:, i]
                for i, h in enumerate(self.horizons)
            },
            index=seq_index,
        ).reindex(X.index)
        proba = proba.fillna(0.5)
        if self.enforce_monotone:
            proba = _enforce_monotone(proba, self.horizons)

        rul = pd.Series(rul_arr, index=seq_index).reindex(X.index)
        rul = rul.fillna(rul.median() if rul.notna().any() else 60.0)
        return proba, np.maximum(rul.to_numpy(dtype=float), 0.0)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities (ensemble mean)."""
        if not self._ensemble and self._tabular is None:
            raise RuntimeError("Model not fitted.")
        proba, _rul = self._predict_arrays(X)
        return proba

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL.

        Uses the sequence branch's regression head when available (the
        tabular branch's RUL estimator is only trained if the fit frame
        carried a ``RUL`` column), and is clipped at zero.
        """
        if not self._ensemble and self._tabular is None:
            raise RuntimeError("Model not fitted.")
        _proba, rul = self._predict_arrays(X)
        return pd.Series(rul, index=X.index, name="RUL")

    def _ensemble_predict_proba(
        self, X_seq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ensemble inference, return (mean_proba, std_proba)."""
        dataset = _SequenceDataset(
            X_seq,
            np.zeros((len(X_seq), len(self.horizons))),
            np.zeros(len(X_seq)),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probas: list[np.ndarray] = []
        for net in self._ensemble:
            net.eval()
            probas: list[np.ndarray] = []
            with torch.no_grad():
                for X_b, _, _ in loader:
                    X_b = X_b.to(self.device)
                    h_logits, _ = net(X_b)
                    probas.append(torch.sigmoid(h_logits).cpu().numpy())
            all_probas.append(np.concatenate(probas, axis=0))

        stacked = np.stack(all_probas, axis=0)  # (M, N, H)
        mean_proba = stacked.mean(axis=0)  # (N, H)
        std_proba = stacked.std(axis=0)  # (N, H)

        return mean_proba, std_proba

    def _ensemble_predict_rul(self, X_seq: np.ndarray) -> np.ndarray:
        """Mean RUL from the sequence branch's regression head, un-scaled."""
        dataset = _SequenceDataset(
            X_seq,
            np.zeros((len(X_seq), len(self.horizons))),
            np.zeros(len(X_seq)),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        out: list[np.ndarray] = []
        for net in self._ensemble:
            net.eval()
            preds: list[np.ndarray] = []
            with torch.no_grad():
                for X_b, _, _ in loader:
                    X_b = X_b.to(self.device)
                    _h_logits, r_pred = net(X_b)
                    preds.append(r_pred.cpu().numpy().ravel())
            out.append(np.concatenate(preds, axis=0))
        return np.mean(np.stack(out, axis=0), axis=0) * self._rul_scale

    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Return max horizon probability as the raw score."""
        proba = self.predict_proba(X)
        return proba.max(axis=1).rename("raw_score")

    def predict_uncertainty(self, X: pd.DataFrame) -> pd.Series:
        """Return ensemble std as an uncertainty measure per row."""
        if not self._ensemble:
            return pd.Series(np.zeros(len(X)), index=X.index, name="uncertainty")

        df = X.copy()
        for col in self.label_cols:
            if col not in df.columns:
                df[col] = 0

        X_seq, row_idx, mask, first_row = self._build_extended_sequences(df)
        _, std_proba = self._ensemble_predict_proba(X_seq)

        # Average std across horizons as a single uncertainty score
        avg_std = std_proba.mean(axis=1)
        return pd.Series(avg_std, index=X.index, name="uncertainty")
