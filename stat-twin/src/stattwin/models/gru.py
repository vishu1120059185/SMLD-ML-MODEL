"""Model 6: GRU / LSTM sequence model over raw sensor cycles.

Processes the last ``seq_len`` cycles of raw sensor readings through a
2-layer GRU (or LSTM) with hidden size 64 and dropout 0.2.  Multi-head
output produces 5 horizon logits (sigmoid → probability) and 1 RUL
scalar (ReLU).

The model handles variable-length sequences by padding shorter runs
with the first-cycle values and masking during aggregation.

References
----------
*  Cho et al. (2014) – GRU.
*  Hochreiter & Schmidhuber (1997) – LSTM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from stattwin.data.schema import FAILURE_HORIZONS, label_col_for
from stattwin.models.base import BaseModel

__all__ = ["GRUModel"]

_UNIT_COL = "unit_id"
_CYCLE_COL = "cycle"


# -----------------------------------------------------------------------
# PyTorch components
# -----------------------------------------------------------------------


class _SequenceDataset(Dataset):
    """Dataset of (sequence, labels, rul) tuples for GRU training."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        rul: np.ndarray | None,
    ) -> None:
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.rul = (
            torch.as_tensor(rul, dtype=torch.float32)
            if rul is not None
            else torch.zeros(len(sequences), dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx], self.rul[idx]


class _GRUNetwork(nn.Module):
    """GRU with multi-head output: 5 horizon logits + 1 RUL scalar."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_horizons: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Multi-head outputs
        self.horizon_head = nn.Linear(hidden_size, num_horizons)
        self.rul_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: (batch, seq_len, input_size)

        Returns
        -------
        horizon_logits: (batch, num_horizons)
        rul_pred: (batch,)  – ReLU-activated
        """
        # GRU: (batch, seq_len, hidden)
        out, _ = self.gru(x)
        # Take the last time step
        last = out[:, -1, :]  # (batch, hidden)
        last = self.dropout(last)

        horizon_logits = self.horizon_head(last)  # (batch, num_horizons)
        rul_pred = self.rul_head(last).squeeze(-1)  # (batch,)
        rul_pred = torch.relu(rul_pred)

        return horizon_logits, rul_pred


class _LSTMNetwork(nn.Module):
    """LSTM with multi-head output: 5 horizon logits + 1 RUL scalar."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_horizons: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.horizon_head = nn.Linear(hidden_size, num_horizons)
        self.rul_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)

        horizon_logits = self.horizon_head(last)
        rul_pred = self.rul_head(last).squeeze(-1)
        rul_pred = torch.relu(rul_pred)

        return horizon_logits, rul_pred


# -----------------------------------------------------------------------
# Sequence builder
# -----------------------------------------------------------------------


def _build_sequences(
    df: pd.DataFrame,
    sensor_cols: list[str],
    seq_len: int,
    unit_col: str = _UNIT_COL,
    cycle_col: str = _CYCLE_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build padded sequences from panel data.

    For each row (unit, t), extract the last ``seq_len`` cycles up to and
    including t, pad shorter sequences with the first cycle's values.

    Returns
    -------
    X_seq: (n_samples, seq_len, n_features)
    y_labels: (n_samples, n_horizons)
    y_rul: (n_samples,)
    row_idx: (n_samples,) – original DataFrame row indices
    mask: (n_samples, seq_len) – 1 for real, 0 for padded
    """
    all_units = df[unit_col].unique()
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rul_list: list[np.ndarray] = []
    idx_list: list[int] = []
    mask_list: list[np.ndarray] = []

    label_cols = [label_col_for(h) for h in FAILURE_HORIZONS]

    for unit in all_units:
        unit_df = df[df[unit_col] == unit].sort_values(cycle_col)
        n_cycles = len(unit_df)
        sensor_vals = unit_df[sensor_cols].to_numpy(dtype=np.float64)
        first_row = sensor_vals[0:1, :]  # (1, n_features)

        for t in range(n_cycles):
            start = max(0, t - seq_len + 1)
            seq = sensor_vals[start : t + 1, :]  # (actual_len, n_features)
            actual_len = seq.shape[0]

            # Pad to seq_len
            if actual_len < seq_len:
                pad = np.tile(first_row, (seq_len - actual_len, 1))
                seq = np.concatenate([pad, seq], axis=0)
                mask = np.concatenate(
                    [np.zeros(seq_len - actual_len), np.ones(actual_len)]
                )
            else:
                mask = np.ones(seq_len)

            X_list.append(seq)
            idx_list.append(unit_df.index[t])

            # Labels
            row_labels = unit_df.iloc[t]
            if all(col in row_labels.index for col in label_cols):
                y_list.append(
                    np.array(
                        [int(row_labels[col]) for col in label_cols],
                        dtype=np.float64,
                    )
                )
            else:
                y_list.append(np.zeros(len(label_cols), dtype=np.float64))

            if "RUL" in row_labels.index:
                rul_list.append(float(row_labels["RUL"]))
            else:
                rul_list.append(60.0)

            mask_list.append(mask)

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64),
        np.array(rul_list, dtype=np.float64),
        np.array(idx_list, dtype=np.int64),
        np.array(mask_list, dtype=np.float64),
    )


# -----------------------------------------------------------------------
# Main model class
# -----------------------------------------------------------------------


class GRUModel(BaseModel):
    """GRU / LSTM sequence model for failure prediction.

    Parameters
    ----------
    backbone:
        Sequence model type: ``"gru"`` or ``"lstm"`` (default ``"gru"``).
    hidden:
        Hidden size (default 64).
    layers:
        Number of RNN layers (default 2).
    dropout:
        Dropout rate (default 0.2).
    seq_len:
        Maximum sequence length in cycles (default 30).
    lr:
        Learning rate (default 1e-3).
    batch_size:
        Mini-batch size (default 64).
    epochs:
        Maximum training epochs (default 50).
    patience:
        Early-stopping patience (default 8).
    horizon_weight:
        Loss weight for horizon BCE vs RUL MSE (default 1.0).
    device:
``"cpu"`` or ``"cuda"`` (default auto-detect).
    horizons:
        Failure horizons (must be 5 for default multi-head).
    """

    def __init__(
        self,
        backbone: Literal["gru", "lstm"] = "gru",
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 30,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 8,
        horizon_weight: float = 1.0,
        device: str | None = None,
        horizons: Sequence[int] | None = None,
    ) -> None:
        super().__init__(horizons=horizons, name=f"{backbone.upper()}Model")
        self.backbone = backbone
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.horizon_weight = horizon_weight

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._net: _GRUNetwork | _LSTMNetwork | None = None
        self._sensor_cols: list[str] = []
        self._input_size: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        """Return raw sensor columns only."""
        return [
            c for c in X.columns
            if c not in {_UNIT_COL, _CYCLE_COL, "RUL"}
            and pd.api.types.is_numeric_dtype(X[c])
        ]

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

    def _train_epoch(
        self,
        loader: DataLoader,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        bce_loss: nn.Module,
        mse_loss: nn.Module,
    ) -> float:
        """Train for one epoch, return average total loss."""
        net.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch, rul_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            rul_batch = rul_batch.to(self.device)

            optimizer.zero_grad()
            horizon_logits, rul_pred = net(X_batch)

            loss_h = bce_loss(horizon_logits, y_batch)
            loss_r = mse_loss(rul_pred, rul_batch)
            loss = self.horizon_weight * loss_h + loss_r

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        net: nn.Module,
        bce_loss: nn.Module,
        mse_loss: nn.Module,
    ) -> tuple[float, float, float]:
        """Evaluate, return (total_loss, bce_loss, mse_loss)."""
        net.eval()
        total = 0.0
        bce_total = 0.0
        mse_total = 0.0
        n = 0
        for X_batch, y_batch, rul_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            rul_batch = rul_batch.to(self.device)

            horizon_logits, rul_pred = net(X_batch)
            l_h = bce_loss(horizon_logits, y_batch).item()
            l_r = mse_loss(rul_pred, rul_batch).item()
            total += l_h + l_r
            bce_total += l_h
            mse_total += l_r
            n += 1

        return total / max(n, 1), bce_total / max(n, 1), mse_total / max(n, 1)

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups: np.ndarray | None = None,
    ) -> GRUModel:
        """Fit the GRU/LSTM model on sequence data.

        Parameters
        ----------
        X_train:
            Feature matrix with ``unit_id``, ``cycle``, and sensor columns.
        y_train:
            Binary label DataFrame (aligned with X_train).
        groups:
            Ignored.
        """
        self._sensor_cols = self._select_features(X_train)
        self._input_size = len(self._sensor_cols)

        # Merge labels into X_train for sequence building
        df = X_train.copy()
        for col in self.label_cols:
            if col not in df.columns and col in y_train.columns:
                df[col] = y_train[col].values

        X_seq, y_labels, y_rul, row_idx, mask = _build_sequences(
            df, self._sensor_cols, self.seq_len
        )

        # Create network
        self._net = self._create_network(self._input_size).to(self.device)

        # Data loader
        dataset = _SequenceDataset(X_seq, y_labels, y_rul)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        # Losses with class weighting
        pos_weights = []
        for i in range(y_labels.shape[1]):
            n_pos = y_labels[:, i].sum()
            n_neg = len(y_labels) - n_pos
            pw = n_neg / max(n_pos, 1.0)
            pos_weights.append(min(pw, 50.0))  # cap extreme weights
        pos_weight = torch.tensor(pos_weights, dtype=torch.float32, device=self.device)

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse_loss = nn.MSELoss()

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Training loop with early stopping
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for _epoch in range(self.epochs):
            _train_loss = self._train_epoch(loader, self._net, optimizer, bce_loss, mse_loss)
            val_loss, _, _ = self._evaluate(loader, self._net, bce_loss, mse_loss)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
            self._net = self._net.to(self.device)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict per-horizon failure probabilities."""
        if self._net is None:
            raise RuntimeError("Model not fitted.")

        df = X.copy()
        # Ensure label columns exist for sequence building
        for col in self.label_cols:
            if col not in df.columns:
                df[col] = 0

        X_seq, _, _, row_idx, _ = _build_sequences(
            df, self._sensor_cols, self.seq_len
        )

        self._net.eval()
        dataset = _SequenceDataset(X_seq, np.zeros((len(X_seq), len(self.horizons))), None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_logits: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _, _ in loader:
                X_batch = X_batch.to(self.device)
                horizon_logits, _ = self._net(X_batch)
                all_logits.append(torch.sigmoid(horizon_logits).cpu().numpy())

        proba_arr = np.concatenate(all_logits, axis=0)
        proba_dict = {
            label_col_for(h): proba_arr[:, i]
            for i, h in enumerate(self.horizons)
        }
        return pd.DataFrame(proba_dict, index=X.index)

    def predict_rul(self, X: pd.DataFrame) -> pd.Series:
        """Predict point RUL."""
        if self._net is None:
            raise RuntimeError("Model not fitted.")

        df = X.copy()
        for col in self.label_cols:
            if col not in df.columns:
                df[col] = 0

        X_seq, _, _, row_idx, _ = _build_sequences(
            df, self._sensor_cols, self.seq_len
        )

        self._net.eval()
        dataset = _SequenceDataset(X_seq, np.zeros((len(X_seq), len(self.horizons))), None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_rul: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _, _ in loader:
                X_batch = X_batch.to(self.device)
                _, rul_pred = self._net(X_batch)
                all_rul.append(rul_pred.cpu().numpy())

        rul_arr = np.concatenate(all_rul, axis=0)
        return pd.Series(np.maximum(rul_arr, 0.0), index=X.index, name="RUL")

    def score_raw(self, X: pd.DataFrame) -> pd.Series:
        """Return max horizon probability as the raw score."""
        proba = self.predict_proba(X)
        return proba.max(axis=1).rename("raw_score")
