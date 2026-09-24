"""TelemetryDataset – dataset-agnostic abstraction for STAT-TWIN.

Wraps a validated ``pandas.DataFrame`` with a minimal, uniform interface so
that downstream components (preprocessing, models, evaluation) never need to
know whether they are working with C-MAPSS, NASA, or a synthetic fixture.

Responsibilities
----------------
* Validate the underlying DataFrame once at construction time.
* Expose read-only properties for features, RUL, labels, unit IDs, and cycles.
* Provide convenience methods for unit-level grouping and splitting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stattwin.data.schema import (
    FAILURE_HORIZONS,
    OP_SETTINGS,
    label_col_for,
)
from stattwin.data.splitter import validate_dataset

__all__ = ["TelemetryDataset"]


@dataclass(frozen=True)
class TelemetryDataset:
    """Immutable, validated wrapper around a telemetry ``DataFrame``.

    Parameters
    ----------
    df:
        Must contain at least ``unit_id`` and ``cycle`` columns, plus any
        sensor / operational-setting / label columns of interest.  The
        DataFrame is **copied** on construction to guarantee immutability.
    name:
        Optional human-readable name (e.g. ``"FD001"``).
    """

    df: pd.DataFrame = field(repr=False)
    name: str = ""

    def __post_init__(self) -> None:
        validate_dataset(self.df, require_rul=False, require_labels=False)
        # freeze the DataFrame
        object.__setattr__(self, "_df_cache", self.df.copy())

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    @property
    def _df(self) -> pd.DataFrame:
        return self._df_cache  # type: ignore[attr-defined]

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    @property
    def columns(self) -> pd.Index:
        return self._df.columns

    @property
    def unit_ids(self) -> np.ndarray:
        """Sorted unique unit identifiers."""
        return np.sort(self._df["unit_id"].unique())

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def cycles(self) -> pd.Series:
        """The ``cycle`` column."""
        return self._df["cycle"]

    @property
    def rul(self) -> pd.Series:
        """The ``RUL`` column (if present, else ``None``)."""
        if "RUL" in self._df.columns:
            return self._df["RUL"]
        raise AttributeError("RUL column not present in the dataset.")

    @property
    def has_labels(self) -> bool:
        """True if at least one ``fail_h{h}`` column is present."""
        return any(label_col_for(h) in self._df.columns for h in FAILURE_HORIZONS)

    @property
    def failure_horizons(self) -> list[int]:
        """Horizons for which labels are currently available."""
        return [h for h in FAILURE_HORIZONS if label_col_for(h) in self._df.columns]

    # ------------------------------------------------------------------
    # Feature / target extraction
    # ------------------------------------------------------------------

    def features(
        self,
        *,
        include_settings: bool = True,
        include_sensors: bool = True,
        sensor_indices: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Return the feature sub-DataFrame.

        Parameters
        ----------
        include_settings:
            Include ``op_setting_1 … op_setting_3``.
        include_sensors:
            Include ``sensor_1 … sensor_21``.
        sensor_indices:
            Optional 1-based sensor indices to include (e.g. ``[2, 7, 11]``).
            If *None* and ``include_sensors`` is True, all 21 sensors are
            returned.
        """
        cols: list[str] = []
        if include_settings:
            cols.extend(OP_SETTINGS)
        if include_sensors:
            if sensor_indices is not None:
                cols.extend(f"sensor_{i}" for i in sensor_indices)
            else:
                cols.extend(f"sensor_{i}" for i in range(1, 22))
        return self._df[cols]

    def labels(self, horizon: int) -> pd.Series:
        """Return binary label column for the given *horizon*."""
        col = label_col_for(horizon)
        if col not in self._df.columns:
            raise KeyError(
                f"Label column '{col}' not found. "
                f"Available horizons: {self.failure_horizons}"
            )
        return self._df[col]

    def all_labels(self) -> pd.DataFrame:
        """Return DataFrame of all failure-label columns."""
        cols = [label_col_for(h) for h in self.failure_horizons]
        if not cols:
            raise AttributeError("No label columns found in the dataset.")
        return self._df[cols]

    # ------------------------------------------------------------------
    # Unit-level operations
    # ------------------------------------------------------------------

    def unit_frame(self, unit_id: int) -> pd.DataFrame:
        """Return all rows for a single unit, sorted by cycle."""
        mask = self._df["unit_id"] == unit_id
        if not mask.any():
            raise KeyError(f"Unit {unit_id} not found.")
        return self._df.loc[mask].sort_values("cycle")

    def units(self, unit_ids: Sequence[int]) -> TelemetryDataset:
        """Return a new ``TelemetryDataset`` containing only the listed units."""
        mask = self._df["unit_id"].isin(unit_ids)
        if not mask.any():
            raise KeyError("None of the requested unit IDs are present.")
        return TelemetryDataset(
            df=self._df.loc[mask].copy(),
            name=f"{self.name}[{len(unit_ids)} units]" if self.name else "",
        )

    def split_by_units(
        self,
        train_units: np.ndarray,
        val_units: np.ndarray,
    ) -> tuple[TelemetryDataset, TelemetryDataset]:
        """Split into two ``TelemetryDataset`` objects by unit ID sets.

        Returns
        -------
        (train_ds, val_ds)
            Two datasets whose unit sets are guaranteed disjoint.
        """
        overlap = set(train_units) & set(val_units)
        if overlap:
            raise ValueError(
                f"Overlapping units between train and val: {overlap}"
            )
        return self.units(train_units), self.units(val_units)

    def copy(self) -> TelemetryDataset:
        """Return a deep copy of this dataset."""
        return TelemetryDataset(df=self._df.copy(), name=self.name)

    # ------------------------------------------------------------------
    # dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        label = f" '{self.name}'" if self.name else ""
        return (
            f"TelemetryDataset{label}: "
            f"{self.n_units} units, {len(self)} rows, "
            f"{len(self.columns)} columns"
        )
