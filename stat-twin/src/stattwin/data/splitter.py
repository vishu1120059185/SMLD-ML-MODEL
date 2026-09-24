"""Unit-level data splitting for STAT-TWIN.

Provides ``GroupKFold(n_splits=5)`` splits at the **unit** level, an inner
unit-level split for early-stopping validation, and dataset validation
assertions.

Design rules (from the masterplan):

* No cycle-level shuffling ever – cycles are always ordered within each unit.
* Outer folds use ``GroupKFold(n_splits=5)`` with *unit_id* as group.
* Inner validation split picks ~15 % of **outer-training** units for early
  stopping; the two unit sets are guaranteed disjoint.
* Validation asserts shape, column count, monotone cycles per unit, and no
  duplicate ``(unit_id, cycle)`` pairs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

__all__ = [
    "make_group_kfold_splits",
    "inner_unit_split",
    "validate_dataset",
]

# Default number of outer folds
N_SPLITS: int = 5

# Fraction of outer-training units reserved for the inner validation set
INNER_VAL_FRAC: float = 0.15


# -----------------------------------------------------------------------
# Outer folds
# -----------------------------------------------------------------------


def make_group_kfold_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    seed: int = 42,
) -> list[dict[str, np.ndarray]]:
    """Yield outer ``GroupKFold`` splits at the **unit** level.

    Parameters
    ----------
    df:
        DataFrame that must contain a ``unit_id`` column.
    n_splits:
        Number of folds (default 5).
    seed:
        Random seed forwarded to ``GroupKFold`` (via ``shuffle``).

    Returns
    -------
    list of dict
        Each dict has keys ``"train_units"`` and ``"val_units"`` containing
        ``np.ndarray`` of integer unit IDs.
    """
    _check_unit_column(df)

    units = df["unit_id"].unique()
    groups = units  # one group per unit
    # GroupKFold does not shuffle; it deterministically assigns groups to folds.
    gkf = GroupKFold(n_splits=min(n_splits, len(units)))

    # ``split`` iterates over (train_idx, val_idx) where idx refer to the
    # *rows of ``units``* (because we pass ``units`` as both X and group).
    splits: list[dict[str, np.ndarray]] = []
    for train_idx, val_idx in gkf.split(X=units, groups=groups):
        splits.append(
            {
                "train_units": units[train_idx],
                "val_units": units[val_idx],
            }
        )

    _assert_disjoint_outer_folds(splits)
    return splits


def _assert_disjoint_outer_folds(
    splits: list[dict[str, np.ndarray]],
) -> None:
    """Ensure no unit appears in more than one outer-validation fold."""
    seen_val: set = set()
    for i, s in enumerate(splits):
        overlap = seen_val & set(s["val_units"])
        if overlap:
            raise ValueError(
                f"Outer fold {i} shares val units with a previous fold: {overlap}"
            )
        seen_val.update(s["val_units"])


# -----------------------------------------------------------------------
# Inner validation split
# -----------------------------------------------------------------------


def inner_unit_split(
    train_units: np.ndarray,
    *,
    val_frac: float = INNER_VAL_FRAC,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split outer-training units into inner-train and inner-validation sets.

    The inner-validation set contains ~``val_frac`` of the units and is
    **guaranteed disjoint** from the inner-train set.

    Parameters
    ----------
    train_units:
        Array of unit IDs belonging to the outer training fold.
    val_frac:
        Fraction of units to allocate to the inner validation set.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (inner_train_units, inner_val_units)
        Two disjoint ``np.ndarray`` of integer unit IDs whose union equals
        *train_units*.
    """
    rng = np.random.default_rng(seed)
    units = np.unique(train_units)
    n_total = len(units)
    n_val = max(1, int(math.floor(n_total * val_frac)))

    perm = rng.permutation(units)
    inner_val = perm[:n_val]
    inner_train = perm[n_val:]

    # Disjointness invariant
    overlap = set(inner_train) & set(inner_val)
    if overlap:
        raise ValueError(
            f"Inner split produced overlapping units: {overlap}"
        )

    # Coverage invariant
    if set(inner_train) | set(inner_val) != set(units):
        raise ValueError("Inner split does not cover all training units.")

    return inner_train, inner_val


# -----------------------------------------------------------------------
# Dataset validation
# -----------------------------------------------------------------------


def validate_dataset(
    df: pd.DataFrame,
    *,
    require_rul: bool = False,
    require_labels: bool = False,
    expected_columns: list[str] | None = None,
) -> None:
    """Validate structural invariants of a telemetry ``DataFrame``.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    require_rul:
        If *True*, check that an ``RUL`` column is present.
    require_labels:
        If *True*, check that ``fail_h{h}`` columns exist for horizons
        10–50.
    expected_columns:
        Optional explicit list of columns that must be present.

    Raises
    ------
    ValueError
        On any invariant violation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(df).__name__}.")

    if len(df) == 0:
        raise ValueError("DataFrame is empty.")

    _check_unit_column(df)

    # Column presence
    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # Cycle column must be integer
    if not pd.api.types.is_integer_dtype(df["cycle"]):
        raise ValueError("cycle column must be integer-typed.")

    # Monotone cycles per unit
    for uid, grp in df.groupby("unit_id"):
        cycles = grp["cycle"].values
        if not all(cycles[i] <= cycles[i + 1] for i in range(len(cycles) - 1)):
            raise ValueError(
                f"Unit {uid}: cycles are not monotonically increasing."
            )

    # No duplicate (unit_id, cycle)
    dup = df.duplicated(subset=["unit_id", "cycle"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["unit_id", "cycle"]].drop_duplicates()
        raise ValueError(
            f"Duplicate (unit_id, cycle) pairs:\n{bad.to_string()}"
        )

    if require_rul and "RUL" not in df.columns:
        raise ValueError("RUL column is required but missing.")

    if require_labels:
        from stattwin.data.schema import FAILURE_HORIZONS, label_col_for

        for h in FAILURE_HORIZONS:
            col = label_col_for(h)
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' is required but missing.")


def _check_unit_column(df: pd.DataFrame) -> None:
    if "unit_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'unit_id' column.")
