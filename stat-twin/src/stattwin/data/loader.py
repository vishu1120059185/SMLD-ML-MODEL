"""C-MAPSS file loader.

Reads space-separated C-MAPSS turbofan engine files (no header, 26 columns),
validates structural invariants, computes RUL and failure labels, and returns a
clean ``pandas.DataFrame`` ready for the modelling pipeline.

Raises
------
FileNotFoundError
    If the requested file does not exist.
ValueError
    If the file has the wrong number of columns, duplicate (unit, cycle)
    pairs, non-monotone cycles, or zero-length data.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from stattwin.data.schema import (
    COLUMN_NAMES,
    FAILURE_HORIZONS,
    label_col_for,
)

__all__ = ["load_cmapss"]


def load_cmapss(
    path: str | pathlib.Path,
    *,
    horizons: list[int] | None = None,
    add_labels: bool = True,
) -> pd.DataFrame:
    """Load a single C-MAPSS text file and return a validated ``DataFrame``.

    Parameters
    ----------
    path:
        Path to a C-MAPSS ``*.txt`` file (space-separated, no header).
    horizons:
        Failure horizons used to compute binary labels.  Defaults to
        ``FAILURE_HORIZONS`` (``[10, 20, 30, 40, 50]``).
    add_labels:
        If *True* (default), compute ``RUL`` **and** binary ``y_h`` columns
        for every horizon.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``COLUMN_NAMES + ["RUL"]`` and optionally
        ``fail_h10 … fail_h50``.
    """
    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    # --- read raw data -----------------------------------------------------
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="c",
    )

    # --- structural validation ---------------------------------------------
    _validate_structure(df)

    # --- RUL computation ---------------------------------------------------
    df = _compute_rul(df)

    # --- binary failure labels ---------------------------------------------
    if add_labels:
        h_list = horizons if horizons is not None else FAILURE_HORIZONS
        df = _add_failure_labels(df, h_list)

    return df


# ===================================================================
# Internal helpers
# ===================================================================


def _validate_structure(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the DataFrame is structurally invalid."""

    if len(df) == 0:
        raise ValueError("Loaded file contains no rows.")

    if len(df.columns) != 26:
        raise ValueError(
            f"Expected 26 columns, got {len(df.columns)}.  "
            "Is this a valid C-MAPSS file?"
        )

    # integer types for identifiers
    if not pd.api.types.is_integer_dtype(df["unit_id"]):
        raise ValueError("unit_id must be integer-typed.")

    if not pd.api.types.is_integer_dtype(df["cycle"]):
        raise ValueError("cycle must be integer-typed.")

    # uniqueness of (unit_id, cycle)
    dup_mask = df.duplicated(subset=["unit_id", "cycle"], keep=False)
    if dup_mask.any():
        bad = df.loc[dup_mask, ["unit_id", "cycle"]].drop_duplicates()
        raise ValueError(
            f"Duplicate (unit_id, cycle) pairs found:\n{bad.to_string()}"
        )

    # cycles must be monotonically increasing within each unit
    for uid, grp in df.groupby("unit_id"):
        cycles = grp["cycle"].values
        if not all(cycles[i] <= cycles[i + 1] for i in range(len(cycles) - 1)):
            raise ValueError(
                f"Unit {uid}: cycles are not monotonically increasing."
            )
        # cycles must start at 1
        if cycles[0] != 1:
            raise ValueError(
                f"Unit {uid}: first cycle is {cycles[0]}, expected 1."
            )


def _compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``RUL`` column (max cycle per unit − current cycle)."""
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).astype(int)
    return df


def _add_failure_labels(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Add binary ``fail_h{h}`` columns for each horizon *h*.

    ``fail_h{h} == 1`` iff ``RUL <= h``.
    """
    df = df.copy()
    for h in horizons:
        col = label_col_for(h)
        df[col] = (df["RUL"] <= h).astype(int)
    return df
