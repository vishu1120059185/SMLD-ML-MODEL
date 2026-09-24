"""Common utilities for STAT-TWIN.

Provides reproducibility helpers (seed setting), config hashing,
and small convenience functions used across the project.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for ``random``, ``numpy``, and ``torch`` for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic ops may slow training; enable only when needed.
        torch.backends.cudnn.deterministic = False  # noqa: F841 – intentional setting
        torch.backends.cudnn.benchmark = True  # noqa: F841
    except ImportError:
        pass


def get_config_hash(config_path: str | Path) -> str:
    """Return a 12-character SHA-256 hex digest of a config file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    str
        Short hex digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(experiment: str, base: str | Path = "results") -> Path:
    """Return ``{base}/{experiment}``, creating it if needed."""
    return ensure_dir(Path(base) / experiment)


def artifacts_dir(subdir: str = "", base: str | Path = "artifacts") -> Path:
    """Return ``{base}/{subdir}``, creating it if needed."""
    p = Path(base) / subdir if subdir else Path(base)
    return ensure_dir(p)


def load_raw_csv(path: str | Path, **kwargs: Any) -> Any:
    """Convenience wrapper around ``pandas.read_csv`` with sensible defaults."""
    import pandas as pd

    defaults: dict[str, Any] = {"encoding": "utf-8", "low_memory": False}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def check_file_exists(path: str | Path, label: str = "File") -> Path:
    """Raise ``FileNotFoundError`` with a clear message if *path* does not exist.

    Parameters
    ----------
    path:
        Path to check.
    label:
        Human-readable name for the file in the error message.

    Returns
    -------
    Path
        Resolved path.

    Raises
    ------
    FileNotFoundError
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found at {p.resolve()}. "
            "Please download the CMAPSS dataset and place it in data/raw/CMAPSS/."
        )
    return p.resolve()


def chunk_dataframe(df: Any, column: str, n_chunks: int) -> list[Any]:
    """Split a DataFrame into *n_chunks* groups based on unique values in *column*.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    column:
        Column whose unique values determine the chunks.
    n_chunks:
        Number of chunks to create (may be fewer if *column* has fewer unique values).

    Returns
    -------
    list[pandas.DataFrame]
    """

    uniques = df[column].unique()
    splits = np.array_split(uniques, min(n_chunks, len(uniques)))
    return [df[df[column].isin(chunk)].copy() for chunk in splits]
