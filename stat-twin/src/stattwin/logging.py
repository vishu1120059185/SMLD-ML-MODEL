"""Centralized logging configuration for STAT-TWIN.

Usage::

    from stattwin.logging import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

_CONFIGURED = False


def _setup_root(level: int = logging.INFO, stream: TextIO = sys.stderr) -> None:
    """Configure the root ``stattwin`` logger once."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger("stattwin")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a namespaced logger under ``stattwin``.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` from the calling module.
    level:
        Optional per-logger level override.

    Returns
    -------
    logging.Logger
    """
    _setup_root()
    logger = logging.getLogger(f"stattwin.{name}")
    if level is not None:
        logger.setLevel(level)
    return logger


def add_file_handler(
    path: str | Path,
    level: int = logging.DEBUG,
    fmt: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.FileHandler:
    """Attach a file handler to the ``stattwin`` root logger.

    Parameters
    ----------
    path:
        File path for the log output.
    level:
        Minimum severity for the file handler.
    fmt:
        Log format string.
    datefmt:
        Date format string.

    Returns
    -------
    logging.FileHandler
        The handler so the caller can remove it later if needed.
    """
    _setup_root()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(p), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger("stattwin").addHandler(fh)
    return fh
