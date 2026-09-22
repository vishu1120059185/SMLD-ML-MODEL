"""Manifest writer for STAT-TWIN experiment runs.

Writes ``results/{experiment}/manifest.json`` containing git hash, config
hash, seeds, package versions, and timestamp.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import STATTWINConfig, config_hash


def _git_hash(repo_dir: str | Path | None = None) -> str:
    """Return the short git HEAD hash, or ``'unknown'`` if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _git_dirty(repo_dir: str | Path | None = None) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _get_package_versions() -> dict[str, str]:
    """Collect versions of key STAT-TWIN dependencies."""
    packages = [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "xgboost",
        "torch",
        "streamlit",
        "pydantic",
        "pydantic",
        "typer",
        "plotly",
        "shap",
        "matplotlib",
        "seaborn",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib

            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            if ver == "unknown":
                try:
                    from importlib.metadata import version as get_ver

                    ver = get_ver(pkg)
                except Exception:
                    ver = "unknown"
            versions[pkg] = str(ver)
        except ImportError:
            versions[pkg] = "not-installed"
    return versions


def _config_to_dict(cfg: STATTWINConfig) -> dict[str, Any]:
    """Serialize config to a plain dict for JSON."""
    return cfg.model_dump(mode="json")


def write_manifest(
    experiment_dir: str | Path,
    cfg: STATTWINConfig,
    config_path: str | Path,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write manifest.json to ``results/{experiment}/manifest.json``.

    Parameters
    ----------
    experiment_dir:
        Path to the experiment output directory, e.g. ``results/e0_data_audit``.
    cfg:
        The validated STAT-TWIN configuration for this run.
    config_path:
        Path to the YAML config file that was loaded (used for hashing).
    extras:
        Optional additional key-value pairs to include in the manifest.

    Returns
    -------
    Path
        The written manifest file path.
    """
    out = Path(experiment_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "timestamp": now,
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "config_file": str(Path(config_path).resolve()),
        "config_hash": config_hash(config_path),
        "seed": cfg.seed,
        "dataset": cfg.dataset.name,
        "profile": None,  # caller can fill via extras
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "package_versions": _get_package_versions(),
        "config": _config_to_dict(cfg),
    }

    if extras:
        manifest.update(extras)

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and return a previously written manifest.json.

    Parameters
    ----------
    path:
        Path to the manifest file.

    Returns
    -------
    dict
        The manifest contents.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
