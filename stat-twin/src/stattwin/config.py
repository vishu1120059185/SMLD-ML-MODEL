"""Pydantic v2 configuration loader for STAT-TWIN.

Loads base.yaml, then overlays profile / dataset YAML via ``load_config``.
Validation errors surface as clear ``ValueError`` messages.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-schemas (mirrors configs/base.yaml structure)
# ---------------------------------------------------------------------------

class DatasetCfg(BaseModel):
    name: str = "FD001"
    raw_dir: str = "data/raw/CMAPSS"
    rul_clip: int = 125
    use_age_feature: bool = False


class SplitCfg(BaseModel):
    scheme: str = "group_kfold"
    n_splits: int = 5
    group_col: str = "unit_id"


class MissingCfg(BaseModel):
    strategy: str = "ffill_then_median"
    add_indicators: bool = True


class OutlierCfg(BaseModel):
    method: str = "robust_z"
    threshold: float = 4.0
    action: str = "flag_and_winsorize"


class PreprocessCfg(BaseModel):
    missing: MissingCfg = Field(default_factory=MissingCfg)
    outliers: OutlierCfg = Field(default_factory=OutlierCfg)
    scaler: str = "standard"
    per_condition_norm: str | bool = "auto"


class CorrCfg(BaseModel):
    methods: list[str] = Field(default_factory=lambda: ["pearson", "spearman"])
    pairs: str = "top_k"
    top_k: int = 10


class StatsCfg(BaseModel):
    windows: list[int] = Field(default_factory=lambda: [5, 10, 20, 30])
    ewma_alpha: list[float] = Field(default_factory=lambda: [0.1, 0.3])
    baseline_cycles: int = 30
    corr: CorrCfg = Field(default_factory=CorrCfg)
    shift: list[str] = Field(default_factory=lambda: ["ks", "wasserstein", "psi"])


class SmoothingCfg(BaseModel):
    method: str = "ewma"
    alpha: float = 0.2


class HealthCfg(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "deviation": 0.25,
            "trend": 0.25,
            "ewma": 0.20,
            "variance": 0.15,
            "corr_shift": 0.15,
        }
    )
    calibrate_weights: bool = False
    smoothing: SmoothingCfg = Field(default_factory=SmoothingCfg)
    states_method: str = Field(default="calibrated", alias="states_method")


class StatesCfg(BaseModel):
    method: str = "calibrated"
    fallback_grid: list[int] = Field(default_factory=lambda: [80, 60, 40, 20])
    persistence: int = 3


class ForecastCfg(BaseModel):
    horizons: list[int] = Field(default_factory=lambda: [10, 20, 30, 40, 50])
    seq_len: int = 30
    enforce_monotone: bool = True


class GRUCfg(BaseModel):
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    ensemble_size: int = 3


class XGBCfg(BaseModel):
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05


class RFCfg(BaseModel):
    n_estimators: int = 200
    max_depth: int = 10


class LRCfg(BaseModel):
    C: float = 1.0
    max_iter: int = 1000


class LSTMCfg(BaseModel):
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    ensemble_size: int = 3


class ModelCfg(BaseModel):
    gru: GRUCfg = Field(default_factory=GRUCfg)
    xgb: XGBCfg = Field(default_factory=XGBCfg)
    rf: RFCfg = Field(default_factory=RFCfg)
    lr: LRCfg = Field(default_factory=LRCfg)
    lstm: LSTMCfg = Field(default_factory=LSTMCfg)


class UncertaintyCfg(BaseModel):
    method: str = "split_conformal"
    alpha: float = 0.10
    normalize_by: str = "ensemble_std"


class CalibrationCfg(BaseModel):
    method: str = "isotonic"
    bins: int = 10


class WarningCfg(BaseModel):
    horizon: int = 30
    persistence: int = 3
    valid_window: int = 100
    far_budget: float = 0.05


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class STATTWINConfig(BaseModel):
    """Top-level STAT-TWIN configuration validated against base.yaml."""

    seed: int = 42
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)
    preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
    stats: StatsCfg = Field(default_factory=StatsCfg)
    health: HealthCfg = Field(default_factory=HealthCfg)
    states: StatesCfg = Field(default_factory=StatesCfg)
    forecast: ForecastCfg = Field(default_factory=ForecastCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    uncertainty: UncertaintyCfg = Field(default_factory=UncertaintyCfg)
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    warning: WarningCfg = Field(default_factory=WarningCfg)

    model_config = {"extra": "ignore", "populate_by_name": True}

    # HealthCfg uses a nested "states" dict in YAML; handle aliasing
    @model_validator(mode="before")
    @classmethod
    def flatten_health_states(cls, data: dict[str, Any]) -> dict[str, Any]:
        """If health.states is a dict, promote it to top-level states key."""
        if isinstance(data, dict):
            health = data.get("health")
            if isinstance(health, dict) and "states" in health:
                data["states"] = health.pop("states")
        return data


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return as dict."""
    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML at {path} did not produce a mapping, got {type(raw).__name__}")
    return raw


def _resolve_extends(base: dict[str, Any], cfg_dir: Path) -> dict[str, Any]:
    """Recursively merge ``extends`` chains."""
    extends = base.pop("extends", None)
    if extends is None:
        return base
    parent_path = (cfg_dir / extends).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"extends target not found: {parent_path}")
    parent = _read_yaml(parent_path)
    parent = _resolve_extends(parent, parent_path.parent)
    merged = {**parent, **base}
    return merged


def load_config(
    config_path: str | Path = "configs/base.yaml",
    profile: str | None = None,
    dataset: str | None = None,
) -> STATTWINConfig:
    """Load and validate STAT-TWIN configuration.

    Parameters
    ----------
    config_path:
        Path to the base YAML configuration file.
    profile:
        Optional profile name (looked up in ``configs/profiles/``).
    dataset:
        Optional dataset name (looked up in ``configs/datasets/``).

    Returns
    -------
    STATTWINConfig
        Fully validated configuration object.
    """
    base_path = Path(config_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    cfg_dir = base_path.parent
    merged = _read_yaml(base_path)
    merged = _resolve_extends(merged, cfg_dir)

    # Overlay profile
    if profile is not None:
        profile_path = (cfg_dir / "profiles" / f"{profile}.yaml").resolve()
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        overlay = _read_yaml(profile_path)
        overlay = _resolve_extends(overlay, profile_path.parent)
        merged = {**merged, **overlay}

    # Overlay dataset
    if dataset is not None:
        ds_path = (cfg_dir / "datasets" / f"{dataset.lower()}.yaml").resolve()
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {ds_path}")
        ds_overlay = _read_yaml(ds_path)
        merged["dataset"] = {**merged.get("dataset", {}), **ds_overlay}

    return STATTWINConfig.model_validate(merged)


def config_hash(config_path: str | Path) -> str:
    """Return a short SHA-256 hex digest of the config file content."""
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]
