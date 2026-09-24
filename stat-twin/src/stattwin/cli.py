"""Typer CLI for STAT-TWIN.

Provides subcommands that map 1-to-1 to Makefile targets::

    stattwin data --ds FD001 --profile fast
    stattwin features --ds FD001 --profile fast
    stattwin train --ds FD001 --model xgb --profile fast
    stattwin eval --ds FD001 --profile fast
    stattwin e0 --ds FD001 --profile fast
    ...
    stattwin app-data --ds FD001 --profile fast
    stattwin app
    stattwin test
    stattwin reproduce --profile full
"""

from __future__ import annotations

import typer

from .config import STATTWINConfig, load_config
from .logging import get_logger
from .manifest import write_manifest
from .utils import ensure_dir, set_seeds

logger = get_logger("cli")

app = typer.Typer(
    name="stattwin",
    help="STAT-TWIN: Statistical Digital Twin for Probabilistic Failure Forecasting.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Shared option helpers
# ---------------------------------------------------------------------------

def _common_options(
    ds: str = typer.Option("FD001", "--ds", help="Dataset name (FD001–FD004, smoke)."),
    profile: str = typer.Option("fast", "--profile", help="Profile: smoke, fast, full."),
    config: str = typer.Option("configs/base.yaml", "--config", help="Path to base config."),
) -> STATTWINConfig:
    """Load config, set seeds, and return validated config."""
    cfg = load_config(config, profile=profile, dataset=ds)
    set_seeds(cfg.seed)
    logger.info("Config loaded: ds=%s profile=%s seed=%d", ds, profile, cfg.seed)
    return cfg


def _write_run_manifest(
    experiment: str,
    cfg: STATTWINConfig,
    config_path: str,
    profile: str | None = None,
) -> None:
    """Write manifest.json into the experiment results dir."""
    out = ensure_dir(f"results/{experiment}")
    extras: dict = {}
    if profile:
        extras["profile"] = profile
    manifest_path = write_manifest(out, cfg, config_path, extras=extras)
    logger.info("Manifest written to %s", manifest_path)


def _run_experiment(
    module_name: str,
    *,
    ds: str = "FD001",
    profile: str | None = None,
    config: str = "configs/base.yaml",
) -> None:
    """Invoke a real experiment module (same entry point as the Makefile)."""
    import importlib
    import sys

    module = f"stattwin.experiments.{module_name}"
    argv_backup = sys.argv[:]
    try:
        sys.argv = [module, "--ds", ds, "--config", config]
        if profile:
            sys.argv.extend(["--profile", profile])
        logger.info("Running experiment module %s", module)
        mod = importlib.import_module(module)
        if hasattr(mod, "main"):
            raise SystemExit(mod.main())
        raise SystemExit(0)
    except ModuleNotFoundError as exc:
        logger.error("Experiment module not found: %s (%s)", module, exc)
        raise typer.Exit(code=1) from exc
    except SystemExit as exc:
        if exc.code not in (0, None):
            logger.error("Experiment %s failed with code %s", module, exc.code)
            raise
    finally:
        sys.argv = argv_backup


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def data(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """Download / validate raw CMAPSS data."""
    cfg = _common_options(ds, profile, config)
    logger.info("data command: ds=%s raw_dir=%s", cfg.dataset.name, cfg.dataset.raw_dir)
    _write_run_manifest("data", cfg, config, profile)


@app.command()
def features(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """Engineer features (sliding-window stats, health indices)."""
    cfg = _common_options(ds, profile, config)
    logger.info("features command: ds=%s", cfg.dataset.name)
    _write_run_manifest("features", cfg, config, profile)


@app.command()
def train(
    ds: str = typer.Option("FD001", "--ds"),
    model: str = typer.Option("xgb", "--model", help="Model type: xgb, rf, lr, gru, lstm."),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """Train a survival / regression model."""
    cfg = _common_options(ds, profile, config)
    logger.info("train command: ds=%s model=%s", cfg.dataset.name, model)
    _write_run_manifest(f"train_{model}", cfg, config, profile)


@app.command()
def eval(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """Evaluate trained models on the test split."""
    cfg = _common_options(ds, profile, config)
    logger.info("eval command: ds=%s", cfg.dataset.name)
    _write_run_manifest("eval", cfg, config, profile)


# Experiments e0 – e9
def _make_experiment_cmd(name: str, description: str):
    """Factory that returns a Typer callback for an experiment sub-command."""

    def _cmd(
        ds: str = typer.Option("FD001", "--ds"),
        profile: str = typer.Option("fast", "--profile"),
        config: str = typer.Option("configs/base.yaml", "--config"),
    ) -> None:
        cfg = _common_options(ds, profile, config)
        logger.info("%s command: ds=%s", name, cfg.dataset.name)
        _write_run_manifest(name, cfg, config, profile)

    _cmd.__doc__ = description
    _cmd.__name__ = name
    return _cmd


@app.command("e0")
def e0(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E0 – Data Audit: completeness, distributions, operating conditions."""
    cfg = _common_options(ds, profile, config)
    logger.info("e0_data_audit: ds=%s", cfg.dataset.name)
    _write_run_manifest("e0_data_audit", cfg, config, profile)
    _run_experiment("e0_data_audit", ds=ds, profile=profile, config=config)


@app.command("e1")
def e1(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E1 – Health Index Construction and calibration."""
    cfg = _common_options(ds, profile, config)
    logger.info("e1_health_index: ds=%s", cfg.dataset.name)
    _write_run_manifest("e1_health_index", cfg, config, profile)
    _run_experiment("e1_health_index", ds=ds, profile=profile, config=config)


@app.command("e2")
def e2(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E2 – Model Comparison across XGB, RF, LR, GRU, LSTM."""
    cfg = _common_options(ds, profile, config)
    logger.info("e2_model_comparison: ds=%s", cfg.dataset.name)
    _write_run_manifest("e2_model_comparison", cfg, config, profile)
    _run_experiment("e2_model_comparison", ds=ds, profile=profile, config=config)


@app.command("e3")
def e3(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E3 – Early Warning thresholds and false-alarm rates."""
    cfg = _common_options(ds, profile, config)
    logger.info("e3_early_warning: ds=%s", cfg.dataset.name)
    _write_run_manifest("e3_early_warning", cfg, config, profile)
    _run_experiment("e3_early_warning", ds=ds, profile=profile, config=config)


@app.command("e4")
def e4(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E4 – Ablation study on features and window sizes."""
    cfg = _common_options(ds, profile, config)
    logger.info("e4_ablation: ds=%s", cfg.dataset.name)
    _write_run_manifest("e4_ablation", cfg, config, profile)
    _run_experiment("e4_ablation", ds=ds, profile=profile, config=config)


@app.command("e5")
def e5(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E5 – Uncertainty quantification: split conformal, bootstrap, ensemble."""
    cfg = _common_options(ds, profile, config)
    logger.info("e5_uncertainty: ds=%s", cfg.dataset.name)
    _write_run_manifest("e5_uncertainty", cfg, config, profile)
    _run_experiment("e5_uncertainty", ds=ds, profile=profile, config=config)


@app.command("e6")
def e6(
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E6 – Generalisation across FD001–FD004 (all datasets)."""
    cfg = _common_options("FD001", profile, config)
    logger.info("e6_generalization")
    _write_run_manifest("e6_generalization", cfg, config, profile)
    _run_experiment("e6_generalization", ds="FD001", profile=profile, config=config)


@app.command("e7")
def e7(
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E7 – Operating-condition sensitivity."""
    cfg = _common_options("FD001", profile, config)
    logger.info("e7_operating_conditions")
    _write_run_manifest("e7_operating_conditions", cfg, config, profile)
    _run_experiment("e7_operating_conditions", ds="FD001", profile=profile, config=config)


@app.command("e8")
def e8(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E8 – Fault injection robustness test."""
    cfg = _common_options(ds, profile, config)
    logger.info("e8_fault_injection: ds=%s", cfg.dataset.name)
    _write_run_manifest("e8_fault_injection", cfg, config, profile)
    _run_experiment("e8_fault_injection", ds=ds, profile=profile, config=config)


@app.command("e9")
def e9(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    """E9 – Uncertainty method comparison."""
    cfg = _common_options(ds, profile, config)
    logger.info("e9_uncertainty_comparison: ds=%s", cfg.dataset.name)
    _write_run_manifest("e9_uncertainty_comparison", cfg, config, profile)
    _run_experiment("e9_uncertainty_comparison", ds=ds, profile=profile, config=config)


@app.command("app-data")
def app_data(
    ds: str = typer.Option("FD001", "--ds"),
    profile: str = typer.Option("fast", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
    machine: str = typer.Option("MACHINE-001", "--machine"),
) -> None:
    """Build dashboard artefacts from C-MAPSS (SHI, forecast, DQ, guidance)."""
    from .app_data import build_artifacts

    cfg = _common_options(ds, profile, config)
    logger.info("app-data: ds=%s machine=%s", cfg.dataset.name, machine)
    written = build_artifacts(cfg.dataset.name, machine)
    _write_run_manifest("app_data", cfg, config, profile)
    logger.info("app-data: wrote %d artifacts", len(written))


@app.command("app")
def launch_app(
    port: int = typer.Option(8501, "--port", help="Streamlit server port."),
) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    logger.info("Launching Streamlit dashboard on port %d", port)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/stattwin/dashboard/app.py",
            "--server.themeBase",
            "dark",
            "--server.port",
            str(port),
        ],
        check=True,
    )


@app.command("test")
def run_test() -> None:
    """Run the test suite via pytest."""
    import subprocess
    import sys

    logger.info("Running test suite")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        check=True,
    )


@app.command("reproduce")
def reproduce(
    profile: str = typer.Option("full", "--profile"),
    config: str = typer.Option("configs/base.yaml", "--config"),
    ds: str = typer.Option("FD001", "--ds"),
    with_dashboard: bool = typer.Option(
        True, "--with-dashboard/--skip-dashboard",
        help="Also rebuild dashboard artifacts (app-data).",
    ),
) -> None:
    """Full end-to-end pipeline: e0→e5, e8, app-data."""
    cfg = _common_options(ds, profile, config)
    logger.info("reproduce: profile=%s ds=%s", profile, cfg.dataset.name)
    steps = [
        "e0_data_audit",
        "e1_health_index",
        "e2_model_comparison",
        "e3_early_warning",
        "e4_ablation",
        "e5_uncertainty",
        "e8_fault_injection",
    ]
    for step in steps:
        logger.info("reproduce step: %s", step)
        _write_run_manifest(step, cfg, config, profile)
        _run_experiment(step, ds=ds, profile=profile, config=config)
    if with_dashboard:
        logger.info("reproduce step: app-data")
        from .app_data import build_artifacts

        build_artifacts(ds, "MACHINE-001")
        _write_run_manifest("app_data", cfg, config, profile)
    logger.info("reproduce: complete")


if __name__ == "__main__":
    app()
