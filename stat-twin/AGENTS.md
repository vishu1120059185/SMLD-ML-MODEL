# AGENTS.md — STAT-TWIN

## Mission
Implement STAT-TWIN exactly as specified in docs/MASTERPLAN.md. Section 1 of the plan is locked.

## Working rules
1. Work on ONE phase at a time (Section 10). Do not start later phases.
2. Read the relevant Section 6 module specs before writing code. Ask before deviating.
3. Never fabricate metrics, results, dataset facts, or citations. All numbers in reports,
   README tables, and the dashboard must come from files in results/ or artifacts/.
4. If data files are missing in data/raw/CMAPSS/, fail loudly with instructions. Never
   substitute synthetic data outside tests and the `smoke` profile.
5. Prevent leakage: unit-level splits only; fit every transform, threshold, calibrator, and
   selector on training portions only; features must be causal (use rows <= t only);
   never derive features from RUL.
6. All science code is vectorized (pandas/NumPy). No per-row Python loops over cycles.
7. Everything is config-driven (configs/*.yaml, pydantic). No magic numbers in code.
8. Set seeds; write a manifest.json (git hash, config hash, seeds, versions) for every run.
9. Every public function has type hints and a docstring; every module has tests.
10. Run `make test` before declaring a phase done. Leakage Guard tests must pass.
11. The What-If simulator and any counterfactual output must show the SIMULATION label and
    disclaimer. Never describe simulated results as real outcomes.
12. Explanations say "contributed to the model's risk estimate", never "caused failure".
13. Keep the dashboard read-only over artifacts, except What-If inference.

## Commands (Makefile targets)
setup, data, features, train, eval, e0..e9, report, app-data, app, test, reproduce

## Stack
Python 3.11+, pandas, numpy, scipy, scikit-learn, xgboost, torch (CPU ok), streamlit,
plotly, pydantic, typer, pytest, ruff, pyarrow, shap (optional).

## Definition of done (per phase)
Tests pass, Leakage Guard passes, artifacts follow docs/MASTERPLAN.md Section 4.6,
manifest written, short summary of DoD pass/fail reported.
