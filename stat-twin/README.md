# STAT-TWIN

**Statistical Digital Twin for Probabilistic Failure Forecasting and Predictive Maintenance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 205 passed](https://img.shields.io/badge/tests-205%20passed-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![Streamlit Dashboard](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](#dashboard)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

> A research-oriented predictive-maintenance platform that learns the statistical behaviour of machines, detects degradation, forecasts failure probability, estimates Remaining Useful Life (RUL), quantifies uncertainty, and explains why risk is increasing.

---

**Authors:** [Vishwesh Penkar](https://github.com/vishu1120059185) · [Daksh Patil](https://github.com/vishu1120059185)
**Under the guidance of:** Rashmi Sartkar

**Repository:** [https://github.com/vishu1120059185/SMLD-ML-MODEL](https://github.com/vishu1120059185/SMLD-ML-MODEL)

---

## Table of Contents

- [Overview](#overview)
- [Authors & Guidance](#authors--guidance)
- [Research Question](#research-question)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

Most predictive-maintenance projects give you a number. STAT-TWIN gives you a *reason and a confidence*. It builds a statistical model of normal behaviour for each engine, tracks how the engine drifts away from it, forecasts failure probability 10 to 50 cycles ahead with calibrated uncertainty, and tells you which sensors and which statistics drove the risk.

The system is built as a **statistical digital twin**: it shows its evidence, states how sure it is, and tests its own claims. Every number in this README comes from measured artifacts in [`results/`](results/), every prediction includes uncertainty quantification, and every explanation is grounded in statistical evidence.

**Primary use cases:**
- College research project / ML-statistics demonstration
- Research paper implementation
- Technical presentation / viva preparation
- Future industrial expansion (dataset-agnostic architecture)

---

## Authors & Guidance

| Role | Name |
|------|------|
| **Author** | **Vishwesh Penkar** |
| **Author** | **Daksh Patil** |
| **Project Guidance** | **Rashmi Sartkar** |

Developed as part of the SMLD (Statistical Machine Learning & Data) project at AIML. All design decisions, experiment protocols, and implementation follow the locked master plan in `docs/` and the working rules in [`AGENTS.md`](AGENTS.md).

---

## Research Question

> Does combining statistical health indicators with temporal ML improve early failure detection, RUL estimation, and prediction reliability compared with threshold-based and ML-only approaches?

**Hypotheses:**
- **H0:** Statistical health indicators do not improve predictive performance.
- **H1:** Statistical health indicators improve predictive performance.

H1 is tested using a leakage-safe ablation (variants A–E) with paired Wilcoxon signed-rank tests, bootstrap confidence intervals, and Holm–Bonferroni correction.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Leakage Guard** | 43 automated tests proving no data leakage (truncation invariance, shuffled-label sanity, unit-disjointness, causality) |
| **Statistical Health Index (SHI)** | 0–100 health indicator combining deviation, trend, EWMA, variance, and correlation-shift evidence |
| **7 Model Baselines** | Fixed threshold, Mahalanobis anomaly, Logistic Regression, Random Forest, XGBoost, GRU, and STAT-TWIN Hybrid |
| **Multi-Horizon Forecasting** | P(failure within h) for h ∈ {10, 20, 30, 40, 50} with monotone enforcement |
| **Conformal RUL Intervals** | Distribution-free prediction intervals with empirical coverage guarantees |
| **Probability Calibration** | Isotonic/Platt calibration with reliability diagrams, Brier score, and ECE |
| **Explainability** | Evidence cards, group-occlusion attribution, and risk-change decomposition |
| **What-If Simulator** | Perturb sensor values → recompute features → re-predict (always labelled SIMULATION) |
| **Ablation Study** | 5-variant ablation (A–E) with statistical significance testing |
| **Live Replay** | Dashboard ages an engine cycle by cycle with provenance badges (OBSERVED / PREDICTED / SIMULATED) |
| **Fault-Injection Benchmark** | Separates sensor faults from real degradation via data-quality gating |
| **Reproducible Runs** | Every run writes `manifest.json` (git hash, config hash, seeds, versions) |

---

## Architecture

```
Raw Telemetry (NASA C-MAPSS FD001–FD004)
    │
    ▼
Data Ingestion + Unit-Level Split (GroupKFold — no leakage)
    │
    ▼
Preprocessing (missing values, outliers, scaling, DQ flags)
    │
    ▼
Statistical Engine (rolling stats, EWMA, trend, correlation, shift)
    │
    ▼
Health Engine (SHI 0–100, 5 health states with hysteresis)
    │
    ▼
ML Layer (7 model families, common fit/predict interface)
    │
    ▼
Forecasting (multi-horizon probabilities, RUL, failure point)
    │
    ▼
Uncertainty + Calibration (conformal intervals, isotonic calibration)
    │
    ▼
Explainability (evidence cards, group occlusion, risk decomposition)
    │
    ▼
What-If Simulator (perturbation → recompute → re-predict)
    │
    ▼
Dashboard (7-page Streamlit app, live 2 s auto-refresh)
```

**Design Principles:**
1. Research validity beats adding technologies.
2. Prevent data leakage at every stage.
3. Every metric has a defined evaluation protocol.
4. Never fabricate performance numbers.
5. Statistical modelling is a core component, not decoration.
6. Every important prediction is explainable.
7. Experiments are reproducible from a single command.

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- NASA C-MAPSS dataset in `data/raw/CMAPSS/` (see [Data Acquisition](#data-acquisition))

### Installation

```bash
# Clone the repository
git clone https://github.com/vishu1120059185/SMLD-ML-MODEL.git
cd SMLD-ML-MODEL/stat-twin

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies + lint
make setup                       # or: pip install -e ".[dev]"
```

### Data Acquisition

1. Download the C-MAPSS archive from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (mirror: [Zenodo 15346912](https://zenodo.org/records/15346912)).
2. Unzip into `data/raw/CMAPSS/`.
3. Expected files: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (and FD002–FD004).

> If the files are missing the pipeline **fails loudly** with instructions — synthetic data is never substituted outside tests and the `smoke` profile.

### Quick Start

```bash
# Full pipeline on FD001 (fast profile)
make data      DS=FD001
make features  DS=FD001
make train     DS=FD001 MODEL=xgb
make eval      DS=FD001

# Experiments (E0–E9)
make e0        # data audit
make e1        # health-index validation
make e2        # model comparison
make e5        # uncertainty & calibration
make e8        # fault-injection benchmark

# Dashboard
make app       # → http://localhost:8501

# Tests
make test
```

---

## Project Structure

```
stat-twin/
├── AGENTS.md                      # Working rules (leakage, provenance, labels)
├── Makefile                       # setup · data · features · train · eval · e0–e9 · app · test
├── pyproject.toml                 # Package + ruff/pytest config
├── README.md                      # This file
├── .streamlit/config.toml         # Dark industrial theme (port 8501)
│
├── data/
│   ├── raw/CMAPSS/               # NASA C-MAPSS files (gitignored)
│   └── processed/                 # Parquet caches
│
├── configs/
│   ├── base.yaml                  # Master configuration (seed 42, folds, horizons)
│   ├── profiles/                  # smoke · fast · full
│   ├── datasets/                  # fd001–fd004
│   └── experiments/               # ablation · generalization · fault_injection
│
├── src/stattwin/
│   ├── cli.py                     # Typer CLI: python -m stattwin.cli <cmd>
│   ├── config.py                  # Pydantic config loader
│   ├── manifest.py                # Run manifest writer
│   ├── data/                      # Ingestion, schema, splitting, synthetic
│   ├── preprocessing/             # Missing, outliers, scalers, DQ engine
│   ├── statistics/                # Rolling, EWMA, trend, correlation, shift
│   ├── health/                    # SHI, states, quality metrics
│   ├── models/                    # 7 model families + trainer
│   ├── forecasting/               # Probability curves, RUL, failure point
│   ├── uncertainty/               # Conformal, calibration, Brier, ECE
│   ├── explainability/            # Evidence cards, occlusion, attribution
│   ├── counterfactual/            # What-If simulator
│   ├── evaluation/                # Metrics, lead-time, significance tests
│   ├── decision/                  # Maintenance guidance (wired into Overview)
│   ├── dashboard/                 # Streamlit app + 7 views + components
│   │   ├── app.py                 # Entry point (sidebar nav, hero, dispatch)
│   │   ├── components/            # theme.py · cards.py · charts.py · live.py
│   │   └── views/                 # 1_overview … 7_comparison
│   ├── experiments/               # Runnable experiments E0–E9
│   └── reports/                   # Report generation
│
├── tests/                         # 205 tests (unit + leakage guard)
│   └── leakage_guard/             # 43 leakage-specific proofs
│
├── results/                       # Experiment artifacts (JSON + figures)
│   ├── e0_data_audit/
│   ├── e1_health_index/
│   ├── e2_model_comparison/
│   ├── e5_uncertainty/
│   └── e8_fault_injection/
│
├── artifacts/                     # Models, features (gitignored)
└── docs/                          # Master plan & viva materials
```

---

## Methodology

### Statistical Health Index (SHI)

The SHI is a 0–100 health indicator combining five evidence components:

| Component | Signal |
|-----------|--------|
| `deviation` | Signed z-score moved in the sensor's learned degradation direction |
| `trend` | Signed slope over window, normalized by baseline sigma |
| `ewma` | EWMA deviation from baseline |
| `variance` | Log variance ratio vs baseline |
| `corr_shift` | Correlation-structure change (Frobenius norm) |

Each component is normalized to [0, 1] using a **train-fitted ECDF** mapping. Sensor aggregation uses informativeness weights learned on training data only.

### Health States

Five states with hysteresis (persistence of *p* cycles to change):

| State | Meaning |
|-------|---------|
| **HEALTHY** | Normal operation |
| **WATCH** | Early signs of deviation |
| **DEGRADING** | Confirmed degradation trend |
| **CRITICAL** | High failure probability |
| **FAILURE-LIKELY** | Imminent failure |

### Model Interface

Every learner implements:

```python
fit(X_train, y_train, groups=None) -> self
predict_proba(X) -> ndarray[n, n_horizons]   # P(fail within h)
predict_rul(X)   -> ndarray[n]
score_raw(X)     -> ndarray[n]               # continuous risk score
```

### Uncertainty Quantification

**Primary method: split conformal prediction** on the ensemble mean with normalized residuals.

- Distribution-free finite-sample coverage under exchangeability
- Nonconformity score: `s = |y − ŷ| / (σ_ens + ε)`
- 90 % prediction intervals by default (α = 0.10)
- Coverage measured by RUL bucket and under distribution shift

### Ablation Design

| Variant | Features |
|---------|----------|
| A | Raw sensors only |
| B | Raw + statistical features |
| C | Raw + temporal features |
| D | Raw + statistical + temporal |
| E | Full STAT-TWIN (D + health + uncertainty) |

Statistical significance: paired Wilcoxon signed-rank with Holm–Bonferroni correction.

---

## Experiments & Results

All numbers below are read directly from `results/*/*_results.json` — nothing is fabricated.

| ID | Name | Question | Status |
|----|------|----------|--------|
| E0 | Data audit | Is the data as expected? | ✅ run |
| E1 | Health-index validation | Is the SHI a good health indicator? | ✅ run |
| E2 | Model comparison | How do the models compare across horizons? | ✅ run (2 folds) |
| E3 | Early-warning benchmark | Who warns earlier at the same false-alarm budget? | 🔧 scripted |
| E4 | **Ablation (A–E) + significance** | Do statistical features add measurable value? | 🔧 scripted |
| E5 | Uncertainty & calibration | Are probabilities and intervals trustworthy? | ✅ run |
| E6 | Generalization matrix | How does performance change across datasets? | 🔧 scripted |
| E7 | Operating-condition study | Does regime normalization matter? | 🔧 scripted |
| E8 | Fault-injection benchmark | Can the system tell sensor faults from degradation? | ✅ run |
| E9 | Uncertainty method comparison | Which uncertainty method is best? | 🔧 scripted |

Run any experiment with `make e0` … `make e9`.

### E0 — Data Audit (FD001)

| Metric | Value |
|--------|-------|
| Units | 100 |
| Rows | 20,631 |
| Sensors | 21 |
| Cycles per unit (min / mean / max) | 1 / 206.31 / 362 |

### E1 — Health-Index Validation

| Metric | Value |
|--------|-------|
| SHI range | 14.30 → 70.94 |
| Monotonicity (mean) | 1.000 |
| Trendability | 1.000 |
| Prognosability | 2.311 |
| Spearman ρ (mean ± std) | 0.825 ± 0.053 |
| Elapsed | 23.41 s |

State distribution across cycles: HEALTHY 4 · WATCH 4,373 · DEGRADING 16,171 · CRITICAL 83.

### E2 — Model Comparison (XGBoost, 2 folds)

| Horizon | AUC (mean) | RUL MAE (mean) |
|---------|-----------:|---------------:|
| h = 10 | 0.8900 | 32.41 |
| h = 20 | 0.9263 | 32.41 |
| h = 30 | 0.9396 | 32.41 |
| h = 40 | 0.9469 | 32.41 |
| h = 50 | 0.9506 | 32.41 |

Elapsed: 4.14 s · source: `results/e2_model_comparison/e2_results.json`.

### E5 — Uncertainty & Calibration

| Horizon | Brier ↓ | ECE ↓ |
|---------|--------:|------:|
| h = 10 | 0.0175 | 0.0174 |
| h = 20 | 0.0258 | 0.0251 |
| h = 30 | 0.0390 | 0.0370 |
| h = 40 | 0.0476 | 0.0413 |
| h = 50 | 0.0582 | 0.0479 |

**Conformal prediction (90 % target):** empirical coverage **0.9005**, mean interval width **131.54** cycles.

### E8 — Fault-Injection Benchmark

| Fault | Precision | Recall |
|-------|----------:|-------:|
| Spike | 0.92 | 0.88 |
| Stuck | 0.95 | 0.91 |
| Dropout | 0.97 | 0.94 |
| Noise | 0.78 | 0.72 |

**False-alarm rate:** 0.15 without DQ gating → **0.04 with gating (73.3 % reduction)**.

---

## Dashboard

A **7-page Streamlit dashboard** with a dark industrial theme, live 2-second auto-refresh, and provenance badges.

| # | Page | Content |
|---|------|---------|
| 01 | **Overview** | SHI gauge, risk timeline, DQ status, recommendations, RUL countdown |
| 02 | **Sensor Monitoring** | Interactive sensor selector, raw + rolling + EWMA + z-score, DQ flags |
| 03 | **Statistical Health** | SHI trajectory, z-score heatmap, variance change, distribution shift |
| 04 | **Failure Forecast** | Multi-horizon probability curve, RUL with conformal interval |
| 05 | **Explainability** | Evidence cards, sensor contributions, risk-change decomposition |
| 06 | **What-If Simulator** | Sensor sliders, original → simulated, SIMULATION disclaimer |
| 07 | **Model Comparison** | Real e2/e5 metrics (AUC, Brier, ECE), calibration, conformal coverage |

**Build dashboard data first** (writes real C-MAPSS artifacts under `results/`):

```bash
make app-data
# or
python -m stattwin.app_data --ds FD001 --machine MACHINE-001
```

**Launch:**

```bash
make app
# or
python -m streamlit run src/stattwin/dashboard/app.py --server.headless true --server.port 8501
```

Then open **http://localhost:8501**. Without `app-data`, views fall back to clearly labelled demo streams (SIMULATED).

**Design system** (`src/stattwin/dashboard/components/theme.py`):

- **Theme:** dark industrial — background `#0A0E17`, cards `#111827`, border `#1F2937`, accent `#3B82F6`
- **Provenance palette:** OBSERVED (blue, solid) · PREDICTED (amber, dashed) · SIMULATED (violet, dotted)
- **Health palette:** HEALTHY (green) · WATCH (yellow) · DEGRADING (orange) · CRITICAL (red) · FAILURE-LIKELY (dark red)
- **Typography:** Inter for UI, JetBrains Mono for numerics
- **Live bar:** blinking LIVE badge, freshness indicator, tick counter, feed label
- Responsive layout with mobile breakpoints and reduced-motion support

---

## Testing

**205 tests, all passing** — including the Leakage Guard suite.

```bash
make test              # full suite (205 tests)
make test-leakage      # leakage guard only
```

### Leakage Guard

Proves no data leakage in the pipeline:

| Test | What it proves |
|------|----------------|
| `test_disjoint_splits` | No `unit_id` in multiple splits within a fold |
| `test_fit_only_on_train` | Preprocessors fit on training portions only |
| `test_feature_causality` | Features at time *t* use cycles ≤ *t* only |
| `test_no_label_in_features` | No feature derived from RUL |
| `test_shuffled_label_sanity` | Shuffled labels yield AUC ≈ 0.5 |
| `test_baseline_uses_past_only` | Healthy baseline uses only first *K* cycles |
| `test_calibration_disjoint` | Calibration data disjoint from test |
| `test_determinism` | Same seed → identical results |
| `test_threshold_selected_on_val` | Warning thresholds never fitted on test |

### Lint

```bash
python -m ruff check src/ tests/     # E,F,W,I,UP,B,SIM — line length 100
```

---

## Reproducibility

Every run writes a `manifest.json` containing:

- Git commit hash
- Config hash
- Random seeds (Python, NumPy, PyTorch, XGBoost) — default **seed 42**
- Package versions and platform info

**Configuration profiles:**

| Profile | Datasets | Folds | Purpose |
|---------|----------|-------|---------|
| `smoke` | synthetic | 2 | CI and unit tests |
| `fast` | FD001 | 2–3 | Development (default) |
| `full` | FD001–FD004 | 5 | Final results |

**Reproduce everything:**

```bash
make reproduce         # full profile, all datasets
```

---

## Roadmap

Remaining experiment scripts are implemented and run via `make e3`, `e4`, `e6`, `e7`, `e9`:

- **E3** — Early-warning benchmark at fixed false-alarm budget
- **E4** — Ablation A–E with Wilcoxon + Holm–Bonferroni
- **E6** — Cross-dataset generalization matrix (FD001–FD004)
- **E7** — Operating-condition / regime-normalization study
- **E9** — Conformal vs bootstrap vs ensemble uncertainty comparison

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{penkar_patil_stat_twin_2026,
  title     = {STAT-TWIN: Statistical Digital Twin for Probabilistic
               Failure Forecasting and Predictive Maintenance},
  authors   = {Penkar, Vishwesh and Patil, Daksh},
  year      = {2026},
  note      = {Under the guidance of Rashmi Sartkar},
  url       = {https://github.com/vishu1120059185/SMLD-ML-MODEL}
}
```

---

## Acknowledgments

- **Rashmi Sartkar** — project guidance and review
- NASA Ames Prognostics Data Repository for the C-MAPSS dataset
- Saxena et al. (2008) for the turbofan degradation simulation
- Vovk et al. and Lei et al. (2018) for conformal / distribution-free inference
- Coble (2010) for health-indicator quality metrics
- Lundberg and Lee (2017) for SHAP
- Streamlit, Plotly, scikit-learn, XGBoost, and PyTorch communities

---

## Limitations

Openly stated:

- C-MAPSS is a simulated dataset with few fault modes.
- Conformal guarantees are marginal and assume exchangeability.
- State thresholds are conventions unless calibrated per fleet.
- What-If simulator shows **model sensitivity**, not causal intervention — output is always labelled SIMULATION.
- Hybrid-model gains may be dataset-specific.
- Results depend on RUL clipping and horizon choices.
- E2 figures above come from the `fast` profile (2 folds); final numbers require `make reproduce` with the `full` profile.

---

## License

This project is licensed under the **MIT License**.

---

<div align="center">

**Vishwesh Penkar · Daksh Patil**  
*Guidance: Rashmi Sartkar*  
AIML — SMLD ML MODEL

[![GitHub](https://img.shields.io/badge/GitHub-SMLD--ML--MODEL-181717?logo=github)](https://github.com/vishu1120059185/SMLD-ML-MODEL)

</div>
