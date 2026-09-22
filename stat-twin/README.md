# STAT-TWIN

**Statistical Digital Twin for Probabilistic Failure Forecasting and Predictive Maintenance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-red.svg)](https://docs.pytest.org/)

> A research-oriented predictive-maintenance platform that learns the statistical behaviour of machines, detects degradation, forecasts future failure probability, estimates Remaining Useful Life (RUL), quantifies uncertainty, and explains why risk is increasing.

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Most predictive-maintenance projects give you a number. STAT-TWIN gives you a *reason and a confidence*. It builds a statistical model of normal behaviour for each engine, tracks how the engine drifts away from it, forecasts failure probability at 10 to 50 cycles ahead with calibrated uncertainty, and tells you which sensors and which statistics drove the risk.

The system is built as a **statistical digital twin**: it shows its evidence, states how sure it is, and tests its own claims. Every metric comes from measured artifacts, every prediction includes uncertainty quantification, and every explanation is grounded in statistical evidence.

**Primary use cases:**
- College research project / ML-statistics demonstration
- Research paper implementation
- Technical presentation / viva preparation
- Future industrial expansion (dataset-agnostic architecture)

---

## Research Question

> Does combining statistical health indicators with temporal ML improve early failure detection, RUL estimation, and prediction reliability compared with threshold-based and ML-only approaches?

**Hypotheses:**
- **H0:** Statistical health indicators do not improve predictive performance.
- **H1:** Statistical health indicators improve predictive performance.

H1 is tested using a leakage-safe ablation (variants A-E) with paired Wilcoxon signed-rank tests, bootstrap confidence intervals, and Holm-Bonferroni correction.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Leakage Guard** | Automated test suite proving no data leakage (truncation invariance, shuffled-label sanity, unit-disjointness) |
| **Statistical Health Index (SHI)** | 0-100 health indicator combining deviation, trend, EWMA, variance, and correlation shift evidence |
| **7 Model Baselines** | Fixed threshold, Mahalanobis anomaly, Logistic Regression, Random Forest, XGBoost, GRU, and STAT-TWIN Hybrid |
| **Multi-Horizon Forecasting** | P(failure within h) for h in {10, 20, 30, 40, 50} with monotone enforcement |
| **Conformal RUL Intervals** | Distribution-free prediction intervals with empirical coverage guarantees |
| **Probability Calibration** | Isotonic/Platt calibration with reliability diagrams, Brier score, and ECE |
| **Explainability** | Evidence cards, group occlusion attribution, and risk-change decomposition |
| **What-If Simulator** | Perturb sensor values, recompute features, re-predict (labelled as simulation) |
| **Ablation Study** | 5-variant ablation (A-E) with statistical significance testing |
| **Live Replay** | Dashboard "ages" an engine cycle by cycle with provenance badges |
| **Fault-Injection Benchmark** | Separates sensor faults from real degradation |

---

## Architecture

```
Raw Telemetry (C-MAPSS)
    |
    v
Data Ingestion + Unit-Level Split (GroupKFold, no leakage)
    |
    v
Preprocessing (missing values, outliers, scaling, DQ flags)
    |
    v
Statistical Engine (rolling stats, EWMA, trend, correlation, shift)
    |
    v
Health Engine (SHI 0-100, 5 health states with persistence)
    |
    v
ML Layer (7 model families, common fit/predict interface)
    |
    v
Forecasting (multi-horizon probabilities, RUL, failure point)
    |
    v
Uncertainty + Calibration (conformal intervals, isotonic calibration)
    |
    v
Explainability (evidence cards, group occlusion, risk decomposition)
    |
    v
What-If Simulator (perturbation -> recompute -> re-predict)
    |
    v
Dashboard (7-page Streamlit app, live 2s auto-refresh)
```

**Design Principles:**
1. Research validity beats adding technologies
2. Prevent data leakage at every stage
3. Every metric has a defined evaluation protocol
4. Never fabricate performance numbers
5. Statistical modelling is a core component, not decoration
6. Every important prediction is explainable
7. Experiments are reproducible

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- NASA C-MAPSS dataset (download from [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/))

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/STAT-TWIN.git
cd STAT-TWIN/stat-twin

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Or use Makefile
make setup
```

### Data Acquisition

1. Download the C-MAPSS archive from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
2. Unzip into `data/raw/CMAPSS/`
3. Expected files: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (and FD002-FD004)

### Quick Start

```bash
# Run the full pipeline on FD001 (fast profile)
make data DS=FD001
make features DS=FD001
make train DS=FD001 MODEL=xgb
make eval DS=FD001

# Run experiments
make e0  # Data audit
make e1  # Health index validation
make e2  # Model comparison

# Launch dashboard
make app
```

---

## Project Structure

```
stat-twin/
|-- AGENTS.md                      # AI agent rules
|-- Makefile                       # Build targets
|-- pyproject.toml                 # Package + tool config
|-- .streamlit/config.toml         # Dashboard theme
|
|-- data/
|   |-- raw/CMAPSS/               # NASA C-MAPSS files (gitignored)
|   +-- processed/                 # Parquet caches
|
|-- configs/
|   |-- base.yaml                 # Master configuration
|   |-- profiles/                 # smoke, fast, full
|   |-- datasets/                 # fd001-fd004
|   +-- experiments/              # ablation, generalization, fault_injection
|
|-- src/stattwin/
|   |-- cli.py                    # Typer CLI: stattwin <command>
|   |-- config.py                 # Pydantic config loader
|   |-- manifest.py               # Run manifest writer
|   |
|   |-- data/                     # Ingestion, schema, splitting, synthetic
|   |-- preprocessing/            # Missing, outliers, scalers, DQ engine
|   |-- statistics/               # Rolling, EWMA, trend, correlation, shift
|   |-- health/                   # SHI, states, quality metrics
|   |-- models/                   # 7 model families + trainer
|   |-- forecasting/              # Probability curves, RUL, failure point
|   |-- uncertainty/              # Conformal, calibration, Brier, ECE
|   |-- explainability/           # Evidence cards, occlusion, SHAP
|   |-- counterfactual/           # What-if simulator
|   |-- evaluation/               # Metrics, lead-time, significance tests
|   |-- decision/                 # Maintenance guidance
|   |-- dashboard/                # Streamlit app + pages + components
|   |-- experiments/              # Runnable experiments E0-E9
|   +-- reports/                  # Report generation
|
|-- tests/                        # Unit tests + leakage guard
|-- results/                      # Metrics JSON, figures, tables
|-- artifacts/                    # Models, features (gitignored)
+-- docs/                         # MASTERPLAN.md, VIVA.md, etc.
```

---

## Methodology

### Statistical Health Index (SHI)

The SHI is a 0-100 health indicator combining five evidence components:

| Component | Signal |
|-----------|--------|
| `deviation` | Signed z-score moved in the sensor's learned degradation direction |
| `trend` | Signed slope over window, normalized by baseline sigma |
| `ewma` | EWMA deviation from baseline |
| `variance` | Log variance ratio vs baseline |
| `corr_shift` | Correlation-structure change (Frobenius norm) |

Each component is normalized to [0, 1] using a train-fitted ECDF mapping. Sensor aggregation uses informativeness weights learned on training data.

### Health States

Five states with hysteresis (persistence of p cycles to change):
- **HEALTHY** - Normal operation
- **WATCH** - Early signs of deviation
- **DEGRADING** - Confirmed degradation trend
- **CRITICAL** - High failure probability
- **FAILURE-LIKELY** - Imminent failure

### Model Interface

Every learner implements:
```python
fit(X_train, y_train, groups=None) -> self
predict_proba(X) -> ndarray[n, n_horizons]  # P(fail within h)
predict_rul(X) -> ndarray[n]
score_raw(X) -> ndarray[n]                  # continuous risk score
```

### Uncertainty Quantification

**Primary method: Split conformal prediction** on the ensemble mean with normalized residuals.

- Distribution-free finite-sample coverage under exchangeability
- Nonconformity score: `s = |y - y_hat| / (sigma_ens + epsilon)`
- 90% prediction intervals by default
- Coverage measured by RUL bucket and under distribution shift

### Ablation Design

| Variant | Features |
|---------|----------|
| A | Raw sensors only |
| B | Raw + statistical features |
| C | Raw + temporal features |
| D | Raw + statistical + temporal |
| E | Full STAT-TWIN (D + health + uncertainty) |

Statistical significance tested via paired Wilcoxon signed-rank with Holm-Bonferroni correction.

---

## Experiments

| ID | Name | Question |
|----|------|----------|
| E0 | Data audit | Is the data as expected? |
| E1 | Health-index validation | Is the SHI a good health indicator? |
| E2 | Model comparison | How do the 7 models compare? |
| E3 | Early-warning benchmark | Who warns earlier at the same false-alarm budget? |
| E4 | **Ablation (A-E)** + significance | Do statistical features add measurable value? |
| E5 | Uncertainty and calibration | Are probabilities and intervals trustworthy? |
| E6 | Generalization matrix | How does performance change across datasets? |
| E7 | Operating-condition study | Does regime normalization matter? |
| E8 | Fault-injection benchmark | Can the system tell sensor faults from degradation? |
| E9 | Uncertainty method comparison | Which uncertainty method is best? |

**Run experiments:**
```bash
make e0  # through e9
```

---

## Dashboard

A 7-page Streamlit dashboard with dark industrial theme, live 2s auto-refresh, and provenance badges.

| Page | Content |
|------|---------|
| **Overview** | Machine health at a glance: SHI gauge, risk timeline, DQ status, recommendations, Live Replay |
| **Sensor Monitoring** | Interactive sensor selector, raw + rolling + EWMA + Z-score, DQ flag markers |
| **Statistical Health** | SHI trajectory, Z-score heatmap, variance-change, distribution-shift table, correlation matrix |
| **Failure Forecast** | Multi-horizon probability curve, RUL with conformal interval, warning timeline |
| **Explainability** | Evidence cards, sensor contributions, risk-change decomposition |
| **What-If Simulator** | Sensor sliders, original vs simulated comparison, SIMULATION disclaimer |
| **Model Comparison** | Metrics table, early warning, ablation forest plot, calibration, generalization heatmap |

**Launch:**
```bash
make app
# or
streamlit run src/stattwin/dashboard/app.py
```

**Design system:**
- Dark theme (#0A0E17 background, #111827 cards)
- Provenance palette: OBSERVED (blue), PREDICTED (amber), SIMULATED (violet)
- Health state palette: HEALTHY (green), WATCH (yellow), DEGRADING (orange), CRITICAL (red)
- Monospace numerics for readings
- Responsive layout with mobile breakpoints

---

## Testing

### Leakage Guard

Proves no data leakage in the pipeline:

| Test | What it proves |
|------|----------------|
| `test_unit_disjoint_splits` | No unit_id in multiple splits within a fold |
| `test_fit_only_on_train` | Preprocessors fit on training portions only |
| `test_feature_causality` | Features at time t use cycles <= t only |
| `test_no_label_in_features` | No feature derived from RUL |
| `test_shuffled_label_sanity` | Shuffled labels yield AUC ~ 0.5 |
| `test_baseline_uses_past_only` | Healthy baseline uses only first K cycles |
| `test_calibration_disjoint` | Calibration data disjoint from test |
| `test_determinism` | Same seed -> identical results |
| `test_threshold_selected_on_val` | Warning thresholds never fitted on test |

**Run:**
```bash
make test              # Full test suite
make test-leakage      # Leakage guard only
```

---

## Reproducibility

Every run writes a `manifest.json` with:
- Git commit hash
- Config hash
- Random seeds (Python, NumPy, PyTorch, XGBoost)
- Package versions
- Platform info

**Reproduce full results:**
```bash
make reproduce         # Full profile, all datasets
```

**Configuration profiles:**

| Profile | Datasets | Folds | Seeds | Purpose |
|---------|----------|-------|-------|---------|
| `smoke` | synthetic | 2 | 1 | CI and unit tests |
| `fast` | FD001 | 3 | 1 | Development |
| `full` | FD001-FD004 | 5 | 5 | Final results |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stattwin2026,
  title = {STAT-TWIN: Statistical Digital Twin for Probabilistic Failure Forecasting},
  year = {2026},
  url = {https://github.com/your-username/STAT-TWIN}
}
```

---

## Acknowledgments

- NASA Ames Prognostics Data Repository for the C-MAPSS dataset
- Saxena et al. (2008) for the turbofan degradation simulation
- Vovk et al. and Lei et al. (2018) for conformal/distribution-free inference
- Coble (2010) for health-indicator quality metrics
- Lundberg and Lee (2017) for SHAP

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Limitations

Openly stated:
- C-MAPSS is a simulated dataset with few fault modes
- Conformal guarantees are marginal and assume exchangeability
- State thresholds are conventions unless calibrated
- What-If simulator shows model sensitivity, not causal intervention
- Hybrid gains may be dataset-specific
- Results depend on RUL clipping and horizon choices
