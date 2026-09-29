# STAT-TWIN

**Statistical Digital Twin for Probabilistic Failure Forecasting**

A degradation-aware predictive-maintenance system for turbofan engines. It builds a
per-unit *statistical* health trajectory, converts it into calibrated failure
probabilities and RUL intervals, and explains every number it shows — with the
provenance of each value stated on screen.

Built on the NASA C-MAPSS degradation dataset (FD001–FD004).

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Results](#results)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Dashboard](#dashboard)
- [Engineering guarantees](#engineering-guarantees)
- [Performance](#performance)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation and attribution](#citation-and-attribution)

---

## What it does

| Capability | Output | Provenance |
|---|---|---|
| **Data audit** | Completeness, plausibility, constant sensors, operating regimes | `OBSERVED` |
| **Health index** | Statistical Health Index (SHI) 0–1 with per-sensor evidence weights | `OBSERVED` |
| **Failure forecast** | P(failure) at 5 horizons + point RUL with conformal interval | `PREDICTED` |
| **Model comparison** | 8 models scored on identical unit-disjoint folds | `PREDICTED` |
| **Early warning** | Lead time at a fixed false-alarm budget | `PREDICTED` |
| **Uncertainty** | 4 interval methods compared on held-out coverage | `PREDICTED` |
| **Explainability** | Per-sensor contribution to the risk estimate | `PREDICTED` |
| **What-if** | Counterfactual sensor perturbation sensitivity | `SIMULATED` |

> Every value in the dashboard carries one of three badges. `SIMULATED` output is
> never presented as a real outcome, and explanations say a feature *contributed
> to the model's risk estimate* — never that it *caused* a failure.

---

## Quick start

**Requirements:** Python 3.11+, ~8 GB RAM (no GPU required — everything runs on CPU).

```bash
cd stat-twin
pip install -e ".[dev]"

# 1. validate the dataset
python -m stattwin.cli data --ds FD001

# 2. build the dashboard artifacts
python -m stattwin.app_data --ds FD001 --machine MACHINE-001

# 3. launch the dashboard
make app            #  ->  http://localhost:8501
```

Or drive the whole pipeline through the Makefile:

```bash
make data features train eval   # core pipeline
make e0 e1 e2 e3 e4 e5 e6 e7 e8 e9   # experiments
make report                     # regenerate results tables
make test                       # full test suite
```

`make` targets accept `PROFILE` and `DS`:

```bash
make e2 PROFILE=full DS=FD002
```

### Data

Place the NASA C-MAPSS files in `data/raw/CMAPSS/` as `train_FD001.txt` …
`train_FD004.txt`. The loader fails loudly with instructions if they are missing —
it never silently substitutes synthetic data outside of tests and the `smoke`
profile.

---

## Results

All figures below are read from `results/*/e*_results.json` on **FD001** using the
`fast` profile. Regenerate them with `make report`.

### Model comparison (e2) — unit-disjoint 3-fold OOF

| Model | ROC-AUC h30 | RUL MAE | RUL RMSE | NASA score | Features |
|---|---|---|---|---|---|
| Ensemble | 0.9893 | **9.86** | 15.11 | 5.72 | 1117 |
| XGBoost | 0.9528 | 10.06 | **15.16** | 4.67 | 1117 |
| RandomForest | 0.9360 | 10.85 | 16.48 | 6.76 | 1117 |
| Hybrid | 0.9923 | 11.57 | 15.54 | 4.41 | 1117 |
| LSTM | 0.9949 | 11.65 | 15.09 | 4.49 | 24 (raw) |
| GRU | **0.9962** | 11.79 | 14.99 | **3.82** | 24 (raw) |
| Logistic | 0.9864 | 11.95 | 19.49 | 20.89 | 1117 |
| Threshold | 0.9772 | 36.98 | 41.69 | 308.34 | 24 (raw) |

The sequence models win on discrimination; the soft-vote ensemble wins on point
RUL. The threshold baseline is deliberately simple and is included as a physics-
style reference point, not a contender.

### Feature ablation (e4) — does engineering actually help?

| Variant | Features | Mean RUL MAE |
|---|---|---|
| A_raw | 24 | 30.58 |
| B_raw_stat | 465 | 27.26 |
| C_raw_temporal | 612 | 26.25 |
| D_raw_stat_temporal | 1053 | 26.04 |
| E_full | 1117 | **24.45** |

All four comparisons against `E_full` are significant after Holm–Bonferroni
correction (adjusted *p* = 0.0000).

### Health index quality (e1)

| Metric | Value |
|---|---|
| Monotonicity (mean ± std) | 1.0000 ± 0.0000 |
| Trendability | 1.0000 |
| Spearman ρ vs true RUL | 0.8152 ± 0.0555 |
| Prognosability | 2.3089 |

### Early warning (e3) — horizon 30, FAR budget 5%

| Model | Mean lead time | Median | FAR |
|---|---|---|---|
| Logistic | 20.69 | 18.0 | 0.000 |
| RandomForest | 15.53 | 12.0 | 0.000 |
| XGBoost | 9.56 | 5.0 | 0.000 |

### Uncertainty methods (e9) — nominal 90% interval

| Method | Coverage (PICP) | Mean width | Winkler |
|---|---|---|---|
| split conformal | 0.9132 | 164.5 | 192.9 |
| ensemble variance | 0.8972 | 66.4 | **90.7** |
| quantile bootstrap | 0.9017 | 71.2 | 93.7 |
| bootstrap (cluster + residuals) | 0.8693 | **64.4** | 96.4 |

Ensemble variance gives the best width–sharpness trade-off. Note that pure
epistemic variance alone would badly under-cover (verified in the test suite at
0.17 on synthetic noise) — the released method adds calibration residual spread.

### Calibration (e5)

| Horizon | Brier | ECE (equal-width) | ECE (equal-mass) |
|---|---|---|---|
| h10 | 0.0197 | 0.0193 | 0.0175 |
| h20 | 0.0268 | 0.0259 | 0.0242 |
| h30 | 0.0371 | 0.0343 | 0.0265 |
| h40 | 0.0485 | 0.0426 | 0.0421 |
| h50 | 0.0583 | 0.0464 | 0.0399 |

Split-conformal intervals on held-out units: **coverage 0.9703**, mean width 66.54,
RUL MAE 9.02.

### Cross-dataset generalization (e6)

The model transfers imperfectly across operating regimes — off-diagonal ROC-AUC in
`results/e6_generalization/` drops as low as 0.57, while the diagonal stays above
0.95. Condition normalization (e7) moves ROC-AUC by less than 0.001
(0.9456 → 0.9460), i.e. **operating-condition normalization is not the lever it was
assumed to be** on this dataset. Both results are reported as measured.

---

## How it works

```
raw C-MAPSS rows
      │
      ├─► preprocessing      missing-value handling, outlier flags, smoothing
      │
      ├─► feature library    rolling stats · EWMA · slope · z-score · KS/PSI shift
      │                      cross-sensor correlation  (~1 100 columns, cached)
      │
      ├─► causal imputation   unit-wise forward fill, then training medians
      │
      ├─► per-fold screening  top-K by train-only correlation  (K = 300)
      │
      ├─► models             logistic · random forest · XGBoost · GRU · LSTM
      │                      hybrid · threshold · soft-vote ensemble
      │
      ├─► uncertainty        split conformal · ensemble variance · quantile
      │                      bootstrap · cluster bootstrap
      │
      └─► artifacts + UI     JSON under results/  ->  Streamlit dashboard
```

**Failure horizons.** A row is labelled `fail_h{h}` when `RUL <= h` for
`h ∈ {10, 20, 30, 40, 50}`. RUL is clipped at 125 cycles before labelling, so
labels are stable for very long-lived units.

**Sequence models.** GRU/LSTM consume raw sensor columns only and standardise
inputs with statistics fitted on the training frame. RUL targets are divided by
the training maximum so the regression MSE is the same order of magnitude as the
horizon BCE loss — otherwise the regression term swamps the classification term
and the network collapses to a constant. Early stopping uses a **unit-disjoint
inner split**, never the training rows.

---

## Repository layout

```
stat-twin/
├── configs/
│   ├── base.yaml                 # all tunables (pydantic-validated)
│   └── profiles/                 # smoke | fast | full
├── data/raw/CMAPSS/              # train_FD00{1..4}.txt
├── src/stattwin/
│   ├── config.py                 # typed config; nothing hard-coded downstream
│   ├── data/                     # loader, schema, splitter, synthetic
│   ├── statistics/               # feature library, rolling, cross-sensor, shift
│   ├── health/                   # SHI, state classification
│   ├── models/                   # 8 models behind one BaseModel interface
│   ├── evaluation/               # metrics, lead time, NASA score
│   ├── uncertainty/              # conformal, calibration
│   ├── decision/                 # thresholds and maintenance guidance
│   ├── explainability/           # contribution analysis
│   ├── forecasting/              # trajectory extrapolation
│   ├── experiments/              # e0 … e9, each writing results/<exp>/
│   ├── reports/generate.py       # auto-generates results tables
│   ├── app_data.py               # builds dashboard artifacts
│   └── dashboard/                # Streamlit app + 7 views
├── results/                      # experiment outputs (committed evidence)
├── reports/                      # generated Markdown tables
└── tests/                        # 313 tests
```

---

## Configuration

Everything tunable lives in `configs/base.yaml` and is validated by pydantic.
Profiles layer on top:

| Profile | Folds | XGBoost | GRU | Feature screen | Intended use |
|---|---|---|---|---|---|
| `smoke` | 2 | 50 trees | 32×1, 1 seed | 300 | CI, no real data needed |
| `fast` | 3 | 400 trees | 32×1, 12 epochs | 300 | development loop |
| `full` | 5 | 400 trees | 64×2, 5 seeds | 300 | final report numbers |

Feature screening (`model.feature_screen`) keeps the top-K columns by
correlation with RUL and the failure labels, refit **inside every fold** so no
validation unit influences the choice. Raw sensor and operating-setting columns
are always retained.

---

## Experiments

| ID | Question | Key output |
|---|---|---|
| **e0** | Is the data fit for modelling? | completeness, constant sensors, regimes |
| **e1** | Does the health index track reality? | monotonicity, trendability, Spearman ρ |
| **e2** | Which model is best? | per-horizon AUC, RUL, NASA, per-fold detail |
| **e3** | How early can we warn? | lead time at a fixed FAR budget |
| **e4** | Do engineered features earn their cost? | ablation + Holm-corrected tests |
| **e5** | Are the probabilities trustworthy? | Brier, ECE, reliability, conformal |
| **e6** | Does it transfer to other regimes? | 4×4 train/test matrix |
| **e7** | Does condition normalization help? | with/without comparison |
| **e8** | Can we detect injected faults? | precision/recall per fault type |
| **e9** | Which uncertainty method to ship? | PICP, width, Winkler |

Each writes `results/<experiment>/<exp>_results.json` plus figures, a
`manifest.json` (git hash, config hash, seeds, versions), and **checkpoints after
every model** — an interrupted e2 run resumes instead of restarting.

---

## Dashboard

```bash
make app
```

Seven views, each refreshing on a 2-second fragment:

1. **Overview** — state, SHI, P(fail ≤30), RUL, risk trajectory
2. **Sensor Monitoring** — raw vs rolling vs EWMA, z-scores, DQ flags
3. **Statistical Health** — SHI trajectory with state bands, heatmap, shift tables
4. **Failure Forecast** — horizon probabilities, RUL countdown, intervals
5. **Explainability** — per-sensor contribution to the risk estimate
6. **What-If Simulator** — counterfactual sliders, clearly labelled `SIMULATED`
7. **Model Comparison** — 8-model table, early warning, ablation, calibration,
   interval comparison, cross-machine generalization

The interface is CSS-only by design: Streamlit rebuilds its DOM on every rerun, so
JavaScript mutations are discarded. The motion layer
(`dashboard/components/motion.py`) therefore uses keyframes, `@property`
registered custom properties, and conic gradients — which survive reruns.

---

## Engineering guarantees

These are enforced by tests, not by convention.

**No leakage.** Every split is unit-level. Feature screening, imputation
medians, calibrators, conformal quantiles, and threshold fitting are all computed
on training units only and then applied to validation units. A test asserts that a
feature correlated with the target *only* in validation units is not selected.

**Causality.** Every feature uses rows `<= t`. Tests verify that a row's
prediction is unchanged when the frame is shuffled or units are interleaved —
i.e. no prediction depends on row order or on future rows.

**No fabricated numbers.** The dashboard reads only from `results/` and
`artifacts/`. Missing data renders an explicit message, never a plausible-looking
placeholder. Tests assert the report generator and the dashboard produce no
invented model names.

**Honest coverage.** Conformal coverage is measured on held-out units, not read
back from the calibration scores (which would report ~1−α by construction).

---

## Performance

Measured on FD001 (20 631 rows × 1 117 features, 12 logical cores, CPU only).

| Operation | Before | After |
|---|---|---|
| XGBoost fit (400 trees) | 203 s | **12.9 s** |
| RandomForest fit (200 trees) | 133 s | **22.3 s** |
| Full e2 (8 models × 3 folds) | > 90 min (unfinished) | **~21 min** |

The gains come from per-fold feature screening, causal imputation, a parquet
feature cache, and checkpoint/resume. Re-runs after the first hit the cache and
skip ~4 minutes of windowed statistics.

---

## Testing

```bash
make test            # 313 tests (249 test functions, many parametrized)
python -m pytest tests/ -q --tb=short
python -m pytest tests/test_enhancements.py -q   # correctness regressions
python -m pytest tests/leakage_guard/ -v         # leakage assertions
```

| Suite | Coverage |
|---|---|
| `test_enhancements.py` | regressions for every bug found in this project |
| `test_dashboard_motion.py` | motion layer + HTML rendering guards |
| `test_models.py` | model interface and training |
| `test_uncertainty.py` | conformal and calibration maths |
| `test_health.py`, `test_statistics.py` | SHI and the feature library |
| `test_decision.py`, `test_forecasting.py` | thresholds, guidance, extrapolation |
| `leakage_guard/` | split and transform discipline |

---

## Troubleshooting

**`Raw C-MAPSS file not found`** — download the dataset and place
`train_FD001.txt` … `train_FD004.txt` in `data/raw/CMAPSS/`.

**A dashboard view shows "not run yet"** — that experiment has not been executed.
Run the matching target, e.g. `make e6`.

**A chart is flat** — check `results/<exp>/` for that experiment. Several views
are driven by artifacts, so a stale or missing result file shows as an empty chart
rather than a fabricated curve.

**e2 seems to restart** — it is resuming. Results are checkpointed per model and
reused only when the config fingerprint (including every hyperparameter) matches.
Use `--no-resume` to force a full recompute.

**Port 8501 in use** — stop the previous instance, then `make app`.

---

## Citation and attribution

- **Data:** NASA Prognostics Center of Excellence — C-MAPSS turbofan degradation
  dataset (FD001–FD004).
- **Method:** adapted per `docs/MASTERPLAN.md`; SHI follows the statistical
  health-monitoring literature; NASA scoring follows the standard
  exponential-weighted RUL metric.
- **Attribution:** Vishwesh Penkar, Daksh Patil — guidance: Rashmi Sartkar.

Every number in this README is generated from committed result files. If a figure
looks wrong, the result JSON is the source of truth — please open an issue rather
than editing this file.
