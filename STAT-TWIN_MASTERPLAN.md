# STAT-TWIN — Master Plan

**Statistical Digital Twin for Probabilistic Failure Forecasting and Predictive Maintenance**

Version 1.0 · Audience: **opencode (AI coding agent)** and the human owner · Status: planning only — no implementation yet

---

## 0. How to use this document

1. Save this file as `docs/MASTERPLAN.md` in the repo. Copy **Appendix A** into `AGENTS.md` at the repo root so opencode loads the rules every session.
2. Build **one phase at a time** (Section 10). For each phase: start in Plan mode, review the plan, then switch to Build mode.
3. Use this kickoff prompt for every phase (replace `N`):

```
Read AGENTS.md and docs/MASTERPLAN.md. Implement ONLY Phase N (Section 10).
Follow the module specs in Section 6 and the Definition of Done for Phase N.
Do not start any other phase. When finished: run the verification commands,
list which DoD items pass/fail, and stop.
```

4. **Precedence:** Section 1 (charter) and Section 3.1 (scope lock) are locked. If anything elsewhere conflicts with Section 1, Section 1 wins.
5. **Traceability:** Section 3.2 maps every element of the original plan to a section here. Sections marked *additive* only add rigor or presentation value; they never remove or alter original requirements.

---

## 1. Project charter (LOCKED — original plan, unchanged)

### 1.1 Identity
**STAT-TWIN** — Statistical Digital Twin for Probabilistic Failure Forecasting and Predictive Maintenance.

### 1.2 Goal
Build a professional, research-oriented, **software-only** predictive-maintenance platform that learns the statistical behaviour of machines, detects degradation, forecasts future failure probability, estimates RUL, quantifies uncertainty, and explains why risk is increasing.

### 1.3 Primary research question
Does combining statistical health indicators with temporal ML improve early failure detection, RUL estimation, and prediction reliability compared with threshold-based and ML-only approaches?

### 1.4 Core concept
Do **not** build a simple "sensor values → failure prediction" app. STAT-TWIN behaves like a statistical digital twin:

```
Sensor telemetry → Data Quality → Statistical Analysis → Health State
→ ML Forecasting → Probabilistic Failure Forecast → RUL + Uncertainty
→ Explainability → What-If Simulation → Dashboard
```

### 1.5 Hypotheses
- **H0:** Statistical health indicators do not improve predictive performance.
- **H1:** Statistical health indicators improve predictive performance.
- Only claim significance when the experiments support it (protocol in Section 7, E4).

### 1.6 CTO principles (binding on every phase)
1. Research validity beats adding technologies.
2. Prevent data leakage at every stage.
3. Every metric has a defined evaluation protocol.
4. **Never fabricate performance numbers.** Every number in a report or dashboard comes from a result artifact.
5. Never claim statistical significance without evidence.
6. Never present simulated counterfactuals as real maintenance guarantees.
7. Statistical modelling is a core component, not decoration.
8. Every important prediction is explainable.
9. Experiments are reproducible.
10. Architecture stays generic enough for future industrial datasets.

### 1.7 Final product
Machine Telemetry → Statistical Behaviour Modelling → Degradation Detection → Health State → Failure Probability → RUL → Uncertainty → Explainability → What-If Simulation → Maintenance Decision Support.

Suitable for: college research project, ML/statistics demonstration, research paper, technical presentation/viva, future industrial expansion.

### 1.8 Non-goals
Hardware/IoT integration, real-time streaming infrastructure, cloud deployment, auth/multi-tenant features, prescriptive maintenance orders.

---

## 2. Positioning: why this is not a "normal" ML project (additive)

A typical portfolio ML repo is: one notebook, one random train/test split, one accuracy number, a Streamlit box that takes inputs and returns a label. STAT-TWIN is deliberately built as an **instrument** rather than a predictor: it shows its evidence, states how sure it is, and tests its own claims.

| Typical project | STAT-TWIN |
|---|---|
| One notebook | Installable Python package, config-driven, CLI, tests |
| Random split (leaks) | Unit-level splits + automated **Leakage Guard** tests |
| One accuracy number | Per-horizon metrics, calibration, coverage, lead time, ablation |
| "Model is 95% accurate" | Paired significance tests with Holm correction; honest negative results allowed |
| Point prediction | Probability curve + RUL **with conformal intervals** and measured coverage |
| Black box | Evidence cards: "Sensor X: z=3.1, trend +18%, EWMA HIGH" |
| Assumes clean data | Fault-injection benchmark separating **sensor faults from real degradation** |
| Trained once on one dataset | 4×4 cross-dataset generalization matrix under distribution shift |
| Static UI | **Live Replay** of an engine's life, cycle by cycle, with provenance badges |
| Hand-typed README numbers | Report and README tables auto-generated from result files |

### 2.1 Signature features (the "wow" list)
Each is cheap because it reuses the core pipeline. Priority tags are defined in Section 3.3.

| # | Feature | What the professor sees | Cost | Tier |
|---|---|---|---|---|
| S1 | **Leakage Guard** test suite (truncation-invariance, shuffled-label sanity, unit-disjointness) | "This student can *prove* there is no leakage." | Low | P0 |
| S2 | **Lead time at matched false-alarm budget** | Fair comparison of models on the metric maintenance people care about | Low | P0 |
| S3 | **Conformal RUL intervals** with empirical coverage (plus coverage under shift) | Distribution-free guarantee, easy to explain in a viva | Low–Med | P0 |
| S4 | **Statistical significance protocol** (paired Wilcoxon + bootstrap CI + Holm) on the ablation | H0/H1 are actually *tested* | Low | P0 |
| S5 | **Health-index quality metrics** (monotonicity, trendability, prognosability) evaluating the SHI itself | The health index is validated, not just invented | Low | P0 |
| S6 | **Live Replay Digital Twin** in the dashboard | Twin "ages" in front of the audience; the demo centrepiece | Low | P0 |
| S7 | **Evidence Cards + provenance badges** (OBSERVED / PREDICTED / SIMULATED) | Clear epistemic hygiene | Low | P0 |
| S8 | **Cross-dataset generalization heatmap** (train FDxxx → test FDyyy) | One striking figure covers "distribution shift" | Low | P1 |
| S9 | **Fault-injection benchmark**: does data-quality gating reduce false alarms caused by sensor faults? | Novel, memorable experiment answering the "data-quality vs machine anomaly" requirement | Med | P1 |
| S10 | **Auto-generated research report + README results** from `results/*.json` | No hand-typed numbers; reproducible | Med | P1 |
| S11 | **Viva cheat-sheet** (`docs/VIVA.md`) | Confidence in the room | Low | P1 |
| S12 | Uncertainty method comparison (conformal vs ensemble vs quantile vs bootstrap) | Justifies the chosen method with data | Med | P2 |

### 2.2 The 30-second pitch
> "Most predictive-maintenance projects give you a number. STAT-TWIN gives you a *reason and a confidence*. It builds a statistical model of normal behaviour for each engine, tracks how the engine drifts away from it, forecasts failure probability at 10 to 50 cycles ahead with calibrated uncertainty, and tells you which sensors and which statistics drove the risk. I tested whether the statistical layer actually helps using a leakage-safe ablation with paired significance tests, and I tested whether the system can tell a broken sensor from a breaking engine."

---

## 3. Scope lock and time strategy

### 3.1 Scope lock
**All original scope stays.** Time pressure is handled by *ordering and depth*, never by deleting original requirements. Every original item has a minimum viable implementation (MVP depth) that is always delivered, and a fuller depth delivered if time allows.

### 3.2 Traceability matrix (original plan → this document)

| Original plan element | Where it lives here |
|---|---|
| Goal, research question, concept, hypotheses, principles | Section 1 |
| NASA C-MAPSS FD001 → FD002/3/4, dataset-agnostic | Sections 5, 6.1, 7 (E6, E7) |
| Data pipeline (missing values, outliers, scalers, train-only fit, unit-level split, no leakage) | Sections 5.3, 6.2, 9 |
| Statistical engine (all listed statistics, distribution shift, DQ vs machine anomalies) | Sections 6.2 (DQ), 6.3 |
| Statistical Health Index 0–100, configurable, documented thresholds | Section 6.4, Appendix C |
| Five health states | Section 6.4 |
| 7 model baselines/hybrid | Sections 6.5, 6.6 |
| Multi-horizon forecasting, predicted failure point, RUL, interval, probability curve | Section 6.7 |
| Uncertainty as first-class; compare approaches; pick best for validity + viva | Section 6.8 (decision: split conformal, primary) |
| Probability calibration (reliability diagram, Brier, curves, calibration error) | Section 6.8, Appendix D |
| Early warning metrics (first valid warning, lead time stats, false alarm, false early-warning) | Section 6.9, Appendix D |
| Explainability ("why did risk increase") | Section 6.10 |
| What-if / counterfactual simulator, labelled as simulation | Section 6.11 |
| Maintenance decision support (4 tiers, evidence shown) | Section 6.12 |
| Research evaluation metrics (classification, RUL, forecasting, probability) | Section 6.13, Appendix D |
| Ablation A–E | Section 7 (E4) |
| Generalization (unit-level, operating conditions, FD001→FD002/FD004, shift) | Section 7 (E6, E7) |
| Streamlit dashboard, 7 pages | Section 8 |
| UI/UX professional industrial look | Section 8.1 |
| Stack and modular architecture | Section 4 |
| Project structure concept | Section 4.3 |
| Phases 1–11 | Section 10 (Phase 0 added as scaffolding) |
| "Do not write implementation code yet; produce architecture, roadmap, module responsibilities, experiment plan, data flow, milestone checklist" | Sections 4, 10, 6, 7, 4.2, 11 |

### 3.3 Priority tiers

| Tier | Meaning | Rule |
|---|---|---|
| **P0** | Must exist for the project to be defensible | Build first. Never cut. |
| **P1** | Strengthens research value and demo | Build if P0 is done and verified |
| **P2** | Nice to have | Only with spare time; skip silently |

### 3.4 Build schedule (assumes about 7 working days at 4–6 focused hours; opencode does the typing, you review and run)

| Day | Phases | Outcome |
|---|---|---|
| 1 | 0, 1 | Repo scaffold, Leakage Guard skeleton, FD001 ingested, splits verified |
| 2 | 2, 3 | Statistical features and SHI with state engine; SHI quality metrics |
| 3 | 4 | All 5 non-neural baselines evaluated with shared harness |
| 4 | 5, 6 | GRU baseline, STAT-TWIN Hybrid, probability + RUL + conformal + calibration |
| 5 | 7, 8, 9 (core) | Explainability, What-If, ablation with significance tests |
| 6 | 10 | Full Streamlit dashboard including Live Replay |
| 7 | 9 (extras), 11 | Generalization matrix, fault injection, auto report, viva sheet, rehearsal |

**Crunch mode (3–4 days):** Do P0 only, FD001 only, XGBoost as the primary learner for the ablation, GRU only for baseline #6 and the Hybrid with `profile: fast`. Skip S8–S12. Dashboard pages 1, 4, 5, 6, 7 first; pages 2, 3 with basic charts.

**Rule for schedule slip:** cut depth (fewer seeds, fewer datasets, fewer plots), never cut Leakage Guard, calibration, or the ablation significance test.

### 3.5 Compute profiles
Every experiment runs under a profile in `configs/`:

| Profile | Datasets | Folds | Seeds | GRU epochs | Purpose |
|---|---|---|---|---|---|
| `smoke` | synthetic mini-fixture | 2 | 1 | 2 | CI and unit tests (no real data needed) |
| `fast` | FD001 | 3 | 1 | 15 | Development loop |
| `full` | FD001–FD004 | 5 | 5 | early stopping | Final results for the report |

Everything must be runnable on a laptop CPU. GPU is optional and never required.

---

## 4. System architecture

### 4.1 Layered view

```mermaid
flowchart LR
  A[Raw telemetry<br/>C-MAPSS files] --> B[Ingestion<br/>+ unit-level split]
  B --> C[Preprocessing<br/>missing / outliers / scaling<br/>train-only fit]
  C --> D[Data-Quality Engine<br/>DQ flags]
  C --> E[Statistical Engine<br/>rolling, EWMA, trend, corr, shift]
  D --> E
  E --> F[Health Engine<br/>SHI 0-100 + 5 states]
  E --> G[ML Layer<br/>7 model families]
  F --> G
  G --> H[Forecasting<br/>P(fail by +h), RUL, failure point]
  H --> I[Uncertainty + Calibration<br/>conformal, isotonic]
  I --> J[Explainability<br/>evidence cards, attribution]
  I --> K[What-If Simulator]
  J --> L[Decision Support]
  K --> L
  L --> M[Streamlit Dashboard]
  I --> N[Evaluation + Experiments<br/>ablation, generalization, significance]
  N --> O[Auto Report / README]
```

### 4.2 Data flow (end to end)

1. **Ingest:** read `train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt`; assign column names; compute train RUL per row (`RUL = max_cycle(unit) − cycle`); apply optional RUL clipping (default 125) for the regression target only.
2. **Split:** `GroupKFold` by `unit_id` on official train units (5 folds). Official test set is a final holdout, touched only in final evaluation.
3. **Preprocess (fit inside each fold on training-fold units only):** missing-value strategy, outlier handling, scaler, operating-condition normalization (FD002/FD004).
4. **DQ engine (causal):** per-sensor flags (missing, stuck, spike, out-of-range, dropout burst). Machine-behaviour anomaly logic in the health layer uses persistence and cross-sensor agreement.
5. **Statistics (causal):** windowed features per unit per cycle. Features at cycle *t* depend only on cycles ≤ *t*.
6. **Health:** normalized evidence components → SHI (0–100) → smoothed → state with persistence.
7. **Models:** train per feature group / model family; produce out-of-fold (OOF) predictions and test predictions.
8. **Forecast:** probability heads at horizons {10, 20, 30, 40, 50}, RUL point estimate, predicted failure cycle.
9. **Uncertainty:** conformal RUL interval calibrated on OOF residuals; isotonic/Platt probability calibration on OOF predictions.
10. **Explain:** evidence cards (deterministic stats) + model attribution (occlusion by feature group; SHAP optional for tree models).
11. **Simulate:** perturb the recent window, recompute features and SHI, re-predict, compare.
12. **Evaluate:** metrics, ablation, significance tests, generalization, fault injection → `results/`.
13. **Present:** dashboard reads precomputed artifacts; only the What-If page calls models live.

### 4.3 Repository structure
Top-level folder names follow the original "project structure concept". Python code lives in a package so it is importable and testable; raw files live in `data/`.

```
stat-twin/
├── AGENTS.md                      # opencode rules (Appendix A)
├── README.md                      # hero screenshot, pitch, quickstart, auto-generated results
├── Makefile                       # setup, data, features, train, eval, report, app, test
├── pyproject.toml                 # package + tool config (ruff, pytest)
├── .streamlit/config.toml         # dark industrial theme
├── data/
│   ├── raw/CMAPSS/                # user drops NASA files here (never committed)
│   └── processed/                 # parquet caches
├── configs/
│   ├── base.yaml
│   ├── profiles/{smoke,fast,full}.yaml
│   ├── datasets/{fd001,fd002,fd003,fd004}.yaml
│   └── experiments/{ablation,generalization,fault_injection}.yaml
├── src/stattwin/
│   ├── data/                      # ingestion, schema, splitting, loaders, synthetic fixture
│   ├── preprocessing/             # missing, outliers, scalers, condition normalization, DQ
│   ├── statistics/                # rolling/EWMA/trend/corr/shift feature library
│   ├── health/                    # SHI, states, HI-quality metrics
│   ├── models/                    # baselines 1-6, hybrid, common interface
│   ├── forecasting/               # multi-horizon probs, RUL, failure point, monotone curve
│   ├── uncertainty/               # conformal, ensemble variance, quantile, bootstrap; calibration
│   ├── explainability/            # evidence cards, group occlusion, optional SHAP
│   ├── counterfactual/            # what-if perturbation engine
│   ├── evaluation/                # metrics, lead-time, protocols, significance tests
│   ├── decision/                  # maintenance guidance rules
│   ├── dashboard/                 # Streamlit app + pages + components + theme
│   ├── experiments/               # runnable experiment scripts E0-E9
│   ├── reports/                   # report templating and figure generation
│   ├── cli.py                     # Typer CLI: stattwin <command>
│   └── config.py                  # pydantic config loader
├── artifacts/                     # models, features, predictions (gitignored)
├── results/                       # metrics JSON, figures, tables (versioned summary only)
├── reports/                       # generated report + templates
├── docs/                          # MASTERPLAN.md, ARCHITECTURE.md, VIVA.md, MODEL_CARD.md, LIMITATIONS.md
└── tests/                         # unit tests + leakage_guard/
```

### 4.4 Module responsibilities (single-responsibility contract)

| Module | Owns | Must never |
|---|---|---|
| `data` | Reading files, schema, RUL labels, unit-level splits | Fit any statistic |
| `preprocessing` | Fit/transform objects, DQ flags | See validation/test rows during `fit` |
| `statistics` | Causal feature computation | Use labels (RUL) or future rows |
| `health` | SHI, states, HI quality | Use test data to set thresholds |
| `models` | Learners with a common `fit/predict_proba/predict_rul` interface | Do their own preprocessing outside the pipeline |
| `forecasting` | Turning model outputs into curve, failure point | Alter model outputs except enforced monotonicity |
| `uncertainty` | Intervals and probability calibration | Calibrate on data used to train the same fold model |
| `explainability` | Attribution and evidence | Present attribution as causal truth |
| `counterfactual` | Perturb → recompute → re-predict | Present results as real outcomes |
| `evaluation` | Metrics and tests | Hand-type or hard-code numbers |
| `dashboard` | Presentation | Recompute science; read from artifacts |

### 4.5 Model interface (contract)

Every learner implements:

```
fit(X_train, y_train, groups=None) -> self
predict_proba(X) -> ndarray[n, n_horizons]      # P(fail within h) for h in horizons
predict_rul(X) -> ndarray[n]
score_raw(X) -> ndarray[n]                       # continuous risk score for ROC/PR
```

Baselines 1–2 (threshold, statistical anomaly) are score-producers; they are mapped to probabilities and RUL by monotone/isotonic mapping fitted on training folds, so all seven models are comparable under one harness.

### 4.6 Artifact contracts

| Artifact | Path | Key columns |
|---|---|---|
| Features | `artifacts/{ds}/features.parquet` | `unit_id, cycle, split/fold, raw_*, stat_*, hi_*, dq_*, rul, rul_clipped` |
| OOF predictions | `artifacts/{ds}/oof_{model}_{variant}.parquet` | `unit_id, cycle, fold, seed, p_h10..p_h50, rul_hat, rul_lo, rul_hi, shi, state` |
| Test predictions | `artifacts/{ds}/test_{model}_{variant}.parquet` | same schema |
| Metrics | `results/{experiment}/metrics.json` | nested by dataset, model, variant, horizon |
| Tables | `results/{experiment}/*.csv` | machine-readable, used by the report |
| Figures | `results/{experiment}/figures/*.png` (+ `.json` Plotly spec) | referenced by report and dashboard |
| Run manifest | `results/{experiment}/manifest.json` | git hash, config hash, seeds, package versions, timestamp |

### 4.7 Config system
YAML validated by pydantic. A run is fully defined by `base.yaml` + dataset + profile + experiment file. Config hash is written into every manifest. Seeds are set for `random`, `numpy`, `torch`, and XGBoost.

Example fragment (`configs/base.yaml`):

```yaml
seed: 42
dataset: {name: FD001, raw_dir: data/raw/CMAPSS, rul_clip: 125, use_age_feature: false}
split: {scheme: group_kfold, n_splits: 5, group_col: unit_id}
preprocess:
  missing: {strategy: ffill_then_median, add_indicators: true}
  outliers: {method: robust_z, threshold: 4.0, action: flag_and_winsorize}
  scaler: standard            # standard | robust
  per_condition_norm: auto    # on for datasets with >1 operating condition
stats:
  windows: [5, 10, 20, 30]
  ewma_alpha: [0.1, 0.3]
  baseline_cycles: 30         # per-unit healthy reference (first K cycles)
  corr: {methods: [pearson, spearman], pairs: top_k, top_k: 10}
  shift: [ks, wasserstein, psi]
health:
  weights: {deviation: 0.25, trend: 0.25, ewma: 0.20, variance: 0.15, corr_shift: 0.15}
  calibrate_weights: false    # if true: train-only NNLS fit
  smoothing: {method: ewma, alpha: 0.2}
  states: {method: calibrated, fallback_grid: [80, 60, 40, 20], persistence: 3}
forecast: {horizons: [10, 20, 30, 40, 50], seq_len: 30, enforce_monotone: true}
model:
  gru: {hidden: 64, layers: 2, dropout: 0.2, ensemble_size: 3}
  xgb: {n_estimators: 400, max_depth: 5, learning_rate: 0.05}
uncertainty: {method: split_conformal, alpha: 0.10, normalize_by: ensemble_std}
calibration: {method: isotonic, bins: 10}
warning: {horizon: 30, persistence: 3, valid_window: 100, far_budget: 0.05}
```

---

## 5. Data

### 5.1 NASA C-MAPSS
Turbofan engine run-to-failure simulations (Saxena et al., PHM08). Each row = one cycle of one engine: `unit_id, cycle, op_setting_1..3, sensor_1..21` (26 columns, space-separated, no header).

| Subset | Train / Test engines | Operating conditions | Fault modes | Role |
|---|---|---|---|---|
| FD001 | ≈100 / 100 | 1 | 1 (HPC degradation) | Primary development set |
| FD002 | ≈260 / 259 | 6 | 1 | Condition-shift study |
| FD003 | ≈100 / 100 | 1 | 2 (HPC + fan) | Fault-mode study |
| FD004 | ≈248 / 249 | 6 | 2 | Hardest; full shift study |

Facts the pipeline must handle:
- Train trajectories run to failure. Test trajectories are **truncated** at an unknown point; `RUL_FDxxx.txt` gives the true RUL at each test engine's last recorded cycle.
- Several sensors are constant or near-constant in some subsets. **Detect them programmatically from training data** (variance threshold), never hard-code (for FD001 this is expected to include roughly s1, s5, s10, s16, s18, s19).
- FD002/FD004 have 6 operating regimes. Sensors are dominated by regime, not health, unless normalized **per operating condition** (cluster `op_setting_1..3` with KMeans, k = 6, fit on train only; standardize sensors within each cluster).
- C-MAPSS has **no missing values or sensor faults**. Preprocessing robustness is therefore demonstrated with the fault-injection benchmark (E8) and unit tests on synthetic corruptions.

Sensor reference (for dashboard labels):

| ID | Symbol | Meaning | ID | Symbol | Meaning |
|---|---|---|---|---|---|
| s1 | T2 | Total temp at fan inlet | s12 | phi | Ratio of fuel flow to Ps30 |
| s2 | T24 | Total temp at LPC outlet | s13 | NRf | Corrected fan speed |
| s3 | T30 | Total temp at HPC outlet | s14 | NRc | Corrected core speed |
| s4 | T50 | Total temp at LPT outlet | s15 | BPR | Bypass ratio |
| s5 | P2 | Pressure at fan inlet | s16 | farB | Burner fuel-air ratio |
| s6 | P15 | Total pressure in bypass duct | s17 | htBleed | Bleed enthalpy |
| s7 | P30 | Total pressure at HPC outlet | s18 | Nf_dmd | Demanded fan speed |
| s8 | Nf | Physical fan speed | s19 | PCNfR_dmd | Demanded corrected fan speed |
| s9 | Nc | Physical core speed | s20 | W31 | HPT coolant bleed |
| s10 | epr | Engine pressure ratio | s21 | W32 | LPT coolant bleed |
| s11 | Ps30 | Static pressure at HPC outlet | | | |

Data acquisition: download the C-MAPSS archive from the NASA Prognostics Data Repository and unzip into `data/raw/CMAPSS/`. If files are missing, the pipeline **fails loudly** with instructions. It must never silently substitute synthetic data (synthetic fixtures are for tests only).

### 5.2 Labels
- `rul = max_cycle(unit) − cycle` on train trajectories.
- Regression target: `rul_clipped = min(rul, rul_clip)` (default 125, config-driven). Reported metrics also include an unclipped variant so the choice is transparent.
- Failure labels per horizon: `y_h = 1[rul ≤ h]` for `h ∈ {10, 20, 30, 40, 50}`.
- Failure is defined as the last recorded cycle of a training trajectory.

### 5.3 Splitting and evaluation protocol (the heart of research validity)

**Primary protocol: nested unit-level design.**

1. Official **train units** → `GroupKFold(n_splits=5)` by `unit_id`. Each fold: train on 4/5 of units, predict the held-out 1/5. The union of held-out predictions is the **OOF set** and covers every train unit exactly once, at every cycle.
2. Inside each training fold, an inner unit-level split (≈15% of the fold's units) is used for early stopping and hyperparameters. It never overlaps the outer held-out units.
3. OOF predictions serve: model comparison, ablation with paired tests, full-trajectory lead-time analysis, probability calibration fitting, and conformal calibration (cross-conformal style: residuals come from models that did not train on the corresponding units).
4. **Official test set** (truncated engines + `RUL_FDxxx.txt`) is the final untouched holdout. Final models are trained on all train units. Standard C-MAPSS reporting (RUL at the last cycle per engine, RMSE and NASA score) is computed here.
5. Preprocessing objects, thresholds, calibrators, and health-state cut-offs are all **fitted inside the training portion only** and stored with the fold.

**Why both OOF and test:** the official test set gives literature-comparable RUL numbers, but only one point per engine. Lead time, false-alarm rate, and horizon-wise classification need full trajectories, which only OOF provides.

### 5.4 Leakage rules (binding)
1. Split by `unit_id` only. No cycle-level shuffling, ever.
2. Fit scalers, imputers, KMeans (regimes), sensor selection, thresholds, calibrators, and SHI weights on train portions only.
3. Statistical features are **causal**: features at cycle *t* use cycles ≤ *t* only. Per-unit healthy baseline uses that unit's first `baseline_cycles`; for *t* below that, an expanding (causal) baseline is used and rows are flagged `warmup`.
4. No feature may be derived from `rul`. A blacklist test enforces it.
5. Cycle index (age) is **excluded** from features by default (`use_age_feature: false`) for comparability with the literature; a sensitivity run may include it and must be labelled.
6. Hyperparameter and threshold selection use validation units only. The test set is used once, at the end.

---

## 6. Module specifications

Each module lists purpose, algorithms, interfaces, tests, and DoD.

### 6.1 `data` — ingestion and splitting
- **Loader:** parse all four subsets to tidy DataFrames with named columns; validate shape, column count, monotone cycles per unit, no duplicate `(unit_id, cycle)`.
- **Dataset abstraction:** `TelemetryDataset` with fields `unit_id, time_index, sensors, op_settings, metadata`, so other industrial datasets plug in by writing a small adapter (dataset-agnostic requirement).
- **Splitter:** unit-level GroupKFold plus inner split; returns unit lists, asserts disjointness.
- **Synthetic fixture:** a tiny generator (a few units with monotone drift, noise, optional regimes) used *only* by tests and the `smoke` profile.
- **DoD:** load FD001; unit counts correct; splits disjoint; RUL labels verified on a known unit; loader fails clearly when files are missing.

### 6.2 `preprocessing` — cleaning, scaling, data quality

**Missing values (configurable):** forward fill (causal), causal linear interpolation (uses only past values for online use; a non-causal variant is allowed for offline analysis but flagged), train-median fill. Optional **missingness indicator** columns.

**Outlier detection (configurable):** IQR fences, Z-score, Robust-Z (median/MAD). Statistics come from train portion. Action: flag, optionally winsorize.

**Scaling:** `StandardScaler` or `RobustScaler`, fitted on train only; optionally **per operating condition** for multi-regime datasets.

**Constant-sensor removal:** variance threshold learned on train.

**Data-quality engine (distinguishes DQ anomalies from machine anomalies).**

| DQ flag | Definition (causal) |
|---|---|
| `missing` | NaN / dropped value |
| `dropout_burst` | ≥ *k* consecutive missing values on a sensor |
| `stuck` | Zero (or near-zero) variance over window *w* while the sensor's healthy variance is non-trivial |
| `spike` | Single-cycle Robust-Z exceeding threshold that reverts within 1–2 cycles |
| `out_of_range` | Value outside physically plausible range learned from train (with margin) |

**Separation logic (documented rule set):**
- **Data-quality anomaly:** single-sensor, abrupt or transient, non-persistent, or structural (missing / stuck / out-of-range), and *not* corroborated by related sensors.
- **Machine-behaviour anomaly:** persistent (≥ *p* cycles), multi-sensor coherent, direction consistent with the learned degradation direction, and accompanied by trend/EWMA evidence.
- Cross-sensor agreement uses the correlation structure learned on healthy training data (e.g., a sensor moving alone against its usual correlated neighbours is more likely a DQ issue).
- **Gating:** when a DQ flag is active, the value is repaired for feature computation, a "data quality degraded" badge is shown, and alarm escalation is delayed until persistence is confirmed. Repair choices are logged.

**Tests:** injected NaNs/spikes/stuck values are flagged; scaler stats match train-only stats; regime normalization reduces regime variance on FD002-like synthetic data; no fit on non-train rows (see Section 9).

**DoD:** pipeline object with `fit(train)` / `transform(df)`; serializable; DQ flags emitted with reasons.

### 6.3 `statistics` — statistical engine (main differentiator)
All features are computed **per unit, per cycle, causally**, vectorized with pandas `groupby().rolling()` or NumPy (no Python row loops).

**Per sensor, per window `w ∈ {5, 10, 20, 30}`:**
- Rolling mean, std, min, max, range (max−min)
- Z-score of the current value against the unit's healthy baseline (mean/std from baseline period, std floored by a train-fitted minimum to avoid blow-ups)
- **EWMA** (α ∈ {0.1, 0.3}) and **EWMA deviation** from baseline (in baseline-σ units)
- **Linear trend / slope** by closed-form OLS over the window (units: σ per 10 cycles)
- **Percentage change** and **rate of change** over the window
- **Coefficient of variation** (std/|mean|, guarded for near-zero mean)

**Cross-sensor:**
- Rolling **Pearson** and **Spearman** correlation on the top-*k* sensor pairs. Pair selection uses training data only, chosen by healthy-phase correlation strength or by a variance/informativeness criterion.
- **Correlation shift:** ‖C_window − C_baseline‖_F (Frobenius), and per-pair Δρ.

**Distribution shift (current window vs healthy baseline window):**
- **KS statistic** (`scipy.stats.ks_2samp`), **Wasserstein-1** (`scipy.stats.wasserstein_distance`), **PSI** (bins from the baseline distribution; definition in Appendix D).

**Feature economy (time-saving and overfitting control):** compute a full library, then apply a train-only feature-selection step (e.g., drop near-duplicates by |ρ| > 0.98, keep top-*N* by mutual information or tree importance on the inner validation set). The selected list is stored in the artifact so the dashboard and What-If module can recompute exactly the same features.

**Tests:** causality (truncation invariance), a hand-computed slope on a known line, EWMA against a reference implementation, PSI zero for identical distributions, output shape and NaN policy for warm-up rows.

**DoD:** `compute_features(df, config) -> df`, deterministic, unit-tested; runtime for FD001 under a couple of minutes on a laptop.

### 6.4 `health` — Statistical Health Index and health states

**SHI (0–100), fully configurable and documented (formula in Appendix C).**
Evidence components, each normalized to [0, 1] by a **train-fitted** percentile (ECDF) mapping:

| Component | Signal |
|---|---|
| `deviation` | Signed z-score moved in the sensor's learned degradation direction |
| `trend` | Signed slope over window, normalized by baseline σ |
| `ewma` | EWMA deviation from baseline |
| `variance` | Log variance ratio vs baseline |
| `corr_shift` | Correlation-structure change |

Sensors are aggregated per component using **informativeness weights learned on train data only** (e.g., |Spearman| between the sensor and normalized life fraction). Component weights λ are configurable (defaults in `base.yaml`); optional train-only NNLS calibration against normalized life fraction is available (`calibrate_weights: true`). SHI = 100 × (1 − Σ λ_k c_k), lightly smoothed causally.

**Important honesty rule:** thresholds are **not** scientifically universal. Two variants are always produced and compared in a sensitivity table:
1. **Calibrated** cut-offs, derived on train units (e.g., median SHI at reference RUL levels or life-fraction quantiles).
2. **Fixed grid** fallback (80 / 60 / 40 / 20), labelled "documented convention".

**Health states:** `HEALTHY`, `WATCH`, `DEGRADING`, `CRITICAL`, `FAILURE-LIKELY`. A state change requires the new state to persist for *p* cycles (hysteresis) to prevent flicker.

**Health-index quality (S5):** report **monotonicity**, **trendability**, and **prognosability** on train/OOF units, plus Spearman correlation with true RUL. This validates the SHI as a health indicator independent of any classifier.

**DoD:** SHI computed for every cycle; state series per unit; quality metrics table; threshold sensitivity table; unit tests for bounds [0, 100] and persistence logic.

### 6.5 `models` — comparable baselines (models 1–5, and 6)

All trained with the same folds, features-by-variant, and evaluation harness.

| # | Model | Definition |
|---|---|---|
| 1 | **Fixed threshold** | Per-sensor limits = healthy mean ± *k*σ (train healthy phase); warn if ≥ *m* sensors violate for *p* cycles. Score = number/severity of violations |
| 2 | **Statistical anomaly detection** | Mahalanobis distance to the healthy reference distribution (robust covariance, train-fitted); threshold from the chi-square quantile or a train percentile; persistence |
| 3 | **Logistic Regression** | One binary model per horizon on the ML-only feature set (raw sensors), class-weighted; regularization tuned on validation |
| 4 | **Random Forest** | Per-horizon classifier + RUL regressor, same features |
| 5 | **XGBoost / Gradient Boosting** | Per-horizon classifier + RUL regressor; the primary fast learner for ablations |
| 6 | **LSTM/GRU** | Sequence model over the last `seq_len` cycles of **raw sensors only**; multi-head output (5 horizon logits + RUL) |

Notes:
- Baselines 3–5 are "ML-only": **raw sensor features only** (this is ablation variant A). This is the fair comparison the research question demands.
- Baselines 1–2 produce scores; probabilities and RUL come from monotone (isotonic) mappings fitted on training folds. State this clearly in the report.
- Class imbalance is severe for short horizons. Use class weights, report PR-AUC alongside ROC-AUC.

### 6.6 `models` — STAT-TWIN Hybrid (model 7, the proposed method)

**Inputs (per time step):** raw sensors + statistical features (Section 6.3) + health indicators (SHI and its five components, plus state one-hot), organized as a sequence of the last `seq_len` cycles (default 30), plus regime embedding for multi-condition data.

**Architecture (small on purpose, CPU-friendly):**
- 1–2 layer GRU (hidden 64), dropout 0.2 → concatenate with last-step health vector → MLP → heads:
  - 5 sigmoid outputs (failure within 10/20/30/40/50 cycles), BCE loss with positive weighting
  - 1 RUL output (Huber loss on scaled clipped RUL)
- Multi-task loss with configurable weights.
- **Ensemble** of *M* seeds (default 3 for `fast`, 5 for `full`): ensemble mean is the prediction; ensemble std feeds the uncertainty normalizer.
- Post-hoc **monotone enforcement** across horizons: P(+10) ≤ P(+20) ≤ … ≤ P(+50) (cumulative max), logged when applied.

**Time-safe fallback (contingency, not a replacement):** if GRU training time or stability threatens the schedule, a *tabular hybrid* is built first: XGBoost on raw + statistical + temporal (lags/differences) + health features. The GRU hybrid remains the proposed model and is completed as soon as the tabular version is verified.

**DoD:** hybrid trains on FD001 in the `fast` profile in a reasonable time on CPU; produces the OOF and test prediction schemas; all five outputs (probabilities, RUL, SHI, state, interval) present.

### 6.7 `forecasting` — probabilities, RUL, failure point
- **Probability curve:** `P(fail by t+h)` for `h ∈ {10,…,50}`; smooth monotone interpolation (PCHIP) for display only. The horizon values themselves are the model's outputs.
- **RUL:** point estimate from the RUL head (or ensemble mean), clipped to ≥ 0.
- **Predicted failure point:** `t_now + RUL_hat`, with interval `[t_now + RUL_lo, t_now + RUL_hi]`.
- **Consistency check (P1):** compare the probability curve with the RUL distribution implied by the conformal residual distribution; report disagreement.
- **DoD:** for any `(unit, cycle)` the API returns curve, RUL, failure point, interval, SHI, state.

### 6.8 `uncertainty` — uncertainty and calibration

**Decision (viva-friendly and research-valid): primary = split/cross-conformal prediction on the ensemble mean, with normalized residuals.**

Rationale: distribution-free, finite-sample marginal coverage guarantee under exchangeability, model-agnostic, cheap, and easy to explain. It also has a clean *failure mode* that becomes a finding: under distribution shift (FD001 → FD002/FD004) exchangeability breaks and coverage degrades, which we measure and report.

**Method:**
1. Collect OOF predictions and true (clipped) RUL for calibration units.
2. Nonconformity score `s = |y − ŷ| / (σ_ens + ε)` (locally adaptive; `σ_ens` = ensemble std).
3. `q = ⌈(n+1)(1−α)⌉/n` empirical quantile of scores (default α = 0.10 → 90% intervals).
4. Interval = `ŷ ± q · (σ_ens + ε)`, lower bound clipped at 0.
5. Report **PICP** (coverage), **mean interval width**, **interval (Winkler) score**, coverage **by RUL bucket** (conditional behaviour), and coverage **under shift**.

**Comparison set (S12, P2):** ensemble variance intervals (Gaussian assumption), quantile regression (XGBoost quantile objective or GRU quantile head), bootstrap ensemble. MC dropout is listed as considered and skipped for cost.

**Probability calibration:** isotonic regression (Platt as alternative) fitted on OOF predictions per horizon. Report **Brier score**, **ECE** (10 bins, plus equal-mass variant), **reliability diagrams**, and before/after calibration. Calibrator is fitted on OOF from *other* folds, never on the units it is evaluated on.

**DoD:** intervals with reported coverage on OOF and test; reliability diagram and calibration table for each horizon; documented choice.

### 6.9 `evaluation` — early warning and lead time
Definitions (also in Appendix D):
- **Warning rule:** `P(fail within H_w) ≥ τ` for `p` consecutive cycles (defaults `H_w = 30`, `p = 3`).
- **First valid warning time:** first cycle satisfying the rule **and** occurring inside the valid window (`true RUL ≤ valid_window`, default 100).
- **Lead time:** `T_fail − t_first_valid_warning` (cycles). Report mean, median, min, max.
- **False early warning:** a warning raised when `true RUL > valid_window` (too early to be useful).
- **False alarm rate:** fraction of units with any false early warning, and alarms per 1000 healthy cycles.
- **Missed warning:** units with no valid warning before failure.
- **Matched-budget comparison (S2):** choose τ per model on **validation units** so the false-alarm rate meets `far_budget` (default 5% of units), then compare lead times. Prevents models that "win" by warning absurdly early.

### 6.10 `explainability` — "Why did the risk increase?"

Two complementary layers:

**Layer 1: Statistical evidence cards (deterministic, model-agnostic).** For the top-*k* contributing sensors:

```
Sensor: T30 (s3)          Current: 1591.4   Baseline mean: 1587.9
Z-score: +3.1             Trend (20 cyc): +18%    EWMA deviation: HIGH
Variance ratio: 1.6x      Distribution shift: PSI 0.31 (moderate-high)
```

Cards include severity chips derived from configurable bins and a link to the sensor's chart.

**Layer 2: Model attribution.**
- **Group occlusion (primary, works for any model):** replace a feature group (a sensor, or a statistic type such as *trend*, *variance*, *EWMA*, *correlation*) with its healthy-baseline value; the drop in predicted risk is that group's contribution.
- **Risk-change decomposition:** attribute `P(now) − P(reference cycle)` to sensors and to statistic types ("trend contributed 41%, variance 22%...").
- **SHAP TreeExplainer (P1)** for tree-based models as a cross-check.

**Wording rule:** the UI says "contributed to the model's risk estimate", never "caused the failure".

**DoD:** given `(unit, cycle)` returns ranked sensors, statistic-type contributions, and evidence cards; unit test that occluding the top contributor lowers risk.

### 6.11 `counterfactual` — What-If simulator

**Mechanism:** the user modifies hypothetical conditions on the recent window (e.g., "T30 −8%", "Ps30 variance −10%", "Nf −5%"). The engine:
1. Applies the perturbation to the raw window (multiplicative shift, additive shift, or variance scaling around the rolling mean; optionally applied as a ramp).
2. Recomputes statistical features and SHI with the **same fitted pipeline** (same code path as production).
3. Re-runs the model.
4. Returns original vs simulated: probability curve, RUL (with interval), SHI, state, and the largest changes.

**Guardrails:**
- Every output carries a persistent **SIMULATION** badge and the text: *"Model-based what-if analysis on a learned statistical model. Not a guaranteed real-world maintenance outcome."*
- Perturbations are clipped to the training range; an **out-of-distribution warning** (Mahalanobis/PSI) appears if the modified window leaves the training distribution.
- C-MAPSS sensors are outputs of a simulation, not controllable inputs. The documentation states that the simulator is a **sensitivity analysis of the learned model**, and labels use C-MAPSS sensor names (T30, Ps30, Nf), not invented ones such as "vibration".

**DoD:** original vs simulated comparison for any window; deterministic; unit test that a zero perturbation reproduces the original prediction exactly.

### 6.12 `decision` — maintenance decision support (non-prescriptive)

| Risk tier | Rule (illustrative, configurable, calibrated on validation) | Guidance text |
|---|---|---|
| Low | P(+30) < 0.10 and state ∈ {HEALTHY} | Continue monitoring |
| Medium | 0.10 ≤ P(+30) < 0.30 or state = WATCH | Increase monitoring frequency |
| High | 0.30 ≤ P(+30) < 0.60 or state = DEGRADING | Inspect highest-contributing factors |
| Critical | P(+30) ≥ 0.60 or state ∈ {CRITICAL, FAILURE-LIKELY} | Engineering inspection according to organizational procedures |

Every recommendation displays the evidence behind it (probability, interval, top evidence cards, DQ status). If DQ degraded, the recommendation says so and lowers confidence display.

### 6.13 `evaluation` — metrics and statistical tests
Full definitions in Appendix D. Summary:

- **Classification (per horizon):** precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR; operating threshold chosen on validation.
- **RUL:** MAE, RMSE, NASA asymmetric score; on OOF (all cycles and late-life cycles) and on the official test set (last cycle per engine).
- **Forecasting:** lead time statistics, false early-warning rate, horizon-specific metrics.
- **Probability:** Brier, ECE, reliability curves.
- **Intervals:** PICP, width, interval score.
- **Statistical testing (S4):** per-unit paired differences; **Wilcoxon signed-rank**; paired **bootstrap CI** (resampling units, 10 000 resamples) on the difference; **Holm–Bonferroni** correction across the family of comparisons; effect sizes (median paired difference, rank-biserial correlation). Results aggregated over seeds first (mean per unit across seeds) to avoid pseudo-replication.

---

## 7. Experiment plan

All experiments are scripted (`stattwin.experiments.eXX`), config-driven, write `results/eXX/` with a manifest, and are re-runnable with `make eXX`.

| ID | Name | Question | Protocol | Outputs |
|---|---|---|---|---|
| **E0** | Data audit | Is the data as expected? | Load all subsets; unit counts, cycle-length distribution, constant sensors, regime clusters | Table + histograms |
| **E1** | Health-index validation | Is the SHI a good health indicator? | Monotonicity, trendability, prognosability, Spearman vs RUL; SHI weights: fixed vs calibrated; threshold sensitivity (calibrated vs 80/60/40/20) | Table + SHI trajectories figure |
| **E2** | Model comparison | How do the 7 models compare? | Section 5.3 protocol; per-horizon classification, RUL metrics, NASA score (test) | Comparison tables + bar/ROC/PR charts |
| **E3** | Early-warning benchmark (S2) | Who warns earlier at the same false-alarm budget? | Threshold τ tuned on validation to `far_budget`; lead-time stats on OOF trajectories | Lead-time distribution plots, table |
| **E4** | **Ablation (A–E)** + significance | Do statistical features add measurable value? | See below | Ablation table, paired-test table, forest plot of effect sizes |
| **E5** | Uncertainty and calibration | Are probabilities and intervals trustworthy? | Reliability diagrams, Brier, ECE before/after; conformal coverage/width by RUL bucket | Figures + table |
| **E6** | Generalization matrix (S8) | How does performance change across datasets? | Train FDx → test FDy for all pairs; RUL RMSE, coverage, ECE | 4×4 heatmaps, degradation table, PSI-vs-drop scatter |
| **E7** | Operating-condition study | Does regime normalization matter? | FD002/FD004 with and without per-condition normalization; naive transfer FD001 → FD002/FD004; optional label-free adaptation (clearly labelled transductive) | Table + figure |
| **E8** | Fault-injection benchmark (S9) | Can the system tell sensor faults from degradation? Does DQ gating cut false alarms? | Inject synthetic faults into held-out healthy-phase segments (spike, stuck, dropout, noise burst, single-sensor bias drift); measure DQ detection precision/recall and false-warning rate **with vs without gating** | Confusion table + false-alarm comparison |
| **E9** | Uncertainty method comparison (S12, P2) | Which uncertainty method is best? | Conformal vs ensemble variance vs quantile vs bootstrap: coverage, width, interval score | Table |

### E4 — Ablation design (required)

Variants (same folds, same learner, same tuning budget):

| Variant | Features |
|---|---|
| **A** | Raw sensors only |
| **B** | Raw + statistical features |
| **C** | Raw + temporal features |
| **D** | Raw + statistical + temporal |
| **E** | Full STAT-TWIN: D + health indicators + conformal uncertainty |

Definitions of "temporal":
- **Tabular learner (XGBoost, primary for speed and many seeds):** temporal = explicit lags (t−1, t−5, t−10) and first differences of raw sensors.
- **Sequence learner (GRU, confirmatory, P1):** temporal = sequence context (`seq_len = 30`); "no temporal" variants use `seq_len = 1`.
- Note for the report: statistical features are themselves windowed summaries, so variant B already carries history. E4 therefore separates *explicit sequence context* from *summary statistics*.

Primary endpoint: per-unit RUL absolute error on late-life cycles (e.g., last 50 cycles) and per-unit PR-AUC at horizon +30; secondary: lead time at matched budget, Brier, coverage. Tests: A vs B, C vs D, D vs E, A vs E (family for Holm correction declared in advance in the config).

Outcome handling: if H1 is supported, report effect sizes and CIs; if not, report honestly and analyze why (this is still a valid, defensible research result).

---

## 8. Dashboard specification (Streamlit)

### 8.1 Design system (professional industrial analytics look)

- **Theme:** dark (`#0E1117` background, slate cards), monospace numerics for readings, generous spacing, consistent card component, subtle borders, no default-looking Streamlit clutter (hide footer/menu via config, custom CSS).
- **Health state palette:** HEALTHY `#2E9E6B` · WATCH `#E0B93B` · DEGRADING `#E8862F` · CRITICAL `#D64545` · FAILURE-LIKELY `#8E1B3A`.
- **Provenance palette (never mixed):**
  - **OBSERVED**: slate-blue `#4C8BF5`, solid lines
  - **PREDICTED**: amber `#F5A623`, dashed lines with translucent uncertainty band
  - **SIMULATED**: violet `#9B6BFF`, dotted lines, "SIMULATION" corner badge
- Every chart carries a provenance badge. Every KPI carries units and a tooltip with its definition.
- **Hierarchy:** state + probability + RUL first (large), evidence second, raw detail third.
- Plotly for all charts, one shared template (fonts, colors, hover style).
- Reads precomputed artifacts for instant loading; only What-If runs live inference (cached with `st.cache_resource`).

### 8.2 Pages (original seven)

1. **OVERVIEW** — machine selector; current health state badge; SHI gauge; failure probability (+30 headline); RUL with prediction interval; current trend arrow; risk timeline; DQ status chip; recommendation card. **Live Replay (S6):** cycle slider plus Play/Pause that advances the "current cycle" (use `st.fragment(run_every=...)` when supported, slider fallback); state, probability, RUL, and timeline update as the engine "ages".
2. **SENSOR MONITORING** — interactive sensor selector; raw values with rolling mean/std band; EWMA; Z-score; trend lines; DQ-flag markers versus machine-anomaly markers on the timeline (visually distinct).
3. **STATISTICAL HEALTH** — SHI over time with state bands; Z-score heatmap (sensors × cycles); variance-change chart; distribution-shift table (KS / Wasserstein / PSI); correlation matrix with baseline-vs-current toggle and shift score; degradation indicators.
4. **FAILURE FORECAST** — multi-horizon probability curve; predicted failure point and RUL with interval; conformal band; warning timeline showing first valid warning and lead time so far.
5. **EXPLAINABILITY** — "Why did risk increase?" panel with reference-cycle selector; ranked contributions by sensor and by statistic type; evidence cards; top anomalies; sensor trend sparklines.
6. **WHAT-IF SIMULATOR** — sliders for hypothetical sensor changes (level and variance), ramp option; original vs simulated curves, RUL, SHI, state; delta cards; persistent SIMULATION disclaimer; OOD warning.
7. **MODEL COMPARISON** — tabs: Metrics (7 models), Early warning (lead time at matched budget), Ablation A–E with significance table and forest plot, Calibration (reliability), Intervals (coverage/width), Generalization heatmap, Fault injection. All values loaded from `results/`.

### 8.3 Demo script (5 minutes for the viva)
1. Overview: pick an engine, start Live Replay, watch state move HEALTHY → WATCH → DEGRADING.
2. Pause at a WATCH transition; open Explainability: "here is why".
3. Forecast page: probability curve and interval; state the measured coverage.
4. What-If: reduce a temperature; show the simulated curve (with the disclaimer).
5. Model Comparison: ablation forest plot and matched-budget lead time; say whether H1 was supported.
6. Fault injection: show the system separating a stuck sensor from real degradation.

---

## 9. Testing and quality gates

### 9.1 Leakage Guard (`tests/leakage_guard/`, S1) — must pass before any result is trusted

| Test | What it proves |
|---|---|
| `test_unit_disjoint_splits` | No `unit_id` appears in more than one of train/val/test/calibration within a fold |
| `test_fit_only_on_train` | Preprocessors, KMeans, sensor selection, calibrators wrap `fit` and assert the input unit ids ⊆ train units |
| `test_feature_causality` | For random *t*, features at *t* computed on the full trajectory equal features computed on the trajectory truncated at *t* (truncation invariance) |
| `test_no_label_in_features` | No feature name or derivation depends on `rul` (blacklist + column provenance) |
| `test_shuffled_label_sanity` | Training on shuffled labels yields ROC-AUC ≈ 0.5 (tolerance) |
| `test_baseline_uses_past_only` | Healthy baseline for unit *u* uses only its first K cycles; warm-up rows flagged |
| `test_calibration_disjoint` | Conformal/probability calibration data are OOF from other folds |
| `test_determinism` | Same seed and config → identical metrics |
| `test_threshold_selected_on_val` | Warning τ and state thresholds never fitted on test units |

### 9.2 Other tests
- Unit tests for each statistic against hand-computed references; SHI bounds; state persistence; monotone enforcement; conformal coverage on synthetic data (empirical coverage ≈ 1−α); What-If identity test; metric implementations vs `scikit-learn` where available.
- Smoke test: full pipeline on the synthetic fixture under the `smoke` profile finishes quickly in CI.
- Lint: `ruff`; formatting consistent; type hints on public interfaces.

### 9.3 Quality gates per phase
A phase is done only when: tests for that phase pass, Leakage Guard passes, artifacts follow the contracts in Section 4.6, and a manifest is written.

---

## 10. Phased build plan (original phases 1–11, plus scaffolding phase 0)

For each phase: **Scope · Deliverables · DoD · Verify**.

### Phase 0 — Scaffolding and guardrails (additive)  · P0
- **Scope:** repo skeleton (Section 4.3), `pyproject.toml`, `Makefile`, config loader, CLI shell, logging, manifest writer, synthetic fixture, Leakage Guard test skeletons, `AGENTS.md`, CI-free local `make test`.
- **DoD:** `make setup && make test` passes on the smoke fixture; empty modules import; config validates.
- **Verify:** `make test`; `stattwin --help`.

### Phase 1 — Dataset ingestion, preprocessing, leakage-safe splitting  · P0
- **Scope:** Sections 6.1 and 6.2 (cleaning, scaling, condition normalization, DQ engine).
- **Deliverables:** loaders for FD001–FD004, splitter, preprocessing pipeline, DQ flags, E0 data audit.
- **DoD:** leakage tests for splits and fit-only-on-train pass; FD001 processed to parquet; E0 outputs generated.
- **Verify:** `make data DS=FD001`, `make e0`, `pytest tests/leakage_guard -k "split or fit"`.

### Phase 2 — Statistical engine and statistical features  · P0
- **Scope:** Section 6.3, full library, causal, vectorized, feature selection (train-only).
- **DoD:** truncation-invariance test passes; features parquet produced for FD001; runtime acceptable.
- **Verify:** `make features DS=FD001`, `pytest -k statistics`.

### Phase 3 — Statistical Health Index and health-state engine  · P0
- **Scope:** Section 6.4, Appendix C; E1.
- **DoD:** SHI in [0, 100]; states with persistence; HI quality metrics table; threshold sensitivity table.
- **Verify:** `make e1`.

### Phase 4 — Baseline ML models  · P0
- **Scope:** Sections 6.5 (models 1–5) and the shared training/evaluation harness (OOF generation, test prediction, metrics).
- **DoD:** OOF and test prediction parquet for all five; per-horizon metrics; NASA score on test.
- **Verify:** `make train MODEL=xgb`, `make e2 -- --models 1-5`.

### Phase 5 — Temporal modelling and STAT-TWIN Hybrid  · P0
- **Scope:** Sections 6.5 (model 6, GRU) and 6.6 (Hybrid); tabular hybrid fallback first if time is short.
- **DoD:** GRU baseline and Hybrid run end to end in `fast`; ensemble seeds; monotone enforcement logged.
- **Verify:** `make train MODEL=gru`, `make train MODEL=hybrid`.

### Phase 6 — Failure probability, RUL, uncertainty, calibration  · P0
- **Scope:** Sections 6.7, 6.8, 6.9; E3, E5.
- **DoD:** curve + RUL + failure point + interval API; coverage reported; reliability diagrams; lead-time benchmark at matched budget.
- **Verify:** `make e3 e5`; `pytest -k "conformal or calibration"`.

### Phase 7 — Explainability and evidence generation  · P0
- **Scope:** Section 6.10.
- **DoD:** evidence cards and group-occlusion attribution for any `(unit, cycle)`; risk-change decomposition; SHAP cross-check if time allows (P1).
- **Verify:** `pytest -k explain`.

### Phase 8 — What-If simulator  · P0
- **Scope:** Section 6.11.
- **DoD:** identity test, disclaimers, OOD warning, comparison output for the dashboard.
- **Verify:** `pytest -k counterfactual`.

### Phase 9 — Research experiments: ablation, generalization  · P0 (E4) / P1 (E6–E9)
- **Scope:** Section 7 — E4 (required), E2 completion, then E6, E7, E8, E9.
- **DoD:** E4 tables and significance results written to `results/e4/`; E6 heatmaps; E8 confusion and false-alarm table.
- **Verify:** `make e4`; `make e6 e7 e8`.

### Phase 10 — Professional Streamlit dashboard  · P0
- **Scope:** Section 8; precomputation script `make app-data`; multipage app; Live Replay.
- **DoD:** all seven pages load from artifacts with no errors; provenance badges present; What-If runs live; app starts with `make app`.
- **Verify:** `make app-data && make app`; manual walkthrough of the demo script (8.3).

### Phase 11 — Final evaluation, documentation, charts, research report, viva preparation  · P0/P1
- **Scope:** run `full` profile; generate report and README results from `results/`; produce `MODEL_CARD.md`, `LIMITATIONS.md`, `ARCHITECTURE.md`, `VIVA.md`.
- **DoD:** `make report` regenerates everything from artifacts; no hand-typed numbers; reproduce command documented; all DoD from earlier phases still green.
- **Verify:** `make reproduce` (or documented equivalent), `make test`.

---

## 11. Milestone checklist

**Foundation**
- [ ] Repo scaffold, config system, CLI, manifest writer (Phase 0)
- [ ] Synthetic fixture and `smoke` profile working
- [ ] FD001–FD004 loaders; unit-level splits verified disjoint (Phase 1)
- [ ] Preprocessing pipeline fitted on train only; DQ flags emitted
- [ ] E0 data audit generated

**Statistics and health**
- [ ] Statistical feature library, causal and tested (Phase 2)
- [ ] Truncation-invariance test passing
- [ ] SHI (0–100) with documented formula and configurable weights (Phase 3)
- [ ] Five health states with persistence; threshold sensitivity table
- [ ] E1 health-index quality metrics (monotonicity, trendability, prognosability)

**Models**
- [ ] Baselines 1–5 with OOF and test predictions (Phase 4)
- [ ] GRU baseline (Phase 5)
- [ ] STAT-TWIN Hybrid with ensemble and monotone horizons
- [ ] E2 model comparison complete

**Forecast, uncertainty, calibration**
- [ ] Multi-horizon probabilities, RUL, failure point (Phase 6)
- [ ] Conformal RUL intervals with measured coverage
- [ ] Reliability diagrams, Brier, ECE, before/after calibration
- [ ] E3 early-warning benchmark at matched false-alarm budget
- [ ] E5 uncertainty and calibration report

**Explain and simulate**
- [ ] Evidence cards and attribution (Phase 7)
- [ ] What-If simulator with disclaimers, OOD warning, identity test (Phase 8)

**Research**
- [ ] E4 ablation A–E with paired tests, bootstrap CIs, Holm correction (Phase 9)
- [ ] H0/H1 conclusion written strictly from results
- [ ] E6 generalization heatmaps; E7 operating-condition study
- [ ] E8 fault-injection benchmark
- [ ] E9 uncertainty comparison (optional)

**Product and delivery**
- [ ] Seven-page dashboard with provenance badges (Phase 10)
- [ ] Live Replay working
- [ ] Model Comparison page fully data-driven
- [ ] Report, README, model card, limitations generated from results (Phase 11)
- [ ] `VIVA.md` complete; demo rehearsed
- [ ] Leakage Guard suite green; `full` profile reproducible from a clean checkout

---

## 12. Reporting and viva preparation

### 12.1 Research report skeleton
1. Abstract 2. Introduction and research question 3. Background (PHM, C-MAPSS, health indicators, conformal prediction) 4. Method (pipeline, statistical engine, SHI, hybrid model, uncertainty) 5. Experimental protocol (splits, leakage controls, metrics) 6. Results (E1–E9) 7. Discussion (does statistics add value, where, and why) 8. Limitations 9. Conclusion 10. Reproducibility appendix (configs, seeds, commands).

**Suggested reading list (verify details before citing):** Saxena et al. (2008) C-MAPSS damage propagation modelling; Vovk et al. and Lei et al. (2018) on conformal / distribution-free inference; Romano et al. (2019) conformalized quantile regression; Barber et al. (2021) jackknife+ / CV+; Guo et al. (2017) calibration of modern neural nets; Coble (2010) health-indicator metrics (monotonicity, trendability, prognosability); Zheng et al. (2017) LSTM for RUL; Lundberg and Lee (2017) SHAP.

### 12.2 Viva cheat-sheet (expand in `docs/VIVA.md`)

| Likely question | Short answer |
|---|---|
| Why not just an LSTM? | LSTM is baseline #6 and the hybrid's backbone; the research question is whether statistical evidence adds value, tested by ablation A–E with paired tests |
| How do you know there is no leakage? | Unit-level GroupKFold, train-only fitting, causal features, and an automated Leakage Guard suite (truncation invariance, shuffled-label sanity) |
| Why conformal prediction? | Distribution-free finite-sample coverage under exchangeability; model-agnostic; measured empirically; degrades under shift, which we quantify |
| Are your probabilities trustworthy? | Reliability diagrams, Brier and ECE before/after isotonic calibration, per horizon |
| Is the health index scientifically valid? | Thresholds are configurable and calibrated on train; SHI is validated with monotonicity/trendability/prognosability and sensitivity analysis; not claimed universal |
| How do you separate sensor faults from degradation? | Persistence + multi-sensor coherence + degradation direction; validated in the fault-injection benchmark |
| What does the What-If simulator really tell us? | Sensitivity of the learned model, not a real maintenance outcome; labelled as simulation with OOD warnings |
| Why is lead time compared at a matched false-alarm budget? | Otherwise a model can "win" by warning constantly |
| What if H1 is not supported? | Report it honestly with CIs and analysis; a null result with rigorous design is valid |
| Does it generalize? | Cross-dataset heatmap; degradation under regime shift measured; exchangeability limits explained |
| Why exclude cycle count as a feature? | Comparability with literature; sensitivity run reported separately |
| Why RUL clipping? | Standard practice: early-life RUL is not identifiable from sensors; both clipped and unclipped metrics reported |
| Limits of C-MAPSS? | Simulated data, few fault modes, no true DQ faults; addressed via fault injection; future work on real telemetry via the dataset adapter |
| What is novel here? | Combination of statistical evidence layer, calibrated probabilities with conformal RUL intervals, matched-budget lead time, and DQ-vs-machine separation in one leakage-audited pipeline |

### 12.3 Limitations to state openly
Simulated dataset; conformal guarantees are marginal and assume exchangeability; state thresholds are conventions unless calibrated; What-If is model sensitivity, not causal intervention; hybrid gains may be dataset-specific; results depend on RUL clipping and horizon choices.

---

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Time overrun | Missing deliverables | Priority tiers, `fast` profile, crunch mode (3.4), tabular-hybrid fallback |
| GRU unstable or slow | Hybrid delayed | Tabular hybrid first; smaller hidden size; fewer seeds |
| Feature explosion / overfit | Poor generalization | Train-only feature selection; regularization; ablation shows contribution |
| H1 not supported | Weaker story | Honest reporting; the rigor, calibration, and DQ experiments still carry the project |
| Silent leakage | Invalid results | Leakage Guard as a hard gate |
| Regime shift on FD002/FD004 | Poor scores | Per-condition normalization; report naive vs condition-aware |
| Dashboard slow | Bad demo | Precomputed artifacts; cache models; only What-If is live |
| Agent invents numbers | Integrity | AGENTS.md rule; reports generated from files; `manifest.json` per run |
| Data files missing | Pipeline blocked | Loud failure with download instructions; synthetic fixture for tests only |

---

## Appendix A — Drop-in `AGENTS.md` for opencode

```markdown
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
```

---

## Appendix B — Suggested `Makefile` targets

| Target | Action |
|---|---|
| `make setup` | Create env, install package in editable mode |
| `make data DS=FD001` | Ingest, split, preprocess, cache parquet |
| `make features DS=FD001` | Statistical features + SHI |
| `make train MODEL=… PROFILE=fast` | Train and produce OOF/test predictions |
| `make eN` | Run experiment N |
| `make app-data` | Precompute dashboard bundles |
| `make app` | Launch Streamlit |
| `make report` | Generate report/README tables and figures from `results/` |
| `make test` | Unit tests + Leakage Guard |
| `make reproduce` | End-to-end `full` profile from clean artifacts |

---

## Appendix C — Statistical Health Index (formal definition)

For engine *u*, cycle *t*, selected sensors *S*, degradation direction `d_i ∈ {+1, −1}` (sign of Spearman(sensor_i, cycle) on train), healthy baseline `(μ0_i, σ0_i)` from that unit's first *K* cycles (causal expanding estimate during warm-up), window *w*:

- Deviation: `dev_i = max(0, d_i · (x_i,t − μ0_i) / max(σ0_i, σ_min_i))`
- Trend: `tr_i = max(0, d_i · slope_w(x_i) · 10 / max(σ0_i, σ_min_i))`
- EWMA: `ew_i = max(0, d_i · (EWMA_i,t − μ0_i) / max(σ0_i, σ_min_i))`
- Variance: `va_i = max(0, log(std_w(x_i) / max(σ0_i, σ_min_i)))`
- Correlation shift: `cs = ‖C_w − C_0‖_F / normalizer` over top-*k* pairs

Each evidence value is mapped to [0, 1] with a train-fitted ECDF (5th–99th percentile clamp). Sensor aggregation for component *k*: `c_k = Σ_i ω_i · ê_k,i / Σ_i ω_i`, with `ω_i` = train-fitted informativeness (e.g., |Spearman| with normalized life fraction).

`SHI_t = 100 · (1 − Σ_k λ_k · c_k,t)`, `Σ λ_k = 1`, then causal EWMA smoothing. Default λ in `configs/base.yaml`; optional NNLS calibration on train life fraction.

Health-state thresholds: **calibrated** (train-derived) with a **fixed-grid** sensitivity variant; persistence of *p* cycles to change state. These are documented conventions for this dataset, not universal standards.

---

## Appendix D — Metric definitions

- **RUL MAE / RMSE:** standard, computed on clipped and unclipped targets.
- **NASA score** (`d = ŷ − y`): `Σ (exp(−d/13) − 1)` if `d < 0`, else `Σ (exp(d/10) − 1)` (late predictions penalized more).
- **Classification:** per horizon `h`, positives `1[rul ≤ h]`; precision, recall, F1, FPR, FNR at validation-chosen threshold; ROC-AUC, PR-AUC threshold-free.
- **Brier score:** mean squared error between predicted probability and outcome, per horizon.
- **ECE:** `Σ_b (n_b / n) · |acc_b − conf_b|` over 10 bins (equal-width and equal-mass reported).
- **PSI:** `Σ_b (p_b − q_b) · ln(p_b / q_b)`, bins from the baseline distribution, small-value smoothing.
- **PICP:** fraction of true values inside the interval; **MPIW:** mean interval width; **interval (Winkler) score:** width plus penalty `(2/α)` × distance outside the interval.
- **Monotonicity** (per unit): `|#(Δ>0) − #(Δ<0)| / (n − 1)`; **prognosability:** `exp(−std(final values) / mean|final − initial|)`; **trendability:** minimum absolute correlation between HI trajectories of different units after time normalization (Coble-style definitions; state the exact variant used).
- **Lead time, false early warning, false-alarm rate:** Section 6.9.
- **Paired testing:** Wilcoxon signed-rank on per-unit paired metrics; paired bootstrap CI (10 000 resamples over units); Holm correction over the pre-declared family; effect sizes reported.

---

## Appendix E — Glossary

**RUL** remaining useful life · **SHI** Statistical Health Index · **OOF** out-of-fold · **EWMA** exponentially weighted moving average · **PSI** population stability index · **KS** Kolmogorov–Smirnov · **PICP** prediction interval coverage probability · **ECE** expected calibration error · **DQ** data quality · **OOD** out of distribution · **Conformal prediction** distribution-free method giving finite-sample coverage under exchangeability.
