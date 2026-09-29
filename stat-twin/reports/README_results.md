# STAT-TWIN Research Results

_Auto-generated from `results/*/e*_results.json`. Do not hand-edit._

## E0: Data Audit

**Dataset:** FD001  
**Units:** 100  
**Rows:** 20631  

| Metric | Value |
|--------|-------|
| Cycles/unit (mean ± std) | 206.3 ± 46.1 |
| Cycles/unit (min–max) | 128–362 |
| RUL range | 0.0–361.0 |
| Constant sensors | sensor_1, sensor_10, sensor_18, sensor_19 |
| Operating regimes | 3 |

## E1: SHI Health-Index Validation

| Metric | Value |
|--------|-------|
| Monotonicity (mean) | 1.0000 |
| Trendability | 1.0000 |
| Prognosability | 2.3089 |
| Spearman ρ (mean) | 0.8152 |

## E2: Model Comparison

| Model | ROC-AUC h10 | ROC-AUC h20 | ROC-AUC h30 | ROC-AUC h40 | ROC-AUC h50 | RUL RMSE | RUL MAE | NASA |
|-------|---------|---------|---------|---------|---------|----------|---------|------|
| LogisticModel | 0.9637 | 0.9866 | 0.9864 | 0.9899 | 0.9844 | 19.4943 | 11.9469 | 20.8866 |
| RandomForestModel | 0.9385 | 0.9475 | 0.9360 | 0.9463 | 0.9553 | 16.4795 | 10.8528 | 6.7590 |
| XGBoostModel | 0.9608 | 0.9651 | 0.9528 | 0.9407 | 0.9387 | 15.1573 | 10.0607 | 4.6699 |
| ThresholdModel | 0.9803 | 0.9820 | 0.9772 | 0.9653 | 0.9479 | 41.6896 | 36.9820 | 308.3416 |
| EnsembleModel | 0.9792 | 0.9894 | 0.9893 | 0.9912 | 0.9865 | 15.1120 | 9.8595 | 5.7223 |
| GRUModel | 0.9976 | 0.9982 | 0.9962 | 0.9944 | 0.9919 | 14.9887 | 11.7939 | 3.8157 |
| LSTMModel | 0.9980 | 0.9976 | 0.9949 | 0.9934 | 0.9908 | 15.0913 | 11.6511 | 4.4896 |
| HybridModel | 0.9930 | 0.9951 | 0.9923 | 0.9912 | 0.9871 | 15.5358 | 11.5664 | 4.4053 |

## E3: Early Warning Benchmark

**Horizon:** 30  
**FAR budget:** 0.05  

| Model | Mean Lead Time | Median Lead Time | FAR |
|-------|---------------|-----------------|-----|
| LogisticModel | 20.6923 | 18.0000 | 0.0000 |
| RandomForestModel | 15.5278 | 12.0000 | 0.0000 |
| XGBoostModel | 9.5556 | 5.0000 | 0.0000 |

## E4: Ablation Study

| Variant | Features | Mean MAE | Std MAE |
|---------|----------|----------|---------|
| A_raw | 24 | 30.5831 | 14.2614 |
| B_raw_stat | 465 | 27.2590 | 14.7341 |
| C_raw_temporal | 612 | 26.2535 | 14.3966 |
| D_raw_stat_temporal | 1053 | 26.0439 | 14.5044 |
| E_full | 1117 | 24.4454 | 14.3308 |

**Holm–Bonferroni significant differences (vs E_full):**
  - E_full_vs_A_raw: adjusted p = 0.0000
  - E_full_vs_B_raw_stat: adjusted p = 0.0000
  - E_full_vs_D_raw_stat_temporal: adjusted p = 0.0000
  - E_full_vs_C_raw_temporal: adjusted p = 0.0000

## E5: Uncertainty & Calibration

**Mean Brier:** 0.0380  
**Mean ECE (equal-width):** 0.0337  

### Per-Horizon Calibration

| Horizon | Brier | ECE (EW) | ECE (EM) |
|---------|-------|----------|----------|
| h10 | 0.0197 | 0.0193 | 0.0175 |
| h20 | 0.0268 | 0.0259 | 0.0242 |
| h30 | 0.0371 | 0.0343 | 0.0265 |
| h40 | 0.0485 | 0.0426 | 0.0421 |
| h50 | 0.0583 | 0.0464 | 0.0399 |

### Conformal Intervals

- Alpha: 0.1
- Quantile q: 34.6564
- Coverage: 0.9703
- Mean width: 66.5360

## E6: Cross-Dataset Generalization

### ROC-AUC Heatmap (Train → Test)

| Train \ Test |FD001 | FD002 | FD003 | FD004 |
|--------------|------ | ------ | ------ | ------ |
| FD001 | 0.9978 | 0.6061 | 0.8367 | 0.5991 |
| FD002 | 0.9822 | 0.9909 | 0.9329 | 0.9128 |
| FD003 | 0.9521 | 0.5651 | 0.9983 | 0.5672 |
| FD004 | 0.9791 | 0.9764 | 0.9831 | 0.9931 |

### RUL RMSE Heatmap (Train → Test)

| Train \ Test |FD001 | FD002 | FD003 | FD004 |
|--------------|------ | ------ | ------ | ------ |
| FD001 | 49.5623 | 49.6016 | 52.4306 | 52.3607 |
| FD002 | 49.5623 | 49.6016 | 52.4306 | 52.3607 |
| FD003 | 49.5623 | 49.6016 | 52.4306 | 52.3607 |
| FD004 | 49.5623 | 49.6016 | 52.4306 | 52.3607 |

## E7: Per-Condition Normalization

| Setting | ROC-AUC (h30) | RUL RMSE |
|----------|---------------|----------|
| Without normalization | 0.9456 | 49.5474 |
| With normalization | 0.9460 | 49.5474 |

## E8: Fault Injection

| Fault Type | Precision | Recall | F1 |
|------------|-----------|--------|-----|
| spike | 0.0582 | 0.9833 | 0.1098 |
| stuck | 0.0628 | 0.9600 | 0.1180 |

## E9: Uncertainty Method Comparison

**Alpha:** 0.1  

| Method | PICP | Mean Width | Winkler |
|--------|------|------------|---------|
| split_conformal | 0.9132 | 164.4616 | 192.9153 |
| ensemble_variance | 0.8972 | 66.4035 | 90.7243 |
| quantile_bootstrap | 0.9017 | 71.1835 | 93.6894 |
| bootstrap | 0.8693 | 64.3776 | 96.4030 |
