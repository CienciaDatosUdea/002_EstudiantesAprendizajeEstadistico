# Model comparison report

**Dataset:** HIGGS (11,000,000 rows; 28 features + 1 target).
**Goal:** Detect signal over background. We compare Logistic Regression (LR), Decision Tree (DT), and Gradient Boosting (GBC), both raw and improved.

**Ranking metric:** `roc_auc`

## Metrics summary

| family   | variant                 |   roc_auc |   pr_auc |   accuracy_thr0.5 |   precision_thr0.5 |   recall_thr0.5 |   f1_thr0.5 |   accuracy_thrOPT |   precision_thrOPT |   recall_thrOPT |   f1_thrOPT |   threshold_opt |   fit_time |   predict_time |   total_time |
|:---------|:------------------------|----------:|---------:|------------------:|-------------------:|----------------:|------------:|------------------:|-------------------:|----------------:|------------:|----------------:|-----------:|---------------:|-------------:|
| DT       | raw                     |  0.66786  | 0.638965 |          0.668505 |           0.69057  |        0.67861  |    0.684538 |        nan        |         nan        |      nan        |  nan        |      nan        |    662.822 |            nan |      677.148 |
| DT       | improved                |  0.794045 | 0.809864 |          0.717452 |           0.733805 |        0.732679 |    0.733241 |          0.716772 |           0.746533 |        0.704963 |    0.725153 |        0.529052 |    383.985 |            nan |      960.289 |
| GBC      | raw                     |  0.790082 | 0.808194 |          0.713862 |           0.724675 |        0.74204  |    0.733255 |        nan        |         nan        |      nan        |  nan        |      nan        |   6318.88  |            nan |     6341.67  |
| GBC      | improved/best           |  0.829674 | 0.844277 |          0.747075 |           0.757688 |        0.768473 |    0.763042 |          0.745945 |           0.773461 |        0.736207 |    0.754374 |        0.531094 |     60.523 |            nan |      nan     |
| GBC      | improved/final_all_data |  0.829523 | 0.84437  |          0.747066 |           0.757693 |        0.768547 |    0.763081 |          0.746604 |           0.76518  |        0.75297  |    0.759026 |        0.514745 |    121.35  |            nan |      nan     |
| LR       | raw                     |  0.683895 | 0.683525 |          0.64135  |           0.639283 |        0.741954 |    0.686802 |        nan        |         nan        |      nan        |  nan        |      nan        |     20.06  |            nan |       36.811 |
| LR       | improved                |  0.683932 | 0.683856 |          0.637338 |           0.655919 |        0.664113 |    0.65999  |          0.64013  |           0.647882 |        0.703168 |    0.674394 |        0.485712 |     58.11  |            nan |      109.296 |

## Ranking

| family   | variant                 |   roc_auc |
|:---------|:------------------------|----------:|
| GBC      | improved/best           |  0.829674 |
| GBC      | improved/final_all_data |  0.829523 |
| DT       | improved                |  0.794045 |
| GBC      | raw                     |  0.790082 |
| LR       | improved                |  0.683932 |
| LR       | raw                     |  0.683895 |
| DT       | raw                     |  0.66786  |

## Raw vs Improved (per family)

| family   |   roc_auc (raw) |   roc_auc (improved) |   Δ roc_auc |
|:---------|----------------:|---------------------:|------------:|
| LR       |        0.683895 |             0.683932 |    3.7e-05  |
| DT       |        0.66786  |             0.794045 |    0.126185 |
| GBC      |        0.790082 |             0.829523 |    0.039441 |

- **Delta bar**: `figures/delta_roc_auc_by_family.png`
- Per-family figures are under `figures/raw_vs_improved/` (ROC, PR, and metric bars).

## Figures
- **ROC overlay (best-of-each):** `figures/roc_overlay_best_of_each.png`
- **PR overlay (best-of-each):** `figures/pr_overlay_best_of_each.png`
- **ROC overlay (raw):** `figures/roc_overlay_raw.png`
- **PR overlay (raw):** `figures/pr_overlay_raw.png`
- **ROC overlay (improved):** `figures/roc_overlay_improved.png`
- **PR overlay (improved):** `figures/pr_overlay_improved.png`
- **Feature importances (top):** `figures/feature_importances_top.png`

