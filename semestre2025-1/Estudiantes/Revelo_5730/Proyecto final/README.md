# Final Project

**Goal:** build a clear, reproducible pipeline to **compare several classifiers on the same dataset** and pick the **best overall model** based on robust, threshold-independent metrics — primarily **ROC-AUC** (and **Average Precision** when class imbalance matters). All runs save **metrics, plots, and JSON summaries**

---

## Models included

Each model has a **raw (baseline)** and an **improved (tuned)** script, all producing the **same plots and metrics** for apples-to-apples comparison.

| Family              | Baseline                                   | Tuned                                                |
| ------------------- | ------------------------------------------ | ---------------------------------------------------- |
| Logistic Regression | `Models/Logistic_Regression/raw/raw_LR.py` | `Models/Logistic_Regression/improved/improved_LR.py` |
| Gradient Boosting   | `Models/gradient_boosting/raw/raw_GBC.py`  | `Models/gradient_boosting/improved/improved_GBC.py`  |
| Decision Tree       | `Models/decision_tree/raw/raw_DT.py`       | `Models/decision_tree/improved/improved_DT.py`       |

> All scripts read **CSV or Parquet** (preferred) and write artifacts to `.../results/` and figures to `.../figures/`. Improved scripts add search artifacts (CV tables, best params).

---

## Data

* Expected **binary classification** dataset (e.g., HIGGS).
* By default the **target** is the `target` column; override with `--target`.
* Convert CSV → Parquet (faster I/O, lower RAM) with:

  ```powershell
  uv run python data\to_parquet.py --in data\HIGGS.csv --out data\higgs_train.parquet --target target --chunksize 1500000 --compression snappy
  ```

---

## Quick start

1. **Create folders (optional):**

```powershell
mkdir Models\Logistic_Regression\{raw,improved}\{results,figures} -ea 0
mkdir Models\gradient_boosting\{raw,improved}\{results,figures} -ea 0
mkdir Models\decision_tree\{raw,improved}\{results,figures} -ea 0
```

2. **Run baselines** (no tuning):

```powershell
# Logistic Regression (baseline)
uv run python Models\Logistic_Regression\raw\raw_LR.py `
  --data data\higgs_train.parquet --target target `
  --optimize_threshold

# Gradient Boosting (baseline)
uv run python Models\gradient_boosting\raw\raw_GBC.py `
  --data data\higgs_train.parquet --target target `
  --optimize_threshold

# Decision Tree (baseline)
uv run python Models\decision_tree\raw\raw_DT.py `
  --data data\higgs_train.parquet --target target `
  --optimize_threshold
```

3. **Run improved (tuned) versions**:

```powershell
# Logistic Regression (two-stage search; refits best model)
uv run python Models\Logistic_Regression\improved\improved_LR.py `
  --data data\higgs_train.parquet --target target `
  --cv_frac 0.10 --final_frac 1.00 `
  --n_candidates1 24 --factor 4 --cv_folds 3 --cv_jobs -1 `
  --stage1_min_resources 10000 --stage1_max_resources 250000 `
  --stage2_kind random --stage2_budget 120 `
  --optimize_threshold

# Gradient Boosting (HGB recommended; search then BEST & FINAL runs)
uv run python Models\gradient_boosting\improved\improved_GBC.py `
  --data data\higgs_train.parquet --target target --feature_set all `
  --algo hgb --search random --n_iter 60 --cv 3 --cv_frac 0.2 --cv_jobs 2 `
  --sample_frac 0.25 --test_size 0.2 --metrics_sample_max 2000000 `
  --optimize_threshold --save_points --random_state 42

# Decision Tree (two-stage search; focused neighborhood)
uv run python Models\decision_tree\improved\improved_DT.py `
  --data data\higgs_train.parquet --target target `
  --test_size 0.2 --random_state 42 `
  --cv_frac 0.10 --final_frac 1.00 `
  --n_candidates1 24 --factor 4 --cv_folds 3 --cv_jobs -1 `
  --stage1_min_resources 10000 --stage1_max_resources 250000 `
  --stage2_kind random --stage2_budget 120 --md_window 2 `
  --metrics_sample_max 2000000 --optimize_threshold
```

---

## How we decide the **best** classifier


1. **Primary metric:** highest **ROC-AUC** on the **same test split**.
2. **If close (<\~0.002 difference):** higher **Average Precision (AP)**.
3. **Operating point quality:** higher **F1** at the **optimized threshold (Youden-J)**, plus sanity-check **confusion matrices**.
4. **Calibration (visual):** probability calibration curve closer to the diagonal.
5. **Simplicity / inference cost:** if metrics are effectively tied, then **simpler** or **faster** models.

> Why ROC-AUC first? It’s threshold-free and robust for overall ranking. On imbalanced data, AP is also very informative — that’s why it’s our tie-breaker.

---

## Outputs

Each run writes the same core artifacts:

* **`results/summary.json`** — inputs, splits, (best) hyperparameters, timings, metrics at **thr=0.5** and at **optimal thr** (if enabled).
* **`classification_report_thr0.5.txt`** — per-class precision/recall/F1.
* **Confusion matrices** — `confusion_matrix_thr0.5.{png,csv}` and (if enabled) `confusion_matrix_thrOPT.{png,csv}`.
* **Curves** — `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png` (+ optional `roc_points.csv`, `pr_points.csv`).
* **Feature interpretation**

  * LR: `feature_importances_*` are **coefficients** (direction + magnitude).
  * Trees/GB: **feature importances** (or permutation importances for HGB in `improved_GBC.py`).

---

## Plots & metrics (recap)

* **ROC curve / ROC-AUC** — threshold-free ranking; curve nearer **top-left** is better.
* **Precision–Recall / AP** — especially useful under positive-class scarcity.
* **Calibration curve** — how well predicted probabilities match observed frequencies.
* **Probability histogram** — uncertainty vs confidence (mass near 0.5 indicates ambiguity).
* **Confusion matrices** — error profile at **0.5** and at **Youden-J** optimized threshold.
* **F1 / Precision / Recall / (Balanced) Accuracy** — operating-point metrics.

---

## Reproducibility & performance

* Use **Parquet** data and set a consistent `--random_state`.
* For quick iterations on large data, use `--sample_frac` (baselines) or search on a **small `--cv_frac`**, then **refit on all data** (`improved_*` scripts).
* Memory: cap curve computations with `--metrics_sample_max`.
* Parallelism: tune `--cv_jobs` on improved scripts during search.

---

## Interpreting models

* **Logistic Regression:**
  Coefficients show **direction** (sign) and **strength** (magnitude). Large |coef| → stronger influence on the log-odds. Compare **top-20** bar chart; use the full CSV for exact values.

* **Gradient Boosting / Decision Tree:**
  Feature importances show which variables the trees split on most. In `improved_GBC.py` with HGB, permutation importance is used when native importances aren’t exposed; it more directly measures **predictive contribution**.

---

## Project folder map

```
C:\Projects\U\Pro_fin3\
├─ data\
│  ├─ HIGGS.csv
│  ├─ higgs_train.parquet
│  └─ to_parquet.py
├─ Models\
│  ├─ Logistic_Regression\
│  │  ├─ raw\{raw_LR.py, results\, figures\}
│  │  └─ improved\{improved_LR.py, results\, figures\}
│  ├─ gradient_boosting\
│  │  ├─ raw\{raw_GBC.py, results\, figures\}
│  │  └─ improved\{improved_GBC.py, results\, figures\}
│  └─ decision_tree\
│     ├─ raw\{raw_DT.py, results\, figures\}
│     └─ improved\{improved_DT.py, results\, figures\}
|─ README.md   

```
