# README — `improved_LR.py` (Two-Stage, Compute-Capped Logistic Regression)

A production-ready Logistic Regression pipeline with **standardization**, **two-stage hyperparameter search** (budget-capped with Halving → focused Grid/Random), and the **same plots/metrics** layout as the baseline. All figures are saved headless (`matplotlib.use("Agg")`) and directories are created automatically.  &#x20;

---

## What the script does

1. **Load + dtype downcast**
   Reads Parquet/CSV, casts numerics to `float32` (CSV), extracts `X`, `y`, and feature names; validates target.&#x20;

2. **Split**
   Stratified train/test split with user-set `--test_size` and `--random_state`.&#x20;

3. **Make a CV subset**
   Subsample the training set for search with `--cv_frac` (stratified). &#x20;

4. **Two-stage tuning (if searching)**

   * **Stage 1: HalvingRandomSearchCV** over a **scaled** LR pipeline (`StandardScaler -> LogisticRegression(saga)`), exploring penalty, `C`, optional `l1_ratio` (for elasticnet), class weights, and intercept. **Resources are capped** via `min_resources`/`max_resources` on `n_samples`. Saves CV table + a JSON summary.   &#x20;
   * **Stage 2: Focused search** around the Stage-1 winner using either **grid** or **random** search on a compact neighborhood of `C` and related params (conditionally adding `l1_ratio` if the chosen penalty is elasticnet). Saves CV table + a JSON summary and **best\_params.json**.     &#x20;

   You can also **skip search and reuse** previously saved params via `--mode refit_only` + `--params_json` (defaults to `<results_dir>/best_params.json`). &#x20;

5. **Refit best model**
   Refit the chosen pipeline on a larger slice of train (`--final_frac` of TRAIN).&#x20;

6. **Evaluate on TEST**
   Compute metrics/plots (details below), save artifacts, and print a run summary. &#x20;

7. **Persist a `summary.json`** with data paths, split/search settings, best params, coefficient stats, metrics at 0.5 and at the optimized threshold (if requested), and timings.&#x20;

---

## Installation & requirements

* Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
* The script forces a non-interactive backend so it runs fine on servers/CI.&#x20;

---

## Usage

```bash
# Typical search + evaluate (auto mode)
python improved_LR.py \
  --data data/higgs_train.parquet --target target \
  --cv_frac 0.10 --final_frac 1.0 \
  --n_candidates1 24 --factor 4 --cv_folds 3 --cv_jobs -1 \
  --stage1_min_resources 10000 --stage1_max_resources 250000 \
  --stage2_kind random --stage2_budget 120 \
  --optimize_threshold

# Search only (save params and exit)
python improved_LR.py --data ... --target target --mode search_only

# Reuse saved params (skip search)
python improved_LR.py --data ... --target target --mode refit_only \
  --params_json C:\...\best_params.json
```

**Key arguments** (defaults shown)

* Outputs:
  `--results_dir C:\Projects\U\Pro_fin3\Models\Logistic_Regression\improved\results`
  `--figures_dir C:\Projects\U\Pro_fin3\Models\Logistic_Regression\improved\figures`&#x20;
* Splits & folds: `--test_size 0.2`, `--random_state 42`, `--cv_frac 0.10`, `--final_frac 1.00`, `--cv_folds 3`. &#x20;
* Stage 1 caps: `--n_candidates1 24`, `--factor 4`, `--stage1_min_resources 10000`, `--stage1_max_resources 250000`, `--cv_jobs -1`.&#x20;
* Stage 2: `--stage2_kind {random,grid}` (default `random`), `--stage2_budget 120`.&#x20;
* Evaluation controls: `--metrics_sample_max 2_000_000`, `--optimize_threshold`.&#x20;
* Modes & reuse: `--mode {auto,search_only,refit_only}`, `--params_json` (optional).&#x20;

---

## How the two-stage tuning works (and why)

* **Scaling is built-in**: LR is sensitive to feature scales; the pipeline always starts with `StandardScaler`.&#x20;
* **Stage 1 (Halving)** aggressively prunes weaker configs while **increasing sample resources** each iteration, scored by **ROC-AUC** under a `StratifiedKFold` CV. This keeps compute bounded yet finds promising regions. &#x20;
* **Stage 2** **zooms in** around the winner with a compact grid or random search, including a **conditional** `l1_ratio` **only** when `penalty='elasticnet'`. This avoids invalid combos and wastes less budget. &#x20;

---

## Outputs

**Search artifacts (if searching)**

* `cv_results_stage1.csv`, `search_summary_stage1.json` (Halving results + summary)&#x20;
* `cv_results_stage2.csv`, `search_summary_stage2.json`, `best_params.json` (focused search results, summary, and best params for reuse)&#x20;

**Evaluation artifacts**

* `summary.json` (all run settings, best params, coefficient stats, metrics, timings)&#x20;
* `classification_report_thr0.5.txt` (per-class precision/recall/F1 at thr=0.5)&#x20;
* Confusion matrices (`*.csv` and `*.png`) at 0.5 and, if enabled, at the optimized threshold:
  `confusion_matrix_thr0.5.csv/png`, `confusion_matrix_thrOPT.csv/png`. &#x20;

**Curves & feature visuals**

* `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png`.  &#x20;
* `roc_points.csv`, `pr_points.csv` (raw points).&#x20;
* `feature_importances_top20.png` and `feature_importances_full.csv` (ordered LR coefficients).&#x20;

---

## Metrics (what they mean)

All **threshold-free** metrics use **probabilities**; **thresholded** ones use class predictions (default `0.5`, plus the optional optimized threshold).

* **ROC-AUC** — Discrimination regardless of threshold (higher is better). Computed via `roc_auc_score(ys, ps)`.&#x20;
* **Average Precision (AP)** — Area under Precision-Recall; more informative under class imbalance. From `average_precision_score(ys, ps)`.&#x20;
* **Accuracy** — Overall correctness at the chosen threshold. `accuracy_score(ys, yp)`.&#x20;
* **Balanced accuracy** — Mean recall across classes; robust to imbalance. `balanced_accuracy_score(ys, yp)`.&#x20;
* **Precision** — Of predicted positives, how many are truly positive. `precision_score(ys, yp)`.&#x20;
* **Recall (TPR)** — Of actual positives, how many were found. `recall_score(ys, yp)`.&#x20;
* **F1** — Harmonic mean of precision and recall at the threshold. `f1_score(ys, yp)`.&#x20;

**Optional threshold optimization**
If `--optimize_threshold` is set, the script finds the **Youden-J** point (argmax of `tpr − fpr`) on the ROC curve and recomputes thresholded metrics + a second confusion matrix at that threshold. The chosen threshold appears in `summary.json` and the console.  &#x20;

---

## Plots (how to read them)

* **ROC curve** (`roc_curve.png`)
  TPR vs FPR across thresholds; nearer the **top-left** and higher AUC indicate better ranking. Generated via `RocCurveDisplay.from_predictions`.&#x20;

* **Precision–Recall curve** (`pr_curve.png`)
  Precision vs recall for the positive class; prefer staying high on the **right side** (high recall at high precision). Generated with `PrecisionRecallDisplay.from_predictions`.&#x20;

* **Calibration curve** (`calibration_curve.png`)
  Plots **predicted probability** vs **observed positive fraction** in 20 quantile bins; a well-calibrated model sits near the diagonal.&#x20;

* **Predicted probability histogram** (`proba_hist.png`)
  The distribution of `p(y=1)`; a **bimodal** shape (mass near 0 and 1) suggests confident predictions; mass near 0.5 indicates uncertainty.&#x20;

* **Confusion matrices**
  At **thr=0.5** and at the **optimized threshold** (if enabled). Diagonals are correct predictions; off-diagonals are errors. Saved as both `*.png` and `*.csv`. &#x20;

* **Top coefficients** (`feature_importances_top20.png`)
  20 features with the largest |coefficient| from the **scaled** LR. Positive weights push toward class 1; negative toward class 0. Full ordered list saved to CSV. &#x20;

---

## Differences vs. `raw_LR.py`

* **Scaling + better solver:** `StandardScaler` + `saga` with extended `max_iter` and `tol` for robust convergence; raw uses sklearn defaults with no scaling. &#x20;
* **Two-stage search:** budget-capped Halving → focused grid/random with conditional spaces; raw has **no tuning**. &#x20;
* **Reusable params:** `best_params.json` enables `refit_only` runs.&#x20;

---

## Performance & reproducibility tips

* **Control compute** with `--cv_frac`, `--n_candidates1`, `--factor`, `--stage1_min_resources`, `--stage1_max_resources`, `--stage2_budget`, and parallelism via `--cv_jobs`.&#x20;
* **Reproducibility**: `--random_state` is passed to the splitter, CV, and searches. &#x20;
* **Fast iterations**: search on a **small `cv_frac`**, then `refit_only` with the saved params on a **larger `final_frac`**. &#x20;

---

## Console summary (what you’ll see)

At the end: the number of LR iterations, test ROC-AUC & AP, and (if enabled) the optimized threshold & F1. Also echoes the resolved results/figures paths.&#x20;

---

## File map (at a glance)

```
results/
  cv_results_stage1.csv
  search_summary_stage1.json
  cv_results_stage2.csv
  search_summary_stage2.json
  best_params.json
  summary.json
  classification_report_thr0.5.txt
  confusion_matrix_thr0.5.csv
  confusion_matrix_thrOPT.csv          # if --optimize_threshold
  roc_points.csv
  pr_points.csv
  feature_importances_full.csv

figures/
  roc_curve.png
  pr_curve.png
  calibration_curve.png
  proba_hist.png
  confusion_matrix_thr0.5.png
  confusion_matrix_thrOPT.png          # if --optimize_threshold
  feature_importances_top20.png
```

---