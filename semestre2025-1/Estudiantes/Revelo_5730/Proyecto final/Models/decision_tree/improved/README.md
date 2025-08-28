# README — `improved_DT.py` (Two-stage, compute-capped Decision Tree)

A fast-tunable DecisionTreeClassifier pipeline with **two-stage hyperparameter search** (Stage-1: Halving Random Search; Stage-2: focused Grid/Random), the **same plots/metrics** layout as your baselines, and headless plotting (`matplotlib.use("Agg")`). Results go to `...\improved\results` and figures to `...\improved\figures`. &#x20;

---

## What the script does

1. **Load data**
   Reads Parquet/CSV; for CSV it down-casts numeric feature columns to `float32`. Validates `--target`, returns `X`, `y`, and `feature_names`.&#x20;

2. **Split**
   Stratified train/test split with `--test_size` and `--random_state`.&#x20;

3. **Make a CV subset**
   Subsample TRAIN for search with `--cv_frac` (stratified) and build a `StratifiedKFold` using `--cv_folds` and the same seed.&#x20;

4. **Stage-1: HalvingRandomSearchCV (budget-capped)**
   Searches a broad DT space (criterion, max\_depth, min\_samples\_\*, max\_features, class\_weight) while **increasing sample resources** each iteration; saves `cv_results_stage1.csv` and a `search_summary_stage1.json`.   &#x20;

5. **Stage-2: Focused search (random or grid)**
   Builds a compact neighborhood around the Stage-1 winner (e.g., `max_depth` ± `--md_window`) and runs `GridSearchCV` **or** `RandomizedSearchCV`; saves `cv_results_stage2.csv` and `search_summary_stage2.json`.   &#x20;

6. **Final refit on TRAIN(`--final_frac`)**
   Optionally subsample TRAIN again for the final fit, then train the best model.&#x20;

7. **Evaluate on TEST**
   Computes metrics/plots, writes confusion matrices (thr=0.5 and optional optimized thr), calibration, probability histogram, ROC/PR curves, and raw curve points.      &#x20;

8. **Feature importances (Gini)**
   Plots top-20 importances and saves the full ordered list.&#x20;

9. **Persist a run summary + params**
   Stores `summary.json` (splits, caps, best params, tree stats, metrics, timings) and `best_params.json`.&#x20;

10. **Console recap**
    Prints tree depth/leaves, test ROC-AUC & AP, and (if optimized) the threshold & F1, plus resolved output paths.&#x20;

---

## Installation & requirements

* Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
* Uses a **non-interactive** Matplotlib backend so plots save on servers/CI.&#x20;

---

## Usage

```bash
# Minimal
python improved_DT.py --data path/to/data.parquet --target target

# Typical tuned run (Windows example)
python improved_DT.py ^
  --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet ^
  --target target ^
  --test_size 0.2 ^
  --random_state 42 ^
  --cv_frac 0.10 ^
  --final_frac 1.00 ^
  --n_candidates1 24 ^
  --factor 4 ^
  --cv_folds 3 ^
  --cv_jobs -1 ^
  --stage1_min_resources 10000 ^
  --stage1_max_resources 250000 ^
  --stage2_kind random ^
  --stage2_budget 120 ^
  --md_window 2 ^
  --metrics_sample_max 2000000 ^
  --optimize_threshold ^
  --results_dir C:\Projects\U\Pro_fin3\Models\decision_tree\improved\results ^
  --figures_dir C:\Projects\U\Pro_fin3\Models\decision_tree\improved\figures
```

**Key arguments** (defaults shown)

* IO & layout:
  `--results_dir` and `--figures_dir` (Windows-style defaults).&#x20;

* Splits & folds:
  `--test_size 0.2`, `--random_state 42`, `--cv_frac 0.10`, `--final_frac 1.00`, `--cv_folds 3`. &#x20;

* Stage-1 caps:
  `--n_candidates1 24`, `--factor 4`, `--stage1_min_resources 10000`, `--stage1_max_resources 250000`, `--cv_jobs -1`.&#x20;

* Stage-2 focus:
  `--stage2_kind {random,grid}` (default `random`), `--stage2_budget 120`, `--md_window 2`.&#x20;

* Evaluation:
  `--metrics_sample_max 2_000_000`, `--optimize_threshold`.&#x20;

---

## Outputs

**Search artifacts**

* `cv_results_stage1.csv`, `search_summary_stage1.json` (Halving results + summary). &#x20;
* `cv_results_stage2.csv`, `search_summary_stage2.json` (focused search results + summary).&#x20;

**Evaluation artifacts**

* `summary.json` (splits/caps, best params, tree stats, metrics at 0.5 and at optimized thr, timings).&#x20;
* `classification_report_thr0.5.txt` (per-class precision/recall/F1 at thr=0.5).&#x20;
* Confusion matrices (`*.csv` and `*.png`): `confusion_matrix_thr0.5` and (if enabled) `confusion_matrix_thrOPT`. &#x20;
* Curves & points: `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png`, plus `roc_points.csv`/`pr_points.csv`.   &#x20;
* Feature visuals: `feature_importances_top20.png`, `feature_importances_full.csv`.&#x20;

---

## Metrics (what they mean)

All **thresholded** metrics use class predictions (`p≥thr`) with **thr=0.5** by default; **threshold-free** metrics use probabilities:

* **Accuracy** — overall correctness at the threshold.
* **Balanced accuracy** — mean recall across classes (handles imbalance).
* **Precision** — of predicted positives, fraction truly positive.
* **Recall (TPR)** — of actual positives, fraction found.
* **F1** — harmonic mean of precision & recall.
* **ROC-AUC** — threshold-free discrimination (rank quality).
* **Average Precision (AP)** — area under the PR curve (threshold-free).&#x20;

**Optional optimized operating point**
With `--optimize_threshold`, the script finds the **Youden-J** threshold `argmax(tpr − fpr)` on the ROC curve, recomputes thresholded metrics there, and writes a second confusion matrix. &#x20;

---

## Plots (how to read them)

* **ROC curve** — TPR vs FPR across thresholds; closer to the **top-left** and higher AUC indicate better ranking.&#x20;
* **Precision–Recall curve** — precision vs recall for the positive class; prefer high precision at high recall.&#x20;
* **Calibration curve** — predicted probability vs observed positive fraction (20 quantile bins); a calibrated model lies near the diagonal.&#x20;
* **Probability histogram** — distribution of `p(y=1)`; bimodal shapes → confident predictions, mass near 0.5 → uncertainty.&#x20;
* **Confusion matrices** — images + CSVs for thr=0.5 and (if enabled) the optimized threshold; cells are annotated and color-mapped. &#x20;
* **Top feature importances** — top-20 by Gini; full ordered list saved to CSV.&#x20;

---

## How the two-stage tuning works (and why)

* **Stage-1 (Halving Random Search)** explores a broad DT space and **prunes** weaker configs while ramping up `n_samples`; scored via ROC-AUC with stratified folds. This caps compute while locating promising regions. &#x20;

* **Stage-2 (focused)** zooms in near the Stage-1 winner: it fixes some choices (e.g., `criterion`, `class_weight`) and searches a **small neighborhood** for `max_depth`, `min_samples_leaf`, `min_samples_split`, and `max_features`, via **grid** or **random** search.  &#x20;

---

## Performance & reproducibility tips

* **Control compute** with `--cv_frac`, Stage-1 caps (`--n_candidates1`, `--factor`, `--stage1_*resources`), Stage-2 knobs (`--stage2_kind`, `--stage2_budget`, `--md_window`), and parallelism via `--cv_jobs`. &#x20;
* **Keep memory in check** by capping curve calculations using `--metrics_sample_max` (the script samples indices before computing curves). &#x20;
* **Reproducibility**: `--random_state` seeds splits, CV, and searches. &#x20;

---

## Console summary (what you’ll see)

At the end: **tree depth/leaves**, test **ROC-AUC & AP**, and (if optimized) `thr_opt` with `F1_opt`. It also echoes the resolved results/figures paths.&#x20;

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

&#x20;    &#x20;

---

## Example: quick run

```powershell
python C:\Projects\U\Pro_fin3\Models\decision_tree\improved\improved_DT.py `
  --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet `
  --target target `
  --test_size 0.2 --random_state 42 `
  --cv_frac 0.10 --final_frac 1.00 `
  --n_candidates1 24 --factor 4 --cv_folds 3 --cv_jobs -1 `
  --stage1_min_resources 10000 --stage1_max_resources 250000 `
  --stage2_kind random --stage2_budget 120 --md_window 2 `
  --metrics_sample_max 2000000 --optimize_threshold
```

---