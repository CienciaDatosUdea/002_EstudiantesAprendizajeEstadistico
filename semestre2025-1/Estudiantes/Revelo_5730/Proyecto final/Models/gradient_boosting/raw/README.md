# README — `raw_GBC.py` (Baseline Gradient Boosting Classifier)

A **plain, out-of-the-box** baseline using scikit-learn’s `GradientBoostingClassifier`. No scaling, no tuning — just defaults — so you get a fast, reproducible reference to compare against improved models. Results go to `...\results\` and figures to `...\figures\`.

---

## What this script does

1. **Loads data** from Parquet/CSV, validates the target, and lightly down-casts numeric columns to save memory (CSV only). Returns `X` (DataFrame), `y`, and `feature_names`.

2. **Creates output folders** (`results_dir`, `figures_dir`).

3. **Optionally subsamples** the dataset via a stratified split using `--sample_frac` to speed up experiments.

4. **Train/test split** with stratification and a reproducible seed.

5. **Fits** a default `GradientBoostingClassifier()` (e.g., `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `subsample=1.0`, etc.).

6. **Predicts** classes and probabilities.

7. **Caps the data used for curves** with `--metrics_sample_max` to keep memory sane.

8. **Computes metrics** at the default `0.5` threshold and (optionally) at the **Youden-J optimal** threshold.

9. **Generates plots**: ROC, PR, calibration curve, probability histogram, confusion matrices, and a “Top 20 feature importances” bar chart.&#x20;

10. **Saves artifacts** (CSV curve points, confusion matrices, full importances, classification report) and a comprehensive `summary.json`.&#x20;

---

## Installation & requirements

* Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
* The script forces a **non-interactive backend** (`matplotlib.use("Agg")`) so it can run headless on servers/CI.

---

## Usage

```bash
# Minimal
python raw_GBC.py --data path/to/data.parquet --target target

# With optional knobs
python raw_GBC.py ^
  --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet ^
  --target target ^
  --test_size 0.2 ^
  --random_state 42 ^
  --sample_frac 0.25 ^
  --metrics_sample_max 2000000 ^
  --optimize_threshold ^
  --results_dir C:\Projects\U\Pro_fin3\Models\gradient_boosting\raw\results ^
  --figures_dir C:\Projects\U\Pro_fin3\Models\gradient_boosting\raw\figures
```

**Key arguments** (defaults shown)

* Output folders:
  `--results_dir` = `C:\Projects\U\Pro_fin3\Models\gradient_boosting\raw\results`
  `--figures_dir` = `C:\Projects\U\Pro_fin3\Models\gradient_boosting\raw\figures`

* Split & sampling:
  `--test_size 0.2`, `--random_state 42` (used for split & sampling), `--sample_frac 1.0`.

* Curve memory cap:
  `--metrics_sample_max 2_000_000`.

* Threshold optimization:
  `--optimize_threshold` (adds metrics & confusion matrix at the ROC-optimal threshold).

---

## Outputs

**Results (`results_dir`)**

* `summary.json` — model, data paths, sizes, defaults, metrics at thr=0.5 and at optimal thr (if enabled), timings, and run settings.

* `classification_report_thr0.5.txt` — per-class precision/recall/F1 at thr=0.5.

* `roc_points.csv`, `pr_points.csv` — raw points to recreate ROC and PR curves.&#x20;

* `confusion_matrix_thr0.5.csv`, `confusion_matrix_thrOPT.csv` (if enabled).

* `feature_importances_full.csv` — all features ordered by model importance.&#x20;

**Figures (`figures_dir`)**

* `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png`.

* `confusion_matrix_thr0.5.png`, `confusion_matrix_thrOPT.png` (if enabled).

* `feature_importances_top20.png`.&#x20;

---

## Metrics (what they mean)

All “thresholded” metrics below use **class predictions** at **0.5** unless you enable `--optimize_threshold`. Threshold-free metrics use **probabilities**.

* **Accuracy** — fraction of correct predictions at the chosen threshold.

* **Balanced accuracy** — mean recall across classes (robust to class imbalance).

* **Precision** — of predicted positives, how many were actually positive.

* **Recall (TPR)** — of actual positives, how many were found.

* **F1** — harmonic mean of precision and recall.

* **ROC-AUC** — probability the classifier ranks a random positive above a random negative; threshold-independent.

* **Average Precision (AP)** — area under the Precision-Recall curve; informative for imbalanced problems.

**Optional: optimized operating point**
With `--optimize_threshold`, the script finds the **Youden-J** point `argmax(tpr − fpr)` from the ROC curve, recomputes thresholded metrics there, and saves a second confusion matrix.

---

## Plots (how to read them)

* **ROC curve** (`roc_curve.png`)
  True Positive Rate vs. False Positive Rate across thresholds. **Closer to the top-left** and **higher AUC** indicate better ranking of positives over negatives.

* **Precision–Recall curve** (`pr_curve.png`)
  Precision vs. recall for the positive class. Favor curves that maintain **high precision at high recall**.

* **Calibration curve** (`calibration_curve.png`)
  Compares **predicted probability** to **observed positive fraction** in 20 quantile bins; a well-calibrated model sits near the diagonal.

* **Probability histogram** (`proba_hist.png`)
  Shows the distribution of `p(y=1)`. **Bimodal** shapes (mass near 0 and 1) suggest confident predictions; mass near 0.5 indicates uncertainty.

* **Confusion matrices**
  `confusion_matrix_thr0.5.png` at threshold 0.5 and `confusion_matrix_thrOPT.png` at the optimized threshold (if enabled). **Diagonal** entries are correct; **off-diagonal** are errors.
  (Plot rendering helper: `plot_confusion_matrix(...)`.)

* **Top feature importances** (`feature_importances_top20.png`)
  **Tree-based importances** (`feature_importances_`) ranked descending; the plot shows the top 20. Full ordered list is written to CSV.&#x20;

---

## Console output (quick sanity check)

At the end you’ll see: **fit time**, test **ROC-AUC** & **AP**, and (if enabled) the optimized threshold and **F1** at that threshold, plus the resolved paths for results and figures.

---

## Reproducibility & performance tips

* `--random_state` controls the **split** and the **curve-sampling** only.

* Use `--sample_frac < 1.0` for faster iterations on large datasets; you’ll still get representative curves/metrics on the sampled set.

* Tune `--metrics_sample_max` if you hit memory pressure when building ROC/PR curves.

---

## File map (at a glance)

```
results/
  summary.json
  classification_report_thr0.5.txt
  roc_points.csv
  pr_points.csv
  confusion_matrix_thr0.5.csv
  confusion_matrix_thrOPT.csv      # if --optimize_threshold
  feature_importances_full.csv

figures/
  roc_curve.png
  pr_curve.png
  calibration_curve.png
  proba_hist.png
  confusion_matrix_thr0.5.png
  confusion_matrix_thrOPT.png      # if --optimize_threshold
  feature_importances_top20.png
```

---
