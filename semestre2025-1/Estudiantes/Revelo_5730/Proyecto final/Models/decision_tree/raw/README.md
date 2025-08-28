# README — `raw_DT.py` (Baseline Decision Tree)

A simple, **out-of-the-box** baseline using scikit-learn’s `DecisionTreeClassifier` with **no hyperparameters set**. It trains a single model, evaluates it, and saves **metrics, plots, and artifacts** to disk so you can compare against improved versions. Figures are saved headlessly (`matplotlib.use("Agg")`). Results and figures default to `...\decision_tree\raw\{results,figures}`.  &#x20;

---

## What the script does

1. **Load data**
   Reads Parquet or CSV; for CSV it down-casts numeric feature columns to `float32` (doesn’t alter model defaults). Validates that `--target` exists, returns `X`, `y`, and `feature_names`.&#x20;

2. **Create output folders**
   Ensures `--results_dir` and `--figures_dir` exist before writing. &#x20;

3. **Optional subsample**
   If `--sample_frac < 1.0`, takes a stratified subsample for a quicker baseline.&#x20;

4. **Train/test split**
   Stratified split using `--test_size` and `--random_state`.&#x20;

5. **Fit a pure-defaults Decision Tree**
   Uses `DecisionTreeClassifier()` **as-is** (e.g., `criterion='gini'`, `max_depth=None`, etc.). &#x20;

6. **Predict** labels and class probabilities.&#x20;

7. **Cap the data used for curves** with `--metrics_sample_max` to keep memory under control.&#x20;

8. **Compute metrics @ thr=0.5** and write a confusion matrix (PNG + CSV).&#x20;

9. **(Optional) Optimize the threshold**
   If `--optimize_threshold`, compute **Youden-J** threshold from the ROC curve and re-report thresholded metrics + a second confusion matrix. &#x20;

10. **Curves & plots**
    Saves ROC, PR, calibration curve, and probability histogram.  &#x20;

11. **Feature importances**
    Plots top-20 Gini importances and saves the full ordered list. &#x20;

12. **Save ROC/PR points** to CSV for external re-plotting.&#x20;

13. **Persist outputs + console summary**
    Writes `summary.json` (incl. **tree depth**, **leaves**, **node count**), `classification_report_thr0.5.txt`, and prints a short recap (AUC, AP, fit time, dirs).  &#x20;

---

## Installation & requirements

* Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
* Non-interactive Matplotlib backend is set so it runs fine on servers/CI.&#x20;

---

## Usage

```bash
# Minimal
python raw_DT.py --data path/to/data.parquet --target target

# With optional knobs (Windows example)
python raw_DT.py ^
  --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet ^
  --target target ^
  --test_size 0.2 ^
  --random_state 42 ^
  --sample_frac 1.0 ^
  --metrics_sample_max 2000000 ^
  --optimize_threshold ^
  --results_dir C:\Projects\U\Pro_fin3\Models\decision_tree\raw\results ^
  --figures_dir C:\Projects\U\Pro_fin3\Models\decision_tree\raw\figures
```

**Key arguments** (defaults shown)

* `--data` (required), `--target target`: input file and target column.&#x20;
* `--results_dir`, `--figures_dir`: where to save artifacts/plots.&#x20;
* `--test_size 0.2`, `--random_state 42`: split control; `random_state` affects split & sampling only. &#x20;
* `--sample_frac 1.0`: optional stratified subsample before splitting. &#x20;
* `--metrics_sample_max 2_000_000`: cap test points used for curves to limit memory. &#x20;
* `--optimize_threshold`: also report metrics at the ROC-optimal (Youden-J) threshold.&#x20;

---

## Outputs

**Results (in `--results_dir`)**

* `summary.json` — model, paths, sizes, **tree stats** (depth/leaves/nodes), metrics at `thr=0.5` and at the optimized threshold (if enabled), timings, and run settings.&#x20;
* `classification_report_thr0.5.txt` — per-class precision/recall/F1 at thr=0.5.&#x20;
* `confusion_matrix_thr0.5.csv`, `confusion_matrix_thrOPT.csv` (if optimized). &#x20;
* `roc_points.csv`, `pr_points.csv` — raw curve points.&#x20;
* `feature_importances_full.csv` — full ordered list of features with Gini importance.&#x20;

**Figures (in `--figures_dir`)**

* `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png`. &#x20;
* `confusion_matrix_thr0.5.png`, `confusion_matrix_thrOPT.png` (if optimized). &#x20;
* `feature_importances_top20.png` — top-20 Gini importances.&#x20;

---

## Metrics (what they mean)

All **thresholded** metrics use class predictions at **0.5** unless you enable `--optimize_threshold`. **Threshold-free** metrics use predicted probabilities.

* **Accuracy** — share of correct predictions at the chosen threshold.
* **Balanced accuracy** — mean recall across classes (robust to imbalance).
* **Precision** — of predicted positives, fraction that are truly positive.
* **Recall (TPR)** — of actual positives, fraction found.
* **F1** — harmonic mean of precision & recall.
* **ROC-AUC** — threshold-free rank quality.
* **Average Precision (AP)** — area under Precision–Recall curve.
  (Computed exactly as coded in `metrics = {...}`.)&#x20;

**Optimized operating point (optional)**
When `--optimize_threshold` is set, the script finds the **Youden-J** threshold `argmax(tpr − fpr)` from the ROC curve, recomputes thresholded metrics, and saves a second confusion matrix at that threshold.  &#x20;

---

## Plots (how to read them)

* **ROC curve** — TPR vs FPR across thresholds; closer to the **top-left** and higher AUC indicate better discrimination.&#x20;
* **Precision–Recall curve** — precision vs recall for the positive class; aim to keep precision high as recall increases.&#x20;
* **Calibration curve** — predicted probability vs observed positive fraction (20 quantile bins); a calibrated model lies near the diagonal.&#x20;
* **Probability histogram** — distribution of `p(y=1)`; bimodal shapes mean confident predictions, mass near 0.5 suggests uncertainty.&#x20;
* **Confusion matrices** — at 0.5 and at the optimized threshold (if enabled). The helper `plot_confusion_matrix` draws labels, counts, and a colorbar. &#x20;

---

## Decision-tree specifics

* The script logs **tree depth**, **number of leaves**, and **node count** into `summary.json` and prints a compact summary to the console (depth, leaves, fit time). &#x20;

---

## Reproducibility & performance tips

* `--random_state` only affects the split and sampling (not the tree’s deterministic fit).&#x20;
* Use `--sample_frac < 1.0` to iterate faster on large datasets; metrics/curves are computed on the sampled set.&#x20;
* Tune `--metrics_sample_max` if you hit memory pressure while building ROC/PR curves.&#x20;

---

## File map (at a glance)

```
results/
  summary.json
  classification_report_thr0.5.txt
  confusion_matrix_thr0.5.csv
  confusion_matrix_thrOPT.csv      # if --optimize_threshold
  roc_points.csv
  pr_points.csv
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

&#x20;  &#x20;

---
