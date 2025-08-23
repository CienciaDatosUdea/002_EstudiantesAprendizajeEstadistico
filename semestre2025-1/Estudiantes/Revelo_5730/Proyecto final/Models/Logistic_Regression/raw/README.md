# README — `raw_LR.py` (Baseline Logistic Regression)

A simple, **out-of-the-box** baseline for binary classification using scikit-learn’s `LogisticRegression`. No scaling, no tuning — just defaults — so you have a fast, reproducible reference to compare against your improved models. Results are written to `...\results\` and figures to `...\figures\`.&#x20;

---

## What this script does (in one pass)

1. **Loads data** from Parquet or CSV, checks the target column, and lightly down-casts numeric feature dtypes to save memory. Returns `X`, `y`, and feature names.&#x20;
2. **Optionally subsamples** the dataset (`--sample_frac`) to speed up experiments.&#x20;
3. **Train/test split** with stratification.&#x20;
4. **Fits** a plain `LogisticRegression()` with **pure defaults** (e.g., `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=100`).&#x20;
5. **Predicts** labels and probabilities.&#x20;
6. **Down-samples predictions** (if huge) for curve plotting using `--metrics_sample_max`.&#x20;
7. **Computes metrics** at the default `0.5` threshold and (optionally) at the **Youden-J optimal threshold**.  &#x20;
8. **Generates plots**: ROC, PR, calibration curve, probability histogram, confusion matrices, and a “top coefficients” bar chart.  &#x20;
9. **Saves artifacts** (`summary.json`, `classification_report_thr0.5.txt`, CSVs with curve points, confusion matrices, and full coefficient list) and prints a brief run summary.    &#x20;

---

## Installation & requirements

* Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
* The script sets Matplotlib’s non-interactive backend (`Agg`) so it can run headless and still save figures.&#x20;

---

## Usage

```bash
# Minimal
python raw_LR.py --data path/to/data.parquet --target target

# With optional knobs
python raw_LR.py \
  --data data/higgs_train.parquet \
  --target target \
  --test_size 0.2 \
  --random_state 42 \
  --sample_frac 0.25 \
  --metrics_sample_max 2000000 \
  --optimize_threshold \
  --results_dir C:\Projects\U\Pro_fin3\Models\Logistic_Regression\raw\results \
  --figures_dir C:\Projects\U\Pro_fin3\Models\Logistic_Regression\raw\figures
```

**Key arguments** (defaults shown):

* `--data` (required): CSV or Parquet path. `--target` must exist as a column. &#x20;
* `--results_dir`, `--figures_dir`: output locations (defaults are Windows paths).&#x20;
* `--test_size`, `--random_state`: split control; note `random_state` is used only for the split and sampling. &#x20;
* `--sample_frac`: quickly run on a fraction of the data.&#x20;
* `--metrics_sample_max`: cap the number of points used for curves to keep memory sane.&#x20;
* `--optimize_threshold`: also report metrics at the **ROC-optimal (Youden-J)** threshold.&#x20;

---

## Outputs

**Results (`results_dir`)**

* `summary.json`: run metadata, params, sizes, runtime, metrics at 0.5 and (optionally) at the optimal threshold.&#x20;
* `classification_report_thr0.5.txt`: precision/recall/F1 by class at threshold 0.5.&#x20;
* `roc_points.csv`, `pr_points.csv`: raw points for ROC and PR curves.&#x20;
* `confusion_matrix_thr0.5.csv`, `confusion_matrix_thrOPT.csv` (if enabled). &#x20;
* `feature_importances_full.csv`: every coefficient (feature, weight).&#x20;

**Figures (`figures_dir`)**

* `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png`.&#x20;
* `confusion_matrix_thr0.5.png`, `confusion_matrix_thrOPT.png`. &#x20;
* `feature_importances_top20.png`.&#x20;

---

## Metrics (what they mean)

All “thresholded” metrics below are computed at **0.5** by default; if `--optimize_threshold` is used, the same set is also reported at the **Youden-J** threshold described later. &#x20;

* **Accuracy**: fraction of correct predictions. Good when classes are balanced.&#x20;
* **Balanced accuracy**: average of recall for each class; robust to class imbalance.&#x20;
* **Precision** (for the positive class): among predicted positives, how many are truly positive. Useful when false positives are costly.&#x20;
* **Recall** (TPR, sensitivity): among true positives, how many were found. Useful when missing positives is costly.&#x20;
* **F1**: harmonic mean of precision and recall; balances both.&#x20;
* **ROC-AUC**: probability the classifier ranks a random positive above a random negative; threshold-independent.&#x20;
* **Average Precision (AP)**: area under the Precision-Recall curve; better reflects performance on imbalanced data.&#x20;

---

## Plots (what to look for)

* **ROC curve (`roc_curve.png`)**
  TPR vs. FPR at all thresholds. The **closer the curve to the top-left** and the **larger the AUC**, the better. Generated with `RocCurveDisplay.from_predictions`.&#x20;

* **Precision–Recall curve (`pr_curve.png`)**
  Precision vs. recall for the positive class. Prefer **high precision at high recall**. Generated with `PrecisionRecallDisplay.from_predictions`.&#x20;

* **Calibration curve (`calibration_curve.png`)**
  Compares **predicted probabilities** to **observed positive fractions**. A **well-calibrated** model lies near the diagonal. Computed with `calibration_curve(..., n_bins=20, strategy="quantile")`.&#x20;

* **Probability histogram (`proba_hist.png`)**
  Distribution of `p(y=1)` predictions. **Bimodal** (mass near 0 and 1) indicates confident predictions; a **bell around 0.5** suggests uncertainty.&#x20;

* **Confusion matrices**
  `confusion_matrix_thr0.5.png` is computed at threshold 0.5; if `--optimize_threshold` is set, `confusion_matrix_thrOPT.png` is also written at the Youden-J threshold. Values are annotated in-cell for quick reading.  &#x20;

* **Top coefficients (`feature_importances_top20.png`)**
  A **barh** plot of the 20 features with largest |coefficient|. Positive weights push predictions toward class 1; negative toward class 0 (assuming standardized features; here we show **raw** coefficients since no scaler is used). Full list is saved to CSV. &#x20;

---

## Threshold optimization (optional)

If `--optimize_threshold` is provided, the script finds the **Youden-J** point `argmax(tpr − fpr)` on the ROC curve and recomputes thresholded metrics and a second confusion matrix at that threshold. The numeric threshold is reported in `summary.json` and printed to the console.  &#x20;

---

## Console output

At the end you’ll see (among other info): **number of LBFGS iterations**, **fit time**, **ROC-AUC**, **AP**, and (if enabled) `thr_opt` and `F1_opt`. This helps you sanity-check training and overall discrimination quality without opening files.&#x20;

---

## Reproducibility & performance tips

* `--random_state` affects **only** the split and the sampling used for plotting (not the solver’s internal randomness). &#x20;
* Use `--sample_frac < 1.0` to iterate quickly on large datasets. You’ll still get representative metrics/curves.&#x20;
* Tune `--metrics_sample_max` if you hit memory pressure when building ROC/PR curves.&#x20;

---

## Limitations of this baseline

* **No scaling** and **no hyperparameter tuning**: it’s intentionally bare-bones (`LogisticRegression()` defaults). Use your improved pipelines for serious modeling.&#x20;
* **Class imbalance**: the default threshold 0.5 may not be ideal; prefer AUC/AP and the PR curve for model selection, and optionally use the Youden-J threshold for a balanced operating point. &#x20;

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

&#x20;  &#x20;

---

## Example: minimal Windows run

```powershell
# Parquet
python raw_LR.py --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet --target target

# CSV
python raw_LR.py --data C:\Projects\U\Pro_fin3\data\HIGGS.csv --target target
```

Default output folders (you can override with flags):
`C:\Projects\U\Pro_fin3\Models\Logistic_Regression\raw\results` and
`C:\Projects\U\Pro_fin3\Models\Logistic_Regression\raw\figures`.&#x20;

---