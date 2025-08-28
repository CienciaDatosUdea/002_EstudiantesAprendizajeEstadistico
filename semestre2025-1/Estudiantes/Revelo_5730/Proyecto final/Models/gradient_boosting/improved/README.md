# README — `improved_GBC.py` (Search-Tuned Gradient Boosting with BEST & FINAL runs)

A stronger, configurable Gradient Boosting pipeline that performs a **single-stage hyperparameter search** (either `RandomizedSearchCV` or `HalvingRandomSearchCV`) and then evaluates the chosen model in **two modes**:

1. **BEST** — trained/evaluated on a subsampled split so it’s directly comparable to your raw baselines (`--sample_frac`, `--test_size`).
2. **FINAL\_ALL\_DATA** — retrained on (up to) all available training data with the same best params, then evaluated on a new test split.&#x20;

---

## What this script does

1. **Loads data** (Parquet) and selects the **feature set**:

   * `low` = columns 1–21, `high` = 22–28, `all` = 1\:end. If the target name isn’t found, it assumes the first column is the target. Returns `(X, y, feature_names)`.&#x20;

2. **Creates output layout** (pre-makes folders):

   * `results/search/`, `results/best/`, `results/final_all_data/` and matching `figures/*` subfolders.&#x20;

3. **BEST split** for apples-to-apples against baselines:

   * Optional stratified subsample via `--sample_frac`, then stratified train/test split with `--test_size`.&#x20;

4. **Search on a subset of BEST-train** (`--cv_frac`):

   * Choose **algo**: `gb` (GradientBoostingClassifier) or **fast** `hgb` (HistGradientBoostingClassifier).
   * Choose **search**: `random` (default) or `halving` (if available).
   * CV uses `StratifiedKFold(n_splits=--cv, shuffle=True, random_state=...)`.&#x20;

5. **Search space** (automatically selected by `--algo`):

   * **GB**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_samples_leaf`, `max_features`.
   * **HGB**: `learning_rate`, `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_bins`.

6. **Saves search artifacts**:

   * `cv_results.csv`, `search_args.json` (with settings + search time), `best_params.json` (best params + best CV ROC-AUC).

7. **BEST training & evaluation**:

   * Fit the best estimator on BEST-train; evaluate on BEST-test; save artifacts (metrics, report, confusion matrices, curves, importances, model).

8. **FINAL\_ALL\_DATA training & evaluation**:

   * New split on the **full dataset**, optionally limit train portion via `--final_train_frac` (default 1.0), clone best params, retrain, and evaluate; save same artifacts.

9. **Console recap** prints best CV ROC-AUC, best params, and BEST/FINAL fit times with their output folders.

---

## Installation & requirements

* Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`.
* Uses a headless backend so plots save correctly on servers/CI (`matplotlib.use("Agg")`).

---

## Usage

**Example** (from the header docstring):

```bash
uv run python C:\Projects\U\Pro_fin3\Models\gradient_boosting\improved\improved_GBC.py ^
  --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet ^
  --target target --feature_set all ^
  --results_dir C:\Projects\U\Pro_fin3\Models\gradient_boosting\improved\results ^
  --figures_dir C:\Projects\U\Pro_fin3\Models\gradient_boosting\improved\figures ^
  --algo hgb --search random --n_iter 60 --cv 3 --cv_frac 0.2 --cv_jobs 2 ^
  --sample_frac 0.25 --test_size 0.2 --metrics_sample_max 2000000 ^
  --optimize_threshold --save_points --random_state 42
```

**Key arguments** (defaults shown):
Here’s a concise, complete guide to **every CLI argument** in `improved_GBC.py`, grouped by purpose. 

---

## I/O & data selection

* `--data` **(required)**: Path to a Parquet file; the script loads **all rows** from it.&#x20;

* `--target` (default: `target`): Name of the target column. If it’s not present, the script falls back to the **first column** in the file. &#x20;

* `--feature_set` (`low|high|all`, default: `all`): Chooses which feature slice to use.

  * `low` → columns 1:22, `high` → 22:29, `all` → 1\:end. Returns `(X_df, y_np, feature_names)`.&#x20;

* `--results_dir`, `--figures_dir` (both have Windows-style defaults): Base folders where the script pre-creates the sub-structure:
  `results/{search,best,final_all_data}` and `figures/{best,final_all_data}`. &#x20;

---

## Holdout split & curve sampling (BEST run comparability)

* `--sample_frac` (float, default: `0.25`): **Stratified subsample** of the full dataset before the BEST split—useful to match the sizing of your raw baselines. Must be `0<frac≤1`.  &#x20;
* `--test_size` (float, default: `0.2`): Fraction for the test split. Used **both** in the BEST and FINAL splits.  &#x20;
* `--metrics_sample_max` (int, default: `2_000_000`): Caps how many **test points** are used to compute curves and confusion matrices to keep memory in check. &#x20;

---

## Hyperparameter search (on a CV subset of the BEST train)

* `--algo` (`gb|hgb`, default: `hgb`): Chooses estimator family:

  * `gb` → `GradientBoostingClassifier`
  * `hgb` → `HistGradientBoostingClassifier` (faster)
    This choice controls the param space and which base estimator is built.   &#x20;

* `--search` (`random|halving`, default: `random`):

  * `random` → `RandomizedSearchCV(n_iter=...)`
  * `halving` → `HalvingRandomSearchCV` (if available), resource = `n_samples`, factor=3.
    Both use ROC-AUC scoring with `StratifiedKFold`.  &#x20;

* `--n_iter` (int, default: `60`): Number of sampled configurations for **RandomizedSearchCV**. (Ignored by Halving.) &#x20;

* `--cv` (int, default: `3`): Number of **StratifiedKFold** splits used by the searchers.  &#x20;

* `--cv_frac` (float, default: `0.2`): Fraction of the **BEST-train** used to run the search (stratified subsample). Keeps search fast while preserving class balance. &#x20;

* `--cv_jobs` (int, default: `2`): Parallel jobs for the searchers (`n_jobs`). Increase to use more cores.  &#x20;

The script logs **search artifacts** (`cv_results.csv`, `search_args.json`, `best_params.json`) for traceability.&#x20;

---

## FINAL training controls

* `--final_train_frac` (float, default: `1.0`): After choosing the best params, the script does a **fresh split on the full data** and optionally trains on a **fraction of that TRAIN** (useful if you want a capped “final” training size). &#x20;

---

## Evaluation & outputs

* `--optimize_threshold` (flag): Also compute metrics and a **second confusion matrix** at the **Youden-J** ROC-optimal threshold; the numeric threshold is saved in results.  &#x20;

* `--save_points` (flag): Save **raw** ROC and PR curve points to CSV (`roc_points.csv`, `pr_points.csv`) so you can re-plot externally. &#x20;

* `--random_state` (int, default: `42`): Seeds all stochastic parts—subsampling, train/test splits, CV splitter, and searchers—for reproducibility. &#x20;

---

# Where each arg “lands” in the flow

1. **Load & pick features** → `--data`, `--target`, `--feature_set`.&#x20;
2. **BEST split** (comparability) → `--sample_frac`, `--test_size`, `--random_state`.&#x20;
3. **Search on CV subset** → `--cv_frac`, `--algo`, `--search`, `--n_iter`, `--cv`, `--cv_jobs`, `--random_state`.&#x20;
4. **Refit BEST, evaluate** → `--metrics_sample_max`, `--optimize_threshold`, `--save_points`. &#x20;
5. **FINAL\_ALL\_DATA split & train** → `--test_size`, `--final_train_frac`, `--random_state`.&#x20;

---

## Outputs

**Search (`results/search/`)**

* `cv_results.csv` — raw CV rows from the searcher.
* `search_args.json` — search config + elapsed seconds.
* `best_params.json` — best hyperparameters + best CV ROC-AUC.

**BEST (`results/best/` + `figures/best/`)**

* `model.joblib` — fitted best estimator on BEST-train.
* `summary.json` — stage, algo, feature\_set, sample/test sizes, fit time, best CV score/params, metrics and (optional) optimized-threshold results.
* `classification_report_thr0.5.txt` — per-class precision/recall/F1 at thr=0.5.
* Confusion matrices at 0.5 (always) and at optimized threshold (if enabled), in PNG and CSV.
* Curves: `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`, `proba_hist.png`.
* Feature importance: `feature_importances_top20.png`, `feature_importances_full.csv`.

**FINAL\_ALL\_DATA (`results/final_all_data/` + `figures/final_all_data/`)**

* Mirrors the BEST set: model, summary, matrices, curves, importances.

---

## Metrics (what they mean)

All **thresholded** metrics default to **0.5** unless `--optimize_threshold` is used; **threshold-free** metrics use the predicted probability:

* **Accuracy** — share of correct predictions at the chosen threshold.
* **Balanced accuracy** — mean of class-wise recalls; better under imbalance.
* **Precision** — of predicted positives, fraction that are truly positive.
* **Recall (TPR)** — of actual positives, fraction found.
* **F1** — harmonic mean of precision & recall.
* **ROC-AUC** — rank quality: probability a random positive scores above a random negative (threshold-free).
* **Average Precision (AP)** — area under Precision–Recall curve (threshold-free).
  (Produced from probability `ps` and, for thresholded metrics, `(ps >= thr)`.)

**Optional optimized operating point**
If `--optimize_threshold`, the script computes the **Youden-J** threshold on the ROC curve (`argmax(tpr - fpr)`), recomputes thresholded metrics there, and saves a second confusion matrix. The chosen numeric threshold is included in the metrics bundle.

---

## Plots (how to read them)

* **ROC curve** — TPR vs FPR across all thresholds. The closer to the **top-left** and the higher the AUC, the better. Saved as `roc_curve.png`.

* **Precision–Recall curve** — precision vs recall for the positive class. Prefer curves that maintain high precision at high recall. Saved as `pr_curve.png`.

* **Calibration curve** — predicted probability vs observed fraction of positives (20 quantile bins). A calibrated model lies near the diagonal. Saved as `calibration_curve.png`.

* **Probability histogram** — distribution of `p(y=1)`. Bimodal shapes (mass near 0 and 1) indicate confident predictions; mass near 0.5 indicates uncertainty. Saved as `proba_hist.png`.

* **Confusion matrices** — `confusion_matrix_thr0.5.png` (always) and `confusion_matrix_thrOPT.png` (if optimized). Diagonal cells are correct; off-diagonals are errors. CSVs mirror the images.

* **Top feature importances** — If the model exposes `feature_importances_` (e.g., GB), it uses that; otherwise it falls back to **permutation importance** (used for HGB) on a capped subset. Plots top-20 and writes the full ordered list to CSV.

* **Curve points (optional)** — With `--save_points`, it stores `roc_points.csv` (`fpr,tpr`) and `pr_points.csv` (`precision,recall`) so you can re-plot externally.

---

## Search: algorithms & spaces

* **Algorithms:**
  `--algo gb` → `GradientBoostingClassifier`; `--algo hgb` → `HistGradientBoostingClassifier`.&#x20;

* **Searcher:**
  `--search random` → `RandomizedSearchCV`; `--search halving` (if available) → `HalvingRandomSearchCV` with resource on `n_samples`. Both use ROC-AUC scoring with stratified K-fold CV and respect `--cv_jobs`.

* **Parameter spaces:**
  Detailed grids/dists for GB and HGB (see above “What this script does · 5”).

---

## Performance & reproducibility tips

* **Compare fairly** with baselines\*\*:\*\* keep `--sample_frac` and `--test_size` identical across runs; the script is designed for that.&#x20;

* **Control search cost:** reduce `--n_iter` (random) or use `--search halving`; lower `--cv`, raise `--cv_jobs` as cores allow. Use a **smaller `--cv_frac`** for faster iterations, then rely on FINAL retraining.&#x20;

* **Large datasets:** set a sane `--metrics_sample_max` to keep ROC/PR memory in check (sampling is applied before computing curves and matrices).

* **Reproducibility:** `--random_state` seeds the samplers/splitters/search CV.&#x20;

---

## File map (at a glance)

```
results/
  search/
    cv_results.csv
    search_args.json
    best_params.json
  best/
    summary.json
    model.joblib
    classification_report_thr0.5.txt
    confusion_matrix_thr0.5.csv
    confusion_matrix_thrOPT.csv      # if --optimize_threshold
    roc_points.csv                   # if --save_points
    pr_points.csv                    # if --save_points
    feature_importances_full.csv
  final_all_data/
    summary.json
    model.joblib
    classification_report_thr0.5.txt
    confusion_matrix_thr0.5.csv
    confusion_matrix_thrOPT.csv      # if --optimize_threshold
    roc_points.csv                   # if --save_points
    pr_points.csv                    # if --save_points
    feature_importances_full.csv

figures/
  best/
    roc_curve.png
    pr_curve.png
    calibration_curve.png
    proba_hist.png
    confusion_matrix_thr0.5.png
    confusion_matrix_thrOPT.png      # if --optimize_threshold
    feature_importances_top20.png
  final_all_data/
    (same plot set as best/)
```

---

## Example: quick HGB run (Windows)

```powershell
uv run python C:\Projects\U\Pro_fin3\Models\gradient_boosting\improved\improved_GBC.py `
  --data C:\Projects\U\Pro_fin3\data\higgs_train.parquet `
  --target target --feature_set all `
  --algo hgb --search random --n_iter 40 --cv 3 --cv_frac 0.2 --cv_jobs 2 `
  --sample_frac 0.25 --test_size 0.2 --metrics_sample_max 2000000 `
  --optimize_threshold --save_points --random_state 42
```

(Adjust `--n_iter` and `--cv_jobs` to balance runtime vs. quality.)

---
