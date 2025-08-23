# How the best hyperparameters are found

## Common ideas across all improved scripts

* **Train/test split + CV subset.**
  Each script splits data into train/test, then takes a stratified subsample of the training set for faster CV (`--cv_frac`). Cross-validation uses `StratifiedKFold` with a configurable number of folds (`--cv_folds` / `--cv`). This keeps tuning fast and class-balanced.

* **Primary CV metric: ROC-AUC.**
  All tuning uses `scoring="roc_auc"` so model selection is threshold-independent and robust to class imbalance.

* **Resource-aware search.**
  When halving search is used, the “resource” is the number of samples (`resource="n_samples"`), capped by `min_resources`/`max_resources` to keep Stage-1 cheap.

* **Artifacts.**
  CV results are written to `cv_results_*.csv` and summaries to JSON so you can audit what was tried and why the winner won.

---

## `improved_LR.py` — Two-stage, compute-capped search for Logistic Regression

**Why two stages?**
Stage-1 is a broad but cheap screen with Halving Random Search; Stage-2 zooms in around the Stage-1 winner with either a focused RandomizedSearch or GridSearch.

### Stage-1 (broad, cheap) — HalvingRandomSearchCV

* Pipeline: `StandardScaler` → `LogisticRegression(solver='saga')` (SAGA enables L1/L2/Elastic-Net).
* Search space:

  * `penalty ∈ {l2, l1, elasticnet}`
  * `C ∈ {0.01…30}`
  * `l1_ratio ∈ {0, .2, .5, .8, 1}`
  * `class_weight ∈ {None, balanced}`
  * `fit_intercept ∈ {True, False}`
* Resource-aware: `resource='n_samples'`, bounded by `stage1_min_resources`/`stage1_max_resources`, with `n_candidates` and `factor` controlling exploration vs. elimination.
* Output: `cv_results_stage1.csv` and `search_summary_stage1.json` (iterations, candidates/resources per iter, best score/params).

### Stage-2 (focused) — RandomizedSearchCV or GridSearchCV around the winner

* Centering logic: take Stage-1’s best hyperparams, then build a compact neighborhood.
  For **C**, create a multiplicative neighborhood `{c/3, c/2, c, 2c, 3c}`.
  Keep `penalty`, `class_weight`, `fit_intercept` near the winner.
  Include `l1_ratio` **only** if `penalty=elasticnet`.
* Controlled with: `--stage2_kind {random|grid}` and `--stage2_budget` (`n_iter` for random).
* Output: `cv_results_stage2.csv`, `search_summary_stage2.json`, and reusable `best_params.json`.

### Control flow & refit/evaluate

* Modes:

  * `auto`: search unless `--params_json` is provided
  * `search_only`: run Stage-1+2 and exit
  * `refit_only`: skip search, load params JSON
* Final model is refit on a larger fraction of the train set (`--final_frac`) before test evaluation. A `summary.json` logs splits, caps, search kind/budget, best params, metrics (thr0.5 & thrOPT), and timings.

---

## `improved_DT.py` — Two-stage, compute-capped search for Decision Trees

### Stage-1 (broad) — HalvingRandomSearchCV

* Estimator: `DecisionTreeClassifier`.
* Search space:

  * `criterion ∈ {gini, entropy}`
  * `max_depth ∈ {6,8,10,12,14,16}`
  * `min_samples_leaf ∈ {1…100}`
  * `min_samples_split ∈ {2…50}`
  * `max_features ∈ {None, sqrt, log2, 0.5}`
  * `class_weight ∈ {None, balanced}`
* Resource-aware halving with the same caps and scoring as LR.
* Output: `cv_results_stage1.csv` and `search_summary_stage1.json`.

### Stage-2 (focused) — around the winner

* Neighborhood grid:
  Fix `criterion` & `class_weight` to Stage-1 winner.
  Search `max_depth` in ±`md_window` (bounded, e.g., \[4,32]).
  Double/half around `min_samples_leaf` and `min_samples_split`.
  Keep `max_features` near the winner.
* Controlled with: `--stage2_kind {random,grid}` and `--stage2_budget`.
* Output: best estimator, params, `cv_results_stage2.csv`, and `search_summary_stage2.json`.

---

## `improved_GBC.py` — Single-stage tuned Gradient Boosting (GB or HistGB)

**Algorithm choice:**

* `--algo gb` → `GradientBoostingClassifier`
* `--algo hgb` → `HistGradientBoostingClassifier`

**Search spaces:**

* **GB:** `n_estimators` (80–400), `learning_rate` (≈0.02–0.3), `max_depth` (2–6), `subsample` (0.6–1.0), `min_samples_leaf` (1–64), `max_features` (None/√/log2).
* **HGB:** `learning_rate` (≈0.02–0.3), `max_depth` (None or 3–12), `max_leaf_nodes` (None or powers of 2), `min_samples_leaf` (5–100), `l2_regularization`, `max_bins`.

**Searcher:**

* `--search halving` → `HalvingRandomSearchCV` (`resource="n_samples"`)
* otherwise `RandomizedSearchCV(n_iter=--n_iter)`.
  Both use ROC-AUC scoring with stratified folds.

**BEST vs FINAL\_ALL\_DATA:**

* **BEST**: fits/evaluates on the same comparable subset as baselines.
* **FINAL\_ALL\_DATA**: retrains with all train data (or `--final_train_frac`) and evaluates again.
  Artifacts include `cv_results.csv`, `search_args.json`, `best_params.json`, and stage-specific results.

---

# Metrics (definitions & interpretation)

All scripts compute the same metrics on the **test set** (optionally subsampled by `--metrics_sample_max` to save memory).

### At default threshold 0.5 (label = 1 if p ≥ 0.5)

* **Accuracy**: proportion correct ((TP+TN)/total). Can be misleading with imbalance.
* **Balanced accuracy**: average recall per class; safer with imbalance.
* **Precision**: of predicted positives, fraction truly positive. High precision = few false positives.
* **Recall (Sensitivity/TPR)**: of actual positives, fraction found. High recall = few false negatives.
* **F1**: harmonic mean of precision & recall, balances both.
* **ROC-AUC**: threshold-free discrimination; probability a random positive ranks above a random negative.
* **Average Precision (AP / PR-AUC)**: area under precision–recall; emphasizes performance on the positive class.

### At optimized threshold (optional) — Youden’s J

* **Youden-J index:** J = TPR − FPR. The script picks the threshold maximizing J.
* At this threshold it recomputes Accuracy, Balanced Accuracy, Precision, Recall, and F1.
* Produces a second confusion matrix (`confusion_matrix_thrOPT`).
* Use this when you want a **balanced operating point**; compare F1/Recall changes between thr=0.5 and thrOPT.

---

# Diagnostic plots

* **Confusion matrices (thr=0.5 & thrOPT):** show TP/FP/TN/FN counts. Good for seeing error types directly.
* **ROC curve:** TPR vs FPR across thresholds. Closer to top-left = better.
* **Precision–Recall curve:** precision vs recall. Useful for rare positives.
* **Calibration curve:** predicted prob vs observed fraction. Closer to diagonal = better calibrated.
* **Probability histogram:** distribution of predicted probabilities.

  * Bimodal (mass near 0 & 1) → confident predictions
  * Mass near 0.5 → uncertainty
* **Classification report (thr=0.5):** textual summary of per-class precision/recall/F1 and support.
* **Feature importance:**

  * **LR:** coefficients (sign = direction, magnitude = strength).
  * **DT/GB:** impurity- or permutation-based feature importances.

---

# Why this setup works (intuition)

* **Stage-1 Halving search:** explores broadly with few resources, prunes weak candidates, and escalates only promising ones. Cheap yet effective.
* **Stage-2 focused search:** zooms into the neighborhood of the Stage-1 winner to refine. Efficiently finds the optimum.
* **ROC-AUC for model selection:** ensures we pick models with globally good ranking ability.
* **Threshold optimization optional:** lets you adapt to the cost of errors (e.g., minimize false negatives vs false positives).

---