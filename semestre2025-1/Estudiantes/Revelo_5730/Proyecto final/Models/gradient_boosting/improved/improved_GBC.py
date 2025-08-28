import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.utils import check_random_state

try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV

    HAVE_HALVING = True
except Exception:
    HAVE_HALVING = False


# Helpers


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_target_and_features(df: pd.DataFrame, target_name: str, feature_set: str):
    if target_name not in df.columns:
        target_name = df.columns[0]
    y = df[target_name].astype(int).to_numpy()
    cols = list(df.columns)
    if feature_set == "low":
        X_cols = cols[1:22]
    elif feature_set == "high":
        X_cols = cols[22:29]
    else:
        X_cols = cols[1:]
    X = df[X_cols].copy()
    return X, y, X_cols


def stratified_subsample_df(X: pd.DataFrame, y: np.ndarray, frac: float, seed: int):
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1]")
    if frac == 1.0:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=seed
    )
    return Xs, ys


def bounded_sample_idx(n, max_n, rng):
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


def plot_confusion_matrix(cm, labels, out_path: Path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importances(names, values, out_png: Path, out_csv: Path, top_k=20):
    order = np.argsort(values)[::-1]
    names = np.array(names)[order]
    values = np.array(values)[order]
    top = min(top_k, len(values))

    fig, ax = plt.subplots(figsize=(6, max(3.2, 0.3 * top)))
    ax.barh(range(top), values[:top][::-1])
    ax.set_yticks(range(top))
    ax.set_yticklabels(names[:top][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importances (GB/HGB)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    pd.DataFrame({"feature": names, "importance": values}).to_csv(out_csv, index=False)


def optimize_threshold(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    return float(thr[j_idx])


# Search spaces


def search_space(algo: str):
    """Return param distributions for RandomizedSearchCV/HalvingRandomSearchCV."""
    rng = np.random.default_rng(123)
    if algo == "gb":
        return {
            "n_estimators": list(np.linspace(80, 400, 11, dtype=int)),
            "learning_rate": list(
                np.round(np.exp(rng.uniform(np.log(0.02), np.log(0.3), 30)), 4)
            ),
            "max_depth": list(range(2, 7)),
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 4, 8, 16, 32, 64],
            "max_features": [None, "sqrt", "log2"],
        }
    return {
        "learning_rate": list(
            np.round(np.exp(rng.uniform(np.log(0.02), np.log(0.3), 30)), 4)
        ),
        "max_depth": [None] + list(range(3, 13)),
        "max_leaf_nodes": [None, 31, 63, 127, 255, 511],
        "min_samples_leaf": [5, 10, 20, 50, 100],
        "l2_regularization": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        "max_bins": [128, 255, 512],
    }


# Train/eval utilities


def fit_and_eval(
    model,
    Xte,
    yte,
    out_results: Path,
    out_figures: Path,
    feature_names,
    *,
    metrics_sample_max: int,
    optimize_thr: bool,
    save_points: bool,
    rs,
):
    # Probabilities
    try:
        y_proba = model.predict_proba(Xte)[:, 1]
    except Exception:
        raw = model.decision_function(Xte)
        y_proba = 1.0 / (1.0 + np.exp(-raw))
    y_pred = (y_proba >= 0.5).astype(int)

    # Downsample for curves/CM if needed
    idx = bounded_sample_idx(len(yte), metrics_sample_max, rs)
    ys = yte[idx]
    ps = y_proba[idx]

    # Metrics at 0.5
    metrics_05 = {
        "accuracy": accuracy_score(ys, (ps >= 0.5).astype(int)),
        "balanced_accuracy": balanced_accuracy_score(ys, (ps >= 0.5).astype(int)),
        "precision": precision_score(ys, (ps >= 0.5).astype(int), zero_division=0),
        "recall": recall_score(ys, (ps >= 0.5).astype(int), zero_division=0),
        "f1": f1_score(ys, (ps >= 0.5).astype(int), zero_division=0),
        "roc_auc": roc_auc_score(ys, ps),
        "average_precision": average_precision_score(ys, ps),
    }

    # Confusion matrices
    cm_05 = confusion_matrix(ys, (ps >= 0.5).astype(int))
    plot_confusion_matrix(
        cm_05,
        ["0", "1"],
        out_figures / "confusion_matrix_thr0.5.png",
        title="Confusion matrix (thr=0.5)",
    )
    pd.DataFrame(
        cm_05, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    ).to_csv(out_results / "confusion_matrix_thr0.5.csv", index=True)

    # Optimal threshold
    thr_opt = None
    metrics_opt = None
    if optimize_thr:
        thr_opt = optimize_threshold(ys, ps)
        y_opt = (ps >= thr_opt).astype(int)
        metrics_opt = {
            "accuracy": accuracy_score(ys, y_opt),
            "balanced_accuracy": balanced_accuracy_score(ys, y_opt),
            "precision": precision_score(ys, y_opt, zero_division=0),
            "recall": recall_score(ys, y_opt, zero_division=0),
            "f1": f1_score(ys, y_opt, zero_division=0),
            "roc_auc": roc_auc_score(ys, ps),
            "average_precision": average_precision_score(ys, ps),
        }
        cm_opt = confusion_matrix(ys, y_opt)
        plot_confusion_matrix(
            cm_opt,
            ["0", "1"],
            out_figures / "confusion_matrix_thrOPT.png",
            title=f"Confusion matrix (thr={thr_opt:.3f})",
        )
        pd.DataFrame(
            cm_opt, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
        ).to_csv(out_results / "confusion_matrix_thrOPT.csv", index=True)

    # Text report at 0.5
    report = classification_report(
        ys, (ps >= 0.5).astype(int), digits=4, zero_division=0
    )
    with open(
        out_results / "classification_report_thr0.5.txt", "w", encoding="utf-8"
    ) as f:
        f.write(report)

    # Curves
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(ys, ps, ax=ax)
    ax.set_title("ROC curve (GBC)")
    fig.tight_layout()
    fig.savefig(out_figures / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(ys, ps, ax=ax)
    ax.set_title("Precision–Recall curve (GBC)")
    fig.tight_layout()
    fig.savefig(out_figures / "pr_curve.png", dpi=160)
    plt.close(fig)

    # Calibration and histogram
    prob_true, prob_pred = calibration_curve(ys, ps, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    fig.tight_layout()
    fig.savefig(out_figures / "calibration_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(ps, bins=40)
    ax.set_title("Predicted probability histogram")
    ax.set_xlabel("p(y=1)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_figures / "proba_hist.png", dpi=160)
    plt.close(fig)

    # Feature importance
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise AttributeError
    except Exception:
        # permutation importance on a capped subset
        pi_n = min(len(ys), 50_000)
        rs_local = check_random_state(13)
        pi_idx = bounded_sample_idx(len(ys), pi_n, rs_local)
        pi = permutation_importance(
            model,
            Xte.iloc[pi_idx] if hasattr(Xte, "iloc") else Xte[pi_idx],
            ys[pi_idx],
            n_repeats=5,
            random_state=13,
        )
        importances = pi.importances_mean

    plot_feature_importances(
        feature_names,
        importances,
        out_png=out_figures / "feature_importances_top20.png",
        out_csv=out_results / "feature_importances_full.csv",
        top_k=20,
    )

    # Save curve points
    if save_points:
        fpr, tpr, _ = roc_curve(ys, ps)
        prec, rec, _ = precision_recall_curve(ys, ps)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
            out_results / "roc_points.csv", index=False
        )
        pd.DataFrame({"precision": prec, "recall": rec}).to_csv(
            out_results / "pr_points.csv", index=False
        )

    metrics = {
        "metrics_thr0.5": metrics_05,
        "metrics_thr_opt": metrics_opt,
        "threshold_opt": thr_opt,
    }
    return metrics


# Main


def main():
    ap = argparse.ArgumentParser(
        description="Improved Gradient Boosting with search and final runs"
    )
    ap.add_argument("--data", required=True, help="Path to .parquet (all rows)")
    ap.add_argument("--target", default="target")
    ap.add_argument("--feature_set", choices=["low", "high", "all"], default="all")

    ap.add_argument(
        "--results_dir",
        default=r"C:\\Projects\\U\\Pro_fin3\\Models\\gradient_boosting\\improved\\results",
    )
    ap.add_argument(
        "--figures_dir",
        default=r"C:\\Projects\\U\\Pro_fin3\\Models\\gradient_boosting\\improved\\figures",
    )

    ap.add_argument(
        "--sample_frac",
        type=float,
        default=0.25,
        help="Subsample fraction for BEST comparison run",
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--metrics_sample_max", type=int, default=2_000_000)

    # Search config
    ap.add_argument("--algo", choices=["gb", "hgb"], default="hgb")
    ap.add_argument("--search", choices=["random", "halving"], default="random")
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument(
        "--cv_frac",
        type=float,
        default=0.2,
        help="Fraction of TRAIN used for CV search",
    )
    ap.add_argument("--cv_jobs", type=int, default=2)

    # Final all-data training fraction
    ap.add_argument("--final_train_frac", type=float, default=1.0)

    ap.add_argument("--optimize_threshold", action="store_true")
    ap.add_argument("--save_points", action="store_true")
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    rs = check_random_state(args.random_state)

    # Prepare dirs
    base_results = Path(args.results_dir)
    base_figures = Path(args.figures_dir)
    for sub in ["search", "best", "final_all_data"]:
        ensure_dir(base_results / sub)
    for sub in ["best", "final_all_data"]:
        ensure_dir(base_figures / sub)

    # Load full data
    df = pd.read_parquet(args.data)
    X_full, y_full, feat_names = pick_target_and_features(
        df, args.target, args.feature_set
    )

    X_sample, y_sample = stratified_subsample_df(
        X_full, y_full, args.sample_frac, args.random_state
    )
    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(
        X_sample,
        y_sample,
        test_size=args.test_size,
        stratify=y_sample,
        random_state=args.random_state,
    )

    Xcv, ycv = stratified_subsample_df(Xtr_b, ytr_b, args.cv_frac, args.random_state)

    # Build base estimator
    if args.algo == "gb":
        base_est = GradientBoostingClassifier(random_state=args.random_state)
    else:
        base_est = HistGradientBoostingClassifier(random_state=args.random_state)

    space = search_space(args.algo)

    if args.search == "halving" and HAVE_HALVING:
        searcher = HalvingRandomSearchCV(
            base_est,
            param_distributions=space,
            factor=3,
            resource="n_samples",
            max_resources=len(Xcv),
            scoring="roc_auc",
            n_jobs=args.cv_jobs,
            cv=StratifiedKFold(
                n_splits=args.cv, shuffle=True, random_state=args.random_state
            ),
            random_state=args.random_state,
            verbose=1,
        )
    else:
        searcher = RandomizedSearchCV(
            base_est,
            param_distributions=space,
            n_iter=args.n_iter,
            scoring="roc_auc",
            n_jobs=args.cv_jobs,
            cv=StratifiedKFold(
                n_splits=args.cv, shuffle=True, random_state=args.random_state
            ),
            random_state=args.random_state,
            verbose=1,
            refit=True,
        )

    t0 = time.time()
    searcher.fit(Xcv, ycv)
    search_time = time.time() - t0

    # Save search results
    cv_df = pd.DataFrame(searcher.cv_results_)
    cv_df.to_csv(base_results / "search" / "cv_results.csv", index=False)
    save_json(
        {
            "algo": args.algo,
            "search": args.search,
            "n_iter": args.n_iter,
            "cv": args.cv,
            "cv_frac": args.cv_frac,
            "cv_jobs": args.cv_jobs,
            "search_time_sec": round(search_time, 2),
        },
        base_results / "search" / "search_args.json",
    )

    best_params = searcher.best_params_
    best_score = float(searcher.best_score_)
    save_json(
        {"best_params": best_params, "best_cv_roc_auc": best_score},
        base_results / "search" / "best_params.json",
    )

    # Build best estimator
    if args.algo == "gb":
        best_est = GradientBoostingClassifier(
            random_state=args.random_state, **best_params
        )
    else:
        best_est = HistGradientBoostingClassifier(
            random_state=args.random_state, **best_params
        )

    # Best fit
    t0 = time.time()
    best_est.fit(Xtr_b, ytr_b)
    fit_time_b = time.time() - t0

    # Eval BEST
    metrics_b = fit_and_eval(
        best_est,
        Xte_b,
        yte_b,
        out_results=base_results / "best",
        out_figures=base_figures / "best",
        feature_names=feat_names,
        metrics_sample_max=args.metrics_sample_max,
        optimize_thr=args.optimize_threshold,
        save_points=args.save_points,
        rs=rs,
    )

    # Save BEST model & summary
    import joblib

    joblib.dump(best_est, base_results / "best" / "model.joblib")
    summary_b = {
        "stage": "BEST",
        "algo": args.algo,
        "feature_set": args.feature_set,
        "sample_frac": args.sample_frac,
        "test_size": args.test_size,
        "fit_time_seconds": round(fit_time_b, 3),
        "search": {
            "kind": args.search,
            "best_cv_roc_auc": best_score,
            "best_params": best_params,
        },
        **metrics_b,
    }
    save_json(summary_b, base_results / "best" / "summary.json")

    # Final all-data fit
    Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(
        X_full,
        y_full,
        test_size=args.test_size,
        stratify=y_full,
        random_state=args.random_state,
    )

    Xtr_f_use, ytr_f_use = stratified_subsample_df(
        Xtr_f, ytr_f, args.final_train_frac, args.random_state
    )

    # Fit final
    t0 = time.time()
    final_est = best_est.__class__(**best_est.get_params())
    final_est.fit(Xtr_f_use, ytr_f_use)
    fit_time_f = time.time() - t0

    # Eval final
    metrics_f = fit_and_eval(
        final_est,
        Xte_f,
        yte_f,
        out_results=base_results / "final_all_data",
        out_figures=base_figures / "final_all_data",
        feature_names=feat_names,
        metrics_sample_max=args.metrics_sample_max,
        optimize_thr=args.optimize_threshold,
        save_points=args.save_points,
        rs=rs,
    )

    # Save final model & summary
    joblib.dump(final_est, base_results / "final_all_data" / "model.joblib")
    summary_f = {
        "stage": "FINAL_ALL_DATA",
        "algo": args.algo,
        "feature_set": args.feature_set,
        "test_size": args.test_size,
        "final_train_frac": args.final_train_frac,
        "fit_time_seconds": round(fit_time_f, 3),
        "best_params": best_params,
        **metrics_f,
    }
    save_json(summary_f, base_results / "final_all_data" / "summary.json")

    # Console recap
    print("\n[Search] best ROC-AUC (cv):", round(best_score, 6))
    print("[Search] best params:", best_params)
    print(
        "[BEST]   fit_time=",
        round(fit_time_b, 2),
        "sec ; results ->",
        base_results / "best",
    )
    print(
        "[FINAL]  fit_time=",
        round(fit_time_f, 2),
        "sec ; results ->",
        base_results / "final_all_data",
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
