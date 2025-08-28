import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
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
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state


# Helpers
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_data(path: Path, target: str, float_dtype="float32"):
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
        for c in df.columns:
            if c != target and df[c].dtype.kind in "fi":
                df[c] = df[c].astype(float_dtype)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in data.")
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target])
    return X, y, [c for c in df.columns if c != target]


def stratified_subsample(X, y, frac, seed):
    if frac >= 1.0:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=seed
    )
    return Xs, ys


def bounded_sample_idx(n, max_n, rng):
    return np.arange(n) if n <= max_n else rng.choice(n, size=max_n, replace=False)


# Plotting
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


def plot_feature_importances(model, feature_names, out_path: Path, top_k=20):
    if not hasattr(model, "feature_importances_"):
        return []
    imps = model.feature_importances_
    order = np.argsort(imps)[::-1]
    top = order[:top_k]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(top)), imps[top][::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feature_names[i] for i in top][::-1], fontsize=9)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_k} feature importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return [(feature_names[i], float(imps[i])) for i in order]


def optimize_threshold(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def evaluate_and_save(
    model,
    Xte,
    yte,
    feature_names,
    results_dir: Path,
    figures_dir: Path,
    random_state=42,
    metrics_sample_max=2_000_000,
    optimize_thr=True,
):
    rs = check_random_state(random_state)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    idx = bounded_sample_idx(len(yte), metrics_sample_max, rs)
    ys, ps, yp = yte[idx], y_proba[idx], y_pred[idx]

    metrics = {
        "accuracy": accuracy_score(ys, yp),
        "balanced_accuracy": balanced_accuracy_score(ys, yp),
        "precision": precision_score(ys, yp, zero_division=0),
        "recall": recall_score(ys, yp, zero_division=0),
        "f1": f1_score(ys, yp, zero_division=0),
        "roc_auc": roc_auc_score(ys, ps),
        "average_precision": average_precision_score(ys, ps),
    }
    cm_05 = confusion_matrix(ys, (ps >= 0.5).astype(int))
    plot_confusion_matrix(
        cm_05,
        ["0", "1"],
        figures_dir / "confusion_matrix_thr0.5.png",
        "Confusion matrix (thr=0.5)",
    )
    pd.DataFrame(
        cm_05, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    ).to_csv(results_dir / "confusion_matrix_thr0.5.csv", index=True)

    thr_opt, metrics_opt = None, None
    if optimize_thr:
        thr_opt = optimize_threshold(ys, ps)
        y_opt = (ps >= thr_opt).astype(int)
        metrics_opt = {
            "threshold": thr_opt,
            "accuracy": accuracy_score(ys, y_opt),
            "balanced_accuracy": balanced_accuracy_score(ys, y_opt),
            "precision": precision_score(ys, y_opt, zero_division=0),
            "recall": recall_score(ys, y_opt, zero_division=0),
            "f1": f1_score(ys, y_opt, zero_division=0),
        }
        cm_opt = confusion_matrix(ys, y_opt)
        plot_confusion_matrix(
            cm_opt,
            ["0", "1"],
            figures_dir / "confusion_matrix_thrOPT.png",
            f"Confusion matrix (thr={thr_opt:.3f})",
        )
        pd.DataFrame(
            cm_opt, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
        ).to_csv(results_dir / "confusion_matrix_thrOPT.csv", index=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(ys, ps, ax=ax)
    ax.set_title("ROC curve (Decision Tree)")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(ys, ps, ax=ax)
    ax.set_title("Precision–Recall curve (Decision Tree)")
    fig.tight_layout()
    fig.savefig(figures_dir / "pr_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    pt, pp = calibration_curve(ys, ps, n_bins=20, strategy="quantile")
    ax.plot(pp, pt, marker="o")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(ps, bins=40)
    ax.set_title("Predicted probability histogram")
    ax.set_xlabel("p(y=1)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(figures_dir / "proba_hist.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(ys, ps)
    prec, rec, _ = precision_recall_curve(ys, ps)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        results_dir / "roc_points.csv", index=False
    )
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(
        results_dir / "pr_points.csv", index=False
    )

    ordered_imps = plot_feature_importances(
        model, feature_names, figures_dir / "feature_importances_top20.png", 20
    )
    if ordered_imps:
        pd.DataFrame(ordered_imps, columns=["feature", "gini_importance"]).to_csv(
            results_dir / "feature_importances_full.csv", index=False
        )

    report = classification_report(
        ys, (ps >= 0.5).astype(int), digits=4, zero_division=0
    )
    (results_dir / "classification_report_thr0.5.txt").write_text(
        report, encoding="utf-8"
    )

    return metrics, metrics_opt, thr_opt


# Helpers
def _to_list_safe(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        out = []
        for v in val:
            if isinstance(v, (np.integer, int)):
                out.append(int(v))
            elif isinstance(v, (np.floating, float)):
                out.append(float(v))
            else:
                out.append(v)
        return out
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return val


def neighbors_int(center, low, high, window):
    lo = max(low, center - window)
    hi = min(high, center + window)
    return list(range(lo, hi + 1))


def stage1_search(
    X, y, cv, n_candidates, factor, n_jobs, random_state, min_resources, max_resources
):
    est = DecisionTreeClassifier(random_state=random_state)
    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": [6, 8, 10, 12, 14, 16],
        "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
        "min_samples_split": [2, 5, 10, 20, 50],
        "max_features": [None, "sqrt", "log2", 0.5],
        "class_weight": [None, "balanced"],
    }

    max_res = int(min(max_resources, len(X)))
    min_res = int(min(min_resources, max_res))

    hs = HalvingRandomSearchCV(
        est,
        param_dist,
        n_candidates=n_candidates,
        factor=factor,
        resource="n_samples",
        min_resources=min_res,
        max_resources=max_res,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        aggressive_elimination=True,
        refit=False,
    )
    hs.fit(X, y)
    return hs


def stage2_search(
    kind, X, y, cv, best_params1, n_jobs, random_state, budget, md_window
):
    # Build a compact neighborhood around Stage-1 winner
    md = int(best_params1.get("max_depth", 12))
    msl = int(best_params1.get("min_samples_leaf", 20))
    mss = int(best_params1.get("min_samples_split", 10))
    mf = best_params1.get("max_features", None)
    crit = best_params1.get("criterion", "gini")
    cw = best_params1.get("class_weight", "balanced")

    grid = {
        "criterion": [crit],
        "class_weight": [cw],
        "max_depth": neighbors_int(md, 4, 32, md_window),
        "min_samples_leaf": sorted({1, max(1, msl // 2), msl, min(400, msl * 2)}),
        "min_samples_split": sorted({2, max(2, mss // 2), mss, min(200, mss * 2)}),
        "max_features": (
            [mf] if mf in [None, "sqrt", "log2", 0.5, 0.75] else ["sqrt", None]
        ),
    }

    if kind == "grid":
        gs = GridSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=1,
        )
        gs.fit(X, y)
        best_est, best_params, best_score = (
            gs.best_estimator_,
            gs.best_params_,
            float(gs.best_score_),
        )
        cv_df = pd.DataFrame(gs.cv_results_)
        search_meta = {
            "kind": "grid",
            "grid_sizes": {k: len(v) for k, v in grid.items()},
        }
    else:
        # Randomized search
        rs = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            param_distributions=grid,
            n_iter=budget,
            scoring="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
            verbose=1,
        )
        rs.fit(X, y)
        best_est, best_params, best_score = (
            rs.best_estimator_,
            rs.best_params_,
            float(rs.best_score_),
        )
        cv_df = pd.DataFrame(rs.cv_results_)
        search_meta = {
            "kind": "random",
            "n_iter": budget,
            "space_sizes": {k: len(v) for k, v in grid.items()},
        }

    return best_est, best_params, best_score, cv_df, search_meta


def main():
    ap = argparse.ArgumentParser(
        description="Improved Decision Tree with compute-capped tuning"
    )
    ap.add_argument("--data", required=True, help="Path to .parquet or .csv")
    ap.add_argument("--target", default="target")

    # Fixed outputs
    ap.add_argument(
        "--results_dir",
        default=r"C:\Projects\U\Pro_fin3\Models\decision_tree\improved\results",
    )
    ap.add_argument(
        "--figures_dir",
        default=r"C:\Projects\U\Pro_fin3\Models\decision_tree\improved\figures",
    )

    # Splits
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument(
        "--cv_frac",
        type=float,
        default=0.10,
        help="Fraction of TRAIN used for CV search.",
    )
    ap.add_argument(
        "--final_frac",
        type=float,
        default=1.00,
        help="Fraction of TRAIN used to refit best model.",
    )

    # Stage 1 (Halving Random Search)
    ap.add_argument("--n_candidates1", type=int, default=24)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--cv_jobs", type=int, default=-1)
    ap.add_argument("--stage1_min_resources", type=int, default=10_000)
    ap.add_argument("--stage1_max_resources", type=int, default=250_000)

    # Stage 2 (focused search)
    ap.add_argument("--stage2_kind", choices=["random", "grid"], default="random")
    ap.add_argument(
        "--stage2_budget",
        type=int,
        default=120,
        help="n_iter for random, ignored for grid",
    )
    ap.add_argument(
        "--md_window", type=int, default=2, help="±window for max_depth in Stage 2"
    )

    # Evaluation
    ap.add_argument("--metrics_sample_max", type=int, default=2_000_000)
    ap.add_argument("--optimize_threshold", action="store_true")

    args = ap.parse_args()
    rs = check_random_state(args.random_state)

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    t0 = time.time()

    # Load data
    X, y, feature_names = load_data(Path(args.data), target=args.target)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # CV subset for search
    Xcv, ycv = stratified_subsample(Xtr, ytr, args.cv_frac, args.random_state)
    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
    )

    # Stage 1
    print("Stage 1: HalvingRandomSearchCV (budget-capped)")
    hs = stage1_search(
        Xcv,
        ycv,
        cv=cv,
        n_candidates=args.n_candidates1,
        factor=args.factor,
        n_jobs=args.cv_jobs,
        random_state=args.random_state,
        min_resources=args.stage1_min_resources,
        max_resources=args.stage1_max_resources,
    )
    pd.DataFrame(hs.cv_results_).to_csv(
        results_dir / "cv_results_stage1.csv", index=False
    )
    best_idx1 = int(np.argmax(hs.cv_results_["mean_test_score"]))
    best_params1 = hs.cv_results_["params"][best_idx1]

    stage1_summary = {
        "n_iterations": int(getattr(hs, "n_iterations_", 0)),
        "n_candidates_per_iter": _to_list_safe(getattr(hs, "n_candidates_", [])),
        "resources_per_iter": _to_list_safe(getattr(hs, "n_resources_", [])),
        "min_resources_": int(getattr(hs, "min_resources_", 0)),
        "max_resources_": int(getattr(hs, "max_resources_", 0)),
        "factor": getattr(hs, "factor", None),
        "best_score": float(hs.cv_results_["mean_test_score"][best_idx1]),
        "best_params": best_params1,
    }
    (results_dir / "search_summary_stage1.json").write_text(
        json.dumps(stage1_summary, indent=2), encoding="utf-8"
    )

    # Stage 2
    print(f"Stage 2: {args.stage2_kind.title()}SearchCV (focused)")
    best_model, best_params2, best_score2, cv_df2, meta2 = stage2_search(
        kind=args.stage2_kind,
        X=Xcv,
        y=ycv,
        cv=cv,
        best_params1=best_params1,
        n_jobs=args.cv_jobs,
        random_state=args.random_state,
        budget=args.stage2_budget,
        md_window=args.md_window,
    )
    cv_df2.to_csv(results_dir / "cv_results_stage2.csv", index=False)
    meta2["best_score"] = best_score2
    meta2["best_params"] = best_params2
    (results_dir / "search_summary_stage2.json").write_text(
        json.dumps(meta2, indent=2), encoding="utf-8"
    )

    # Final refit on TRAIN(final_frac)
    Xfinal, yfinal = stratified_subsample(Xtr, ytr, args.final_frac, args.random_state)
    fit_t0 = time.time()
    best_model.set_params(random_state=args.random_state)
    best_model.fit(Xfinal, yfinal)
    fit_t1 = time.time()

    # Evaluate on TEST
    metrics, metrics_opt, thr_opt = evaluate_and_save(
        best_model,
        Xte,
        yte,
        feature_names,
        results_dir,
        figures_dir,
        random_state=args.random_state,
        metrics_sample_max=args.metrics_sample_max,
        optimize_thr=args.optimize_threshold,
    )

    tree_stats = {
        "depth": int(best_model.get_depth()),
        "n_leaves": int(best_model.get_n_leaves()),
        "node_count": int(best_model.tree_.node_count),
    }

    # Persist overall summary
    summary = {
        "model": "DecisionTreeClassifier",
        "data": str(Path(args.data).resolve()),
        "target": args.target,
        "splits": {
            "test_size": args.test_size,
            "cv_frac": args.cv_frac,
            "final_frac": args.final_frac,
        },
        "cv": {
            "folds": args.cv_folds,
            "n_candidates_stage1": args.n_candidates1,
            "factor_stage1": args.factor,
        },
        "stage1_caps": {
            "min_resources": args.stage1_min_resources,
            "max_resources": args.stage1_max_resources,
        },
        "stage2": {
            "kind": args.stage2_kind,
            "budget": args.stage2_budget,
            "md_window": args.md_window,
        },
        "best_params": best_params2,
        "tree_stats": tree_stats,
        "metrics_thr0.5": metrics,
        "metrics_thr_opt": metrics_opt,
        "threshold_opt": thr_opt,
        "fit_time_seconds": round(fit_t1 - fit_t0, 3),
        "total_time_seconds": round(time.time() - t0, 3),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "best_params.json").write_text(
        json.dumps(best_params2, indent=2), encoding="utf-8"
    )

    print(
        f"[Improved DT Fast] depth={tree_stats['depth']}, leaves={tree_stats['n_leaves']}"
    )
    print(
        f"[Improved DT Fast] Test ROC-AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}"
    )
    if thr_opt is not None:
        print(
            f"[Improved DT Fast] thr_opt={thr_opt:.4f}, F1_opt={metrics_opt['f1']:.6f}"
        )
    print(f"Results  -> {results_dir.resolve()}")
    print(f"Figures  -> {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
