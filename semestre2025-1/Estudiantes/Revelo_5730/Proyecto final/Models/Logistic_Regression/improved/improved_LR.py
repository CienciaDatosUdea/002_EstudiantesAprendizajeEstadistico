import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from warnings import filterwarnings

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

filterwarnings("ignore", category=ConvergenceWarning)


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


def plot_top_coefficients(
    model_lr: LogisticRegression, feature_names, out_path: Path, top_k=20
):
    if not hasattr(model_lr, "coef_"):
        return []
    coefs = model_lr.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1]
    top = order[:top_k]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(top)), coefs[top][::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feature_names[i] for i in top][::-1], fontsize=9)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Top {top_k} coefficients by |value|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    # full ordered list
    return [(feature_names[i], float(coefs[i])) for i in order]


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

    # Curves
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(ys, ps, ax=ax)
    ax.set_title("ROC curve (Logistic Regression)")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(ys, ps, ax=ax)
    ax.set_title("Precision–Recall curve (Logistic Regression)")
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

    # Save curve points
    fpr, tpr, _ = roc_curve(ys, ps)
    prec, rec, _ = precision_recall_curve(ys, ps)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        results_dir / "roc_points.csv", index=False
    )
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(
        results_dir / "pr_points.csv", index=False
    )

    # Coefficients from inner LR
    lr = model.named_steps["lr"] if hasattr(model, "named_steps") else model
    coef_list = plot_top_coefficients(
        lr, feature_names, figures_dir / "feature_importances_top20.png", 20
    )
    if coef_list:
        pd.DataFrame(coef_list, columns=["feature", "coefficient"]).to_csv(
            results_dir / "feature_importances_full.csv", index=False
        )

    report = classification_report(
        ys, (ps >= 0.5).astype(int), digits=4, zero_division=0
    )
    (results_dir / "classification_report_thr0.5.txt").write_text(
        report, encoding="utf-8"
    )

    return (
        metrics,
        metrics_opt,
        thr_opt,
        int(getattr(lr, "n_iter_", [0])[0]) if hasattr(lr, "n_iter_") else None,
    )


# Search helpers
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


def stage1_search(
    X, y, cv, n_candidates, factor, n_jobs, random_state, min_resources, max_resources
):
    # Pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="saga", penalty="l2", max_iter=2000, tol=1e-3
                ),
            ),
        ]
    )
    # Discrete candidate lists
    param_dist = {
        "lr__penalty": ["l2", "l1", "elasticnet"],
        "lr__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "lr__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],  # only used if elasticnet
        "lr__class_weight": [None, "balanced"],
        "lr__fit_intercept": [True, False],
    }

    max_res = int(min(max_resources, len(X)))
    min_res = int(min(min_resources, max_res))

    hs = HalvingRandomSearchCV(
        pipe,
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


def stage2_search(kind, X, y, cv, best_params1, n_jobs, random_state, budget):
    # Focus around Stage-1 winner
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(solver="saga", max_iter=3000, tol=1e-3)),
        ]
    )

    bp = {
        k.replace("lr__", ""): v
        for k, v in best_params1.items()
        if k.startswith("lr__")
    }

    C_vals = [0.5, 1.0, 2.0, 3.0]
    if "C" in bp:
        c = float(bp["C"])
        C_vals = sorted(set([c / 3, c / 2, c, c * 2, c * 3]))
        C_vals = [max(1e-4, float(x)) for x in C_vals]

    penalty = bp.get("penalty", "l2")
    class_weight = bp.get("class_weight", None)
    fit_intercept = bp.get("fit_intercept", True)

    grid = {
        "lr__penalty": [penalty] if penalty in ["l1", "l2", "elasticnet"] else ["l2"],
        "lr__C": C_vals,
        "lr__class_weight": (
            [class_weight] if class_weight in [None, "balanced"] else [None, "balanced"]
        ),
        "lr__fit_intercept": (
            [fit_intercept] if isinstance(fit_intercept, bool) else [True, False]
        ),
    }
    if penalty == "elasticnet":
        grid["lr__l1_ratio"] = [bp.get("l1_ratio", 0.5), 0.2, 0.5, 0.8]

    if kind == "grid":
        gs = GridSearchCV(
            base,
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
        meta = {"kind": "grid", "grid_sizes": {k: len(v) for k, v in grid.items()}}
    else:
        rs = RandomizedSearchCV(
            base,
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
        meta = {
            "kind": "random",
            "n_iter": budget,
            "space_sizes": {k: len(v) for k, v in grid.items()},
        }

    return best_est, best_params, best_score, cv_df, meta


def main():
    ap = argparse.ArgumentParser(
        description="Improved Logistic Regression with compute-capped tuning"
    )
    ap.add_argument("--data", required=True, help="Path to .parquet or .csv")
    ap.add_argument("--target", default="target")

    # Fixed outputs
    ap.add_argument(
        "--results_dir",
        default=r"C:\Projects\U\Pro_fin3\Models\Logistic_Regression\improved\results",
    )
    ap.add_argument(
        "--figures_dir",
        default=r"C:\Projects\U\Pro_fin3\Models\Logistic_Regression\improved\figures",
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
    ap.add_argument("--stage2_budget", type=int, default=120)

    # Evaluation
    ap.add_argument("--metrics_sample_max", type=int, default=2_000_000)
    ap.add_argument("--optimize_threshold", action="store_true")

    # Control flow: reuse/search-only
    ap.add_argument(
        "--mode",
        choices=["auto", "search_only", "refit_only"],
        default="auto",
        help="auto: run search unless --params_json provided; search_only: run search and exit; refit_only: load params JSON and refit/evaluate.",
    )
    ap.add_argument(
        "--params_json",
        default="",
        help="Path to JSON with params to reuse. If empty in refit_only mode, defaults to <results_dir>/best_params.json",
    )

    args = ap.parse_args()
    rs = check_random_state(args.random_state)

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    t0 = time.time()

    # ---------- Load & split ----------
    X, y, feature_names = load_data(Path(args.data), target=args.target)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # CV subset for search
    Xcv, ycv = stratified_subsample(Xtr, ytr, args.cv_frac, args.random_state)
    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
    )

    # Decide whether to SEARCH or REUSE
    do_search = (args.mode == "search_only") or (
        args.mode == "auto" and not args.params_json
    )
    best_params2 = None
    best_model = None

    if do_search:
        # ---------- Stage 1 ----------
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

        # ---------- Stage 2 ----------
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
        )
        cv_df2.to_csv(results_dir / "cv_results_stage2.csv", index=False)
        meta2["best_score"] = best_score2
        meta2["best_params"] = best_params2
        (results_dir / "search_summary_stage2.json").write_text(
            json.dumps(meta2, indent=2), encoding="utf-8"
        )

        # Save params for reuse
        (results_dir / "best_params.json").write_text(
            json.dumps(best_params2, indent=2), encoding="utf-8"
        )

        if args.mode == "search_only":
            print(
                f"[Search only] Saved best params -> {(results_dir / 'best_params.json').resolve()}"
            )
            return

    else:
        # ---------- Reuse params ----------
        params_path = (
            Path(args.params_json)
            if args.params_json
            else (results_dir / "best_params.json")
        )
        if not params_path.exists():
            raise FileNotFoundError(f"No params JSON found at: {params_path}")
        best_params2 = json.loads(params_path.read_text(encoding="utf-8"))
        # Build model from params
        best_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(solver="saga", max_iter=3000, tol=1e-3)),
            ]
        )
        best_model.set_params(**best_params2)

    # Final refit on full TRAIN set
    Xfinal, yfinal = stratified_subsample(Xtr, ytr, args.final_frac, args.random_state)
    fit_t0 = time.time()
    best_model.set_params()  # no-op; ensures pipeline exists
    best_model.fit(Xfinal, yfinal)
    fit_t1 = time.time()

    # Evaluate on TEST
    metrics, metrics_opt, thr_opt, n_iter_lr = evaluate_and_save(
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

    lr = best_model.named_steps["lr"]
    coef_stats = {
        "n_features": int(len(feature_names)),
        "n_iter_": (
            int(getattr(lr, "n_iter_", [0])[0]) if hasattr(lr, "n_iter_") else None
        ),
    }

    # Summary
    summary = {
        "model": "LogisticRegression (pipeline: StandardScaler -> LR)",
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
        "stage2": {"kind": args.stage2_kind, "budget": args.stage2_budget},
        "best_params": best_params2,
        "coef_stats": coef_stats,
        "metrics_thr0.5": metrics,
        "metrics_thr_opt": metrics_opt,
        "threshold_opt": thr_opt,
        "fit_time_seconds": round(fit_t1 - fit_t0, 3),
        "total_time_seconds": round(time.time() - t0, 3),
        "mode": args.mode,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[Improved LR] n_iter={coef_stats['n_iter_']}")
    print(
        f"[Improved LR] Test ROC-AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}"
    )
    if thr_opt is not None:
        print(f"[Improved LR] thr_opt={thr_opt:.4f}, F1_opt={metrics_opt['f1']:.6f}")
    print(f"Results  -> {results_dir.resolve()}")
    print(f"Figures  -> {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
