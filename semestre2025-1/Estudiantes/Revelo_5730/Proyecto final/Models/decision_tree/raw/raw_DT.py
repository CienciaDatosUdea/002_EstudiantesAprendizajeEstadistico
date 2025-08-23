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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state


def load_data(path: Path, target: str, float_dtype="float32"):
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
        for c in df.columns:
            if c != target and df[c].dtype.kind in "fi":
                df[c] = df[c].astype(float_dtype)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target])
    return X, y, [c for c in df.columns if c != target]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


def optimize_threshold(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def main():
    ap = argparse.ArgumentParser(description="Plain DecisionTree baseline (no tuning)")
    ap.add_argument("--data", required=True, help="Path to .parquet or .csv")
    ap.add_argument("--target", default="target")

    # Output locations
    ap.add_argument(
        "--results_dir",
        default=r"C:\Projects\U\Pro_fin3\Models\decision_tree\raw\results",
    )
    ap.add_argument(
        "--figures_dir",
        default=r"C:\Projects\U\Pro_fin3\Models\decision_tree\raw\figures",
    )

    # Split & evaluation settings
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Used only for the train/test split & sampling",
    )
    ap.add_argument(
        "--sample_frac", type=float, default=1.0, help="Use 1.0 for all data (default)."
    )
    ap.add_argument(
        "--metrics_sample_max",
        type=int,
        default=2_000_000,
        help="Cap for curve calculations",
    )
    ap.add_argument(
        "--optimize_threshold",
        action="store_true",
        help="Report extra metrics at Youden-J optimal threshold",
    )

    args = ap.parse_args()
    rs = check_random_state(args.random_state)

    t0 = time.time()
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dir(figures_dir)

    # Load data
    X, y, feature_names = load_data(Path(args.data), target=args.target)

    # Optional subsample (default 1.0 = no subsample)
    if args.sample_frac < 1.0:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=args.sample_frac,
            stratify=y,
            random_state=args.random_state,
        )

    # Split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Train (PURE DEFAULTS)
    # Default sklearn params: criterion='gini', splitter='best', max_depth=None,
    # min_samples_split=2, min_samples_leaf=1, max_features=None, class_weight=None, etc.
    model = DecisionTreeClassifier()  # <-- no params set

    fit_t0 = time.time()
    model.fit(Xtr, ytr)
    fit_t1 = time.time()

    # Predict
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    # Light sampling for curves to keep memory sane
    idx = (
        np.arange(len(yte))
        if len(yte) <= args.metrics_sample_max
        else rs.choice(len(yte), size=args.metrics_sample_max, replace=False)
    )
    ys, yp, ps = yte[idx], y_pred[idx], y_proba[idx]

    # Compute metrics (thr=0.5)
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

    # Optional metrics at ROC-optimal threshold
    thr_opt = None
    metrics_opt = None
    if args.optimize_threshold:
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

    # Feature importances
    imps = getattr(model, "feature_importances_", None)
    if imps is not None:
        order = np.argsort(imps)[::-1]
        top = order[:20]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(range(len(top)), imps[top][::-1])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([feature_names[i] for i in top][::-1], fontsize=9)
        ax.set_xlabel("Gini importance")
        ax.set_title("Top 20 feature importances")
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importances_top20.png", dpi=160)
        plt.close(fig)
        pd.DataFrame(
            {
                "feature": [feature_names[i] for i in order],
                "gini_importance": imps[order],
            }
        ).to_csv(results_dir / "feature_importances_full.csv", index=False)

    # Save ROC/PR points
    fpr, tpr, _ = roc_curve(ys, ps)
    prec, rec, _ = precision_recall_curve(ys, ps)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        results_dir / "roc_points.csv", index=False
    )
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(
        results_dir / "pr_points.csv", index=False
    )

    # Summary
    elapsed = time.time() - t0
    fit_time = fit_t1 - fit_t0
    summary = {
        "model": "DecisionTreeClassifier",
        "data": str(Path(args.data).resolve()),
        "target": args.target,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "used_for_metrics": int(len(ys)),
        "params": model.get_params(),  # pure defaults
        "metrics_thr0.5": metrics,
        "metrics_thr_opt": metrics_opt,
        "threshold_opt": thr_opt,
        "tree_stats": {
            "depth": int(model.get_depth()),
            "n_leaves": int(model.get_n_leaves()),
            "node_count": int(model.tree_.node_count),
        },
        "fit_time_seconds": round(fit_time, 3),
        "total_time_seconds": round(elapsed, 3),
        "test_size": args.test_size,
        "sample_frac": args.sample_frac,
        "random_state_split": args.random_state,
    }

    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "classification_report_thr0.5.txt").write_text(
        classification_report(ys, (ps >= 0.5).astype(int), digits=4, zero_division=0),
        encoding="utf-8",
    )

    print(
        f"[RAW DT] depth={summary['tree_stats']['depth']}, leaves={summary['tree_stats']['n_leaves']}, fit_time={fit_time:.2f}s"
    )
    print(
        f"[RAW DT] ROC-AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}"
    )
    if thr_opt is not None:
        print(f"[RAW DT] thr_opt={thr_opt:.4f}, F1_opt={metrics_opt['f1']:.6f}")
    print(f"Results -> {results_dir.resolve()}")
    print(f"Figures -> {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
