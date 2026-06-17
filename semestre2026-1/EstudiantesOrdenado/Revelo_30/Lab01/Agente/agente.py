import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, kruskal, pearsonr, spearmanr


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = ROOT_DIR / "artifacts"
PENGUINS_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
)


@dataclass
class RunnerContext:
    dataset_name: str
    data: pd.DataFrame
    artifacts_dir: Path
    plots_dir: Path
    registry_path: Path


def load_dataset(dataset_name: str, dataset_path: str | None) -> pd.DataFrame:
    if dataset_path:
        path = Path(dataset_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("Only CSV datasets are supported by this runner.")
        return pd.read_csv(path)

    if dataset_name == "penguins":
        return pd.read_csv(PENGUINS_URL)

    raise ValueError("A dataset path is required for datasets other than 'penguins'.")


def make_context(dataset_name: str, dataset_path: str | None) -> RunnerContext:
    artifacts_dir = ARTIFACTS_ROOT / dataset_name
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return RunnerContext(
        dataset_name=dataset_name,
        data=load_dataset(dataset_name, dataset_path),
        artifacts_dir=artifacts_dir,
        plots_dir=plots_dir,
        registry_path=artifacts_dir / "05_visual_registry.json",
    )


def records_to_json(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: records_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [records_to_json(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(records_to_json(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in numeric_columns(df)]


def low_cardinality_columns(df: pd.DataFrame, threshold: int = 10) -> list[str]:
    return [col for col in df.columns if df[col].nunique(dropna=True) <= threshold]


def pairwise_complete(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    subset = df[columns].replace([np.inf, -np.inf], np.nan)
    return subset.dropna()


def save_plot_registry_entry(context: RunnerContext, entry: dict[str, Any]) -> None:
    if context.registry_path.exists():
        registry = read_json(context.registry_path)
    else:
        registry = {"dataset_name": context.dataset_name, "plots": []}

    plots = [
        item for item in registry.get("plots", []) if item.get("path") != entry["path"]
    ]
    plots.append(entry)
    registry["plots"] = sorted(plots, key=lambda item: item["path"])
    write_json(context.registry_path, registry)


def op_profile_dataset(
    context: RunnerContext, params: dict[str, Any]
) -> dict[str, Any]:
    df = context.data
    profile = {
        "dataset_name": context.dataset_name,
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }
    write_json(
        context.artifacts_dir / "rows_cols.json",
        [{"num_rows": profile["num_rows"], "num_cols": profile["num_cols"]}],
    )
    return profile


def op_infer_schema(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    df = context.data
    numeric = numeric_columns(df)
    categorical = categorical_columns(df)
    write_json(
        context.artifacts_dir / "numeric_cols.json",
        [{"numeric_columns": col} for col in numeric],
    )
    write_json(
        context.artifacts_dir / "categorical_cols.json",
        [{"categorical_columns": col} for col in categorical],
    )
    return {
        "dataset_name": context.dataset_name,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
    }


def op_missing_report(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    counts = context.data.isna().sum().to_dict()
    write_json(context.artifacts_dir / "null_counts.json", [counts])
    return counts


def op_duplicates_report(
    context: RunnerContext, params: dict[str, Any]
) -> dict[str, Any]:
    duplicated_rows = int(context.data.duplicated().sum())
    payload = {"duplicated_rows": duplicated_rows}
    write_json(context.artifacts_dir / "duplicated_rows.json", [payload])
    return payload


def op_low_cardinality_report(
    context: RunnerContext, params: dict[str, Any]
) -> dict[str, Any]:
    threshold = int(params.get("threshold", 10))
    columns = low_cardinality_columns(context.data, threshold=threshold)
    write_json(
        context.artifacts_dir / "low_cardinality_cols.json",
        [{"low_cardinality_columns": col} for col in columns],
    )
    return {"threshold": threshold, "low_cardinality_columns": columns}


def op_numeric_summary(
    context: RunnerContext, params: dict[str, Any]
) -> list[dict[str, Any]]:
    df = context.data
    summaries: list[dict[str, Any]] = []
    for col in numeric_columns(df):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        summaries.append(
            {
                "column": col,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            }
        )
    write_json(context.artifacts_dir / "numeric_summary.json", summaries)
    return summaries


def op_categorical_summary(
    context: RunnerContext, params: dict[str, Any]
) -> list[dict[str, Any]]:
    df = context.data
    summaries: list[dict[str, Any]] = []
    total_rows = len(df)
    for col in categorical_columns(df):
        counts = df[col].fillna("<MISSING>").value_counts(dropna=False)
        for category, count in counts.items():
            summaries.append(
                {
                    "column": col,
                    "category": category,
                    "count": int(count),
                    "percentage": round(float(count / total_rows * 100), 2),
                }
            )
    write_json(context.artifacts_dir / "categorical_counts.json", summaries)
    return summaries


def op_crosstab(context: RunnerContext, params: dict[str, Any]) -> list[dict[str, Any]]:
    x = params["x"]
    y = params["y"]
    table = pd.crosstab(
        context.data[x].fillna("<MISSING>"), context.data[y].fillna("<MISSING>")
    )
    result = table.reset_index().to_dict(orient="records")
    write_json(context.artifacts_dir / f"cross_table_{x}_{y}.json", result)
    return result


def op_correlation_matrix(
    context: RunnerContext, params: dict[str, Any]
) -> list[dict[str, Any]]:
    method = params.get("method", "pearson")
    numeric_df = context.data[numeric_columns(context.data)].apply(
        pd.to_numeric, errors="coerce"
    )
    corr = numeric_df.corr(method=method)
    result = (
        corr.reset_index()
        .rename(columns={"index": "variable"})
        .to_dict(orient="records")
    )
    write_json(context.artifacts_dir / f"{method}_corr.json", result)
    return result


def describe_histogram(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    mean = float(clean.mean())
    median = float(clean.median())
    std = float(clean.std())
    iqr = float(clean.quantile(0.75) - clean.quantile(0.25))
    skewness = float(clean.skew())

    if skewness > 0.2:
        skew_label = "skewed to the right"
    elif skewness < -0.2:
        skew_label = "skewed to the left"
    else:
        skew_label = "approximately symmetric"

    center_label = (
        "closer to the mean"
        if abs(mean - median) <= max(abs(mean) * 0.1, 1e-9)
        else "closer to the median"
    )
    spread_label = (
        "small relative spread"
        if std <= max(abs(mean) * 0.5, 1e-9)
        else "large relative spread"
    )

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = int(((clean < lower) | (clean > upper)).sum())

    counts, _ = np.histogram(clean, bins=20)
    gaps = int((counts == 0).sum())

    return {
        "skewness": skew_label,
        "center": center_label,
        "spread": spread_label,
        "outliers": outliers,
        "gaps": gaps,
        "mean": mean,
        "median": median,
        "max": float(clean.max()),
        "min": float(clean.min()),
    }


def op_histogram_descriptions(
    context: RunnerContext, params: dict[str, Any]
) -> dict[str, Any]:
    descriptions = {
        col: describe_histogram(context.data[col])
        for col in numeric_columns(context.data)
    }
    write_json(context.artifacts_dir / "histogram_descriptions.json", descriptions)
    return descriptions


def plot_output_path(context: RunnerContext, name: str) -> Path:
    return context.plots_dir / f"{name}.png"


def finalize_plot(
    context: RunnerContext, fig: plt.Figure, entry: dict[str, Any]
) -> dict[str, Any]:
    output = context.artifacts_dir / entry["path"].replace("artifacts/", "")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    save_plot_registry_entry(context, entry)
    return entry


def op_plot_count(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    column = params["column"]
    output = plot_output_path(context, f"categorical_count_plot_{column}")
    counts = context.data[column].fillna("<MISSING>").value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax)
    ax.set_title(f"Count plot of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    entry = {
        "kind": "count",
        "path": f"artifacts/plots/{output.name}",
        "variables": {"column": column},
        "evidence_refs": [f"artifacts/categorical_counts.json:{column}"],
    }
    return finalize_plot(context, fig, entry)


def op_plot_hist(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    column = params["column"]
    output = plot_output_path(context, f"histogram_{column}")
    series = pd.to_numeric(context.data[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=20)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    entry = {
        "kind": "hist",
        "path": f"artifacts/plots/{output.name}",
        "variables": {"column": column},
        "evidence_refs": [f"artifacts/histogram_descriptions.json:{column}"],
    }
    return finalize_plot(context, fig, entry)


def op_plot_box(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    # Accept x/y as a fallback alias from agent plans, but prefer num_col/cat_col.
    num_col = params.get("num_col") or params.get("y")
    cat_col = params.get("cat_col") or params.get("x")
    if not num_col or not cat_col:
        raise ValueError("plot_box requires 'num_col' and 'cat_col'.")
    output = plot_output_path(context, f"boxplot_{num_col}_by_{cat_col}")
    plot_df = context.data[[cat_col, num_col]].copy()
    plot_df[cat_col] = plot_df[cat_col].fillna("<MISSING>")
    plot_df[num_col] = pd.to_numeric(plot_df[num_col], errors="coerce")
    plot_df = plot_df.dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x=cat_col, y=num_col, ax=ax)
    ax.set_title(f"Box plot of {num_col} by {cat_col}")
    ax.tick_params(axis="x", rotation=45)
    entry = {
        "kind": "box",
        "path": f"artifacts/plots/{output.name}",
        "variables": {"num_col": num_col, "cat_col": cat_col},
        "evidence_refs": [
            f"artifacts/numeric_summary.json:{num_col}",
            f"artifacts/categorical_counts.json:{cat_col}",
        ],
    }
    return finalize_plot(context, fig, entry)


def op_plot_scatter(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    x = params["x"]
    y = params["y"]
    hue = params.get("hue")
    suffix = f"_by_{hue}" if hue else ""
    output = plot_output_path(context, f"scatterplot_{x}_vs_{y}{suffix}")
    columns = [x, y] + ([hue] if hue else [])
    plot_df = context.data[columns].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    if hue:
        plot_df[hue] = plot_df[hue].fillna("<MISSING>")
    plot_df = plot_df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"Scatter plot of {x} vs {y}")
    entry = {
        "kind": "scatter",
        "path": f"artifacts/plots/{output.name}",
        "variables": {"x": x, "y": y, "hue": hue},
        "evidence_refs": [
            f"artifacts/pearson_corr.json:{x}:{y}",
            f"artifacts/spearman_corr.json:{x}:{y}",
        ],
    }
    return finalize_plot(context, fig, entry)


def op_plot_heatmap_corr(
    context: RunnerContext, params: dict[str, Any]
) -> dict[str, Any]:
    method = params.get("method", "pearson")
    output = plot_output_path(context, f"heatmap_{method}_corr")
    numeric_df = context.data[numeric_columns(context.data)].apply(
        pd.to_numeric, errors="coerce"
    )
    corr = numeric_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(f"Heatmap of {method.title()} Correlation Matrix")
    entry = {
        "kind": "heatmap_corr",
        "path": f"artifacts/plots/{output.name}",
        "variables": {"method": method},
        "evidence_refs": [f"artifacts/{method}_corr.json"],
    }
    return finalize_plot(context, fig, entry)


def op_pearson_test(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    x = params["x"]
    y = params["y"]
    pair = pairwise_complete(context.data, [x, y])
    statistic, p_value = pearsonr(pair[x], pair[y])
    return {
        "test": "Pearson",
        "variables": {"x": x, "y": y},
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": int(len(pair)),
    }


def op_spearman_test(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    x = params["x"]
    y = params["y"]
    pair = pairwise_complete(context.data, [x, y])
    statistic, p_value = spearmanr(pair[x], pair[y])
    return {
        "test": "Spearman",
        "variables": {"x": x, "y": y},
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": int(len(pair)),
    }


def group_samples(
    context: RunnerContext, value_col: str, group_col: str
) -> list[pd.Series]:
    clean = context.data[[value_col, group_col]].copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean[group_col] = clean[group_col].fillna("<MISSING>")
    clean = clean.dropna(subset=[value_col])
    groups = [
        group[value_col] for _, group in clean.groupby(group_col) if len(group) > 0
    ]
    if len(groups) < 2:
        raise ValueError("At least two groups are required for this test.")
    return groups


def op_anova_test(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    value_col = params["value_col"]
    group_col = params["group_col"]
    groups = group_samples(context, value_col, group_col)
    statistic, p_value = f_oneway(*groups)
    return {
        "test": "ANOVA",
        "variables": {"value_col": value_col, "group_col": group_col},
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_groups": len(groups),
    }


def op_kruskal_test(context: RunnerContext, params: dict[str, Any]) -> dict[str, Any]:
    value_col = params["value_col"]
    group_col = params["group_col"]
    groups = group_samples(context, value_col, group_col)
    statistic, p_value = kruskal(*groups)
    return {
        "test": "Kruskal",
        "variables": {"value_col": value_col, "group_col": group_col},
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_groups": len(groups),
    }


def op_chi_square_test(
    context: RunnerContext, params: dict[str, Any]
) -> dict[str, Any]:
    x = params["x"]
    y = params["y"]
    table = pd.crosstab(
        context.data[x].fillna("<MISSING>"), context.data[y].fillna("<MISSING>")
    )
    statistic, p_value, dof, _ = chi2_contingency(table)
    return {
        "test": "Chi-square",
        "variables": {"x": x, "y": y},
        "statistic": float(statistic),
        "p_value": float(p_value),
        "dof": int(dof),
    }


OPERATIONS = {
    "profile_dataset": op_profile_dataset,
    "infer_schema": op_infer_schema,
    "missing_report": op_missing_report,
    "duplicates_report": op_duplicates_report,
    "low_cardinality_report": op_low_cardinality_report,
    "numeric_summary": op_numeric_summary,
    "categorical_summary": op_categorical_summary,
    "crosstab": op_crosstab,
    "correlation_matrix": op_correlation_matrix,
    "histogram_descriptions": op_histogram_descriptions,
    "plot_count": op_plot_count,
    "plot_hist": op_plot_hist,
    "plot_box": op_plot_box,
    "plot_scatter": op_plot_scatter,
    "plot_heatmap_corr": op_plot_heatmap_corr,
    "pearson_test": op_pearson_test,
    "spearman_test": op_spearman_test,
    "anova_test": op_anova_test,
    "kruskal_test": op_kruskal_test,
    "chi_square_test": op_chi_square_test,
}


def load_atomic_artifact(context: RunnerContext, name: str) -> Any:
    path = context.artifacts_dir / name
    return read_json(path) if path.exists() else None


def consolidate_observe(context: RunnerContext) -> dict[str, Any]:
    payload = {
        "dataset_name": context.dataset_name,
        "rows_cols": (load_atomic_artifact(context, "rows_cols.json") or [{}])[0],
        "schema": {
            "numeric_columns": [
                item["numeric_columns"]
                for item in load_atomic_artifact(context, "numeric_cols.json") or []
            ],
            "categorical_columns": [
                item["categorical_columns"]
                for item in load_atomic_artifact(context, "categorical_cols.json") or []
            ],
        },
        "missing_values": (load_atomic_artifact(context, "null_counts.json") or [{}])[
            0
        ],
        "duplicated_rows": (
            load_atomic_artifact(context, "duplicated_rows.json") or [{}]
        )[0],
        "low_cardinality_columns": [
            item["low_cardinality_columns"]
            for item in load_atomic_artifact(context, "low_cardinality_cols.json") or []
        ],
        "evidence_refs": [
            "artifacts/rows_cols.json",
            "artifacts/numeric_cols.json",
            "artifacts/categorical_cols.json",
            "artifacts/null_counts.json",
            "artifacts/duplicated_rows.json",
            "artifacts/low_cardinality_cols.json",
        ],
    }
    write_json(context.artifacts_dir / "00_raw_profile.json", payload)
    return payload


def consolidate_describe(context: RunnerContext) -> dict[str, Any]:
    crosstabs: dict[str, Any] = {}
    for path in sorted(context.artifacts_dir.glob("cross_table_*.json")):
        crosstabs[path.name] = read_json(path)

    payload = {
        "dataset_name": context.dataset_name,
        "numeric_summary": load_atomic_artifact(context, "numeric_summary.json") or [],
        "categorical_summary": load_atomic_artifact(context, "categorical_counts.json")
        or [],
        "crosstabs": crosstabs,
        "correlations": {
            "pearson": load_atomic_artifact(context, "pearson_corr.json") or [],
            "spearman": load_atomic_artifact(context, "spearman_corr.json") or [],
        },
        "histogram_descriptions": load_atomic_artifact(
            context, "histogram_descriptions.json"
        )
        or {},
        "evidence_refs": [
            "artifacts/numeric_summary.json",
            "artifacts/categorical_counts.json",
            "artifacts/pearson_corr.json",
            "artifacts/spearman_corr.json",
            "artifacts/histogram_descriptions.json",
        ]
        + [f"artifacts/{name}" for name in crosstabs],
    }
    write_json(context.artifacts_dir / "04_descriptive_stats.json", payload)
    if not context.registry_path.exists():
        write_json(
            context.registry_path, {"dataset_name": context.dataset_name, "plots": []}
        )
    return payload


def write_interpretation_artifacts(
    context: RunnerContext, plan: dict[str, Any], test_results: list[dict[str, Any]]
) -> None:
    artifacts_to_write = plan.get("artifacts_to_write", {})
    for relative_path, payload in artifacts_to_write.items():
        output = context.artifacts_dir / Path(relative_path).name
        if relative_path.endswith("08_tests.json"):
            if payload:
                write_json(output, payload)
            elif test_results:
                write_json(output, test_results)
            continue

        if relative_path.endswith("report.md"):
            if isinstance(payload, str) and payload:
                output.write_text(payload, encoding="utf-8")
            continue

        if payload not in ({}, None):
            write_json(output, payload)


def build_phase_context(context: RunnerContext, phase: str) -> dict[str, Any]:
    available: dict[str, Any] = {}
    for path in sorted(context.artifacts_dir.rglob("*")):
        if path.is_dir():
            continue
        relative_path = (
            f"artifacts/{path.relative_to(context.artifacts_dir).as_posix()}"
        )
        if path.suffix == ".json":
            available[relative_path] = read_json(path)
        elif path.suffix == ".md":
            available[relative_path] = path.read_text(encoding="utf-8")
        elif path.suffix == ".png":
            available[relative_path] = {"path": relative_path}
    return {
        "phase": phase,
        "dataset_name": context.dataset_name,
        "available_artifacts": available,
    }


def execute_plan(context: RunnerContext, plan: dict[str, Any]) -> dict[str, Any]:
    phase = plan["phase"]
    test_results: list[dict[str, Any]] = []
    operation_results: list[dict[str, Any]] = []

    for action in plan.get("actions_to_run", []):
        op_name = action["op"]
        params = action.get("params", {})
        if op_name not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {op_name}")
        result = OPERATIONS[op_name](context, params)
        operation_results.append({"op": op_name, "params": params, "result": result})
        if op_name.endswith("_test"):
            test_results.append(result)

    if phase == "OBSERVE":
        consolidate_observe(context)
    elif phase == "DESCRIBE":
        consolidate_describe(context)
    elif phase == "HYPOTHESIZE_AND_CONCLUDE":
        write_interpretation_artifacts(context, plan, test_results)

    return {
        "phase": phase,
        "dataset_name": context.dataset_name,
        "executed_actions": operation_results,
        "tests_written": bool(test_results),
        "available_artifacts": build_phase_context(context, phase)[
            "available_artifacts"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runner for the agent-based observatory workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    context_parser = subparsers.add_parser(
        "context", help="Build the artifact context JSON for a phase."
    )
    context_parser.add_argument("--phase", required=True)
    context_parser.add_argument("--dataset-name", default="penguins")
    context_parser.add_argument("--dataset-path")
    context_parser.add_argument("--output")

    execute_parser = subparsers.add_parser(
        "execute-plan", help="Execute a planner JSON file."
    )
    execute_parser.add_argument("--plan", required=True)
    execute_parser.add_argument("--dataset-name", default="penguins")
    execute_parser.add_argument("--dataset-path")
    execute_parser.add_argument("--output")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = make_context(args.dataset_name, args.dataset_path)

    if args.command == "context":
        payload = build_phase_context(context, args.phase)
    else:
        plan = read_json(Path(args.plan))
        payload = execute_plan(context, plan)

    if getattr(args, "output", None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
    else:
        print(json.dumps(records_to_json(payload), indent=2))


if __name__ == "__main__":
    main()
