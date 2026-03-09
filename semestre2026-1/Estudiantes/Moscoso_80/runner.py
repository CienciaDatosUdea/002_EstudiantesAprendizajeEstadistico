"""
runner.py — Reproducible Data Observatory Runner
Course: Computational Physics / Statistical Learning 2026-1
Student: Moscoso_80

USAGE:
  Single-dataset lab (e.g. Lab 01):
    python runner.py --lab Lab_01_pinguinos --phase OBSERVE  --origin seaborn --dataset penguins
    python runner.py --lab Lab_01_pinguinos --phase DESCRIBE --origin seaborn --dataset penguins

  Multi-dataset lab (e.g. Lab 02) — use --dataset-label to isolate each dataset:
    python runner.py --lab Lab_02_pandas --dataset-label eurusd        --phase OBSERVE --origin url  --dataset https://raw.githubusercontent.com/.../EURUSD.csv
    python runner.py --lab Lab_02_pandas --dataset-label breast_cancer --phase OBSERVE --origin url  --dataset https://archive.ics.uci.edu/...

OUTPUT STRUCTURE:
  Single dataset:
    semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/artifacts/
    semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/artifacts/plots/

  Multi-dataset (--dataset-label provided):
    semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/artifacts/{dataset-label}/
    semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/artifacts/{dataset-label}/plots/

  run_log.json is ALWAYS at the lab root (shared across all datasets in a lab):
    semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/artifacts/run_log.json

  student_log.md lives beside the artifacts folder:
    semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/student_log.md

Artifacts are NEVER overwritten — a phase + timestamp suffix is always appended.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42  # Fixed seed for ALL stochastic operations — ensures reproducibility
np.random.seed(RANDOM_SEED)

# Root that is NEVER changed — all output must live inside here.
_STUDENT_ROOT = Path("semestre2026-1/Estudiantes/Moscoso_80")

# These three globals are set dynamically in main() once --lab and
# --dataset-label are parsed. Every function reads them from module scope.
# Do NOT hard-code paths anywhere else in this file.
LAB_DIR: Path       = _STUDENT_ROOT  # overwritten in main()
ARTIFACTS_DIR: Path = _STUDENT_ROOT  # overwritten in main()
PLOTS_DIR: Path     = _STUDENT_ROOT  # overwritten in main()


def _resolve_paths(lab: str, dataset_label: str | None) -> None:
    """
    Set the three module-level path globals based on CLI arguments.
    Must be called once at the start of main() before any other function runs.

    Path logic:
      - LAB_DIR      → .../Laboratorios/{lab}/
      - If dataset_label is provided:
          ARTIFACTS_DIR → LAB_DIR/artifacts/{dataset_label}/
          PLOTS_DIR     → LAB_DIR/artifacts/{dataset_label}/plots/
      - If dataset_label is None (single-dataset lab):
          ARTIFACTS_DIR → LAB_DIR/artifacts/
          PLOTS_DIR     → LAB_DIR/artifacts/plots/

    The run_log.json is ALWAYS at LAB_DIR/artifacts/run_log.json regardless
    of dataset_label (it is the shared audit trail for the entire lab).

    Inputs:
        lab           (str):       Lab folder name, e.g. 'Lab_01_pinguinos'.
        dataset_label (str|None):  Optional dataset sub-folder, e.g. 'eurusd'.

    Output:
        None (mutates module globals LAB_DIR, ARTIFACTS_DIR, PLOTS_DIR).
    """
    global LAB_DIR, ARTIFACTS_DIR, PLOTS_DIR

    LAB_DIR = _STUDENT_ROOT / "Laboratorios" / lab

    if dataset_label:
        ARTIFACTS_DIR = LAB_DIR / "artifacts" / dataset_label
    else:
        ARTIFACTS_DIR = LAB_DIR / "artifacts"

    PLOTS_DIR = ARTIFACTS_DIR / "plots"


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_directories():
    """
    Create the required directory tree for the current lab if it does not exist.
    Also creates an empty student_log.md at the lab root if one does not exist yet.
    This must be the very first call in any run so all subsequent writes succeed.

    Inputs:  None (reads module globals LAB_DIR, ARTIFACTS_DIR, PLOTS_DIR).
    Actions: Creates LAB_DIR, ARTIFACTS_DIR, PLOTS_DIR. Touches student_log.md.
    Output:  None (side effect: directories and placeholder file exist on disk).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create a blank student_log.md at the lab root if it doesn't exist yet.
    # The student fills this in manually — the runner never writes to it.
    student_log = LAB_DIR / "student_log.md"
    if not student_log.exists():
        student_log.write_text(
            "# Student Log\n\n"
            "> Fill in your Phase 0 pre-analysis here before running the agent.\n\n"
            "## Phase 0 — Pre-Analysis\n\n"
            "**Dataset:**\n\n"
            "1. What is the dataset about? ...\n"
            "2. What are you curious about? ...\n"
            "3. Which variables do you think are related? ...\n"
            "4. What would surprise you? ...\n",
            encoding="utf-8"
        )
        print(f"  [SETUP] Created blank student_log.md at {student_log}")

    print(f"[SETUP] Directory tree ready:")
    print(f"  └─ {ARTIFACTS_DIR}")
    print(f"  └─ {PLOTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(origin: str, dataset: str) -> pd.DataFrame:
    """
    Load a dataset from three possible origins.

    Inputs:
        origin  (str): 'seaborn' → load a named seaborn built-in dataset.
                       'csv'     → load from a local file path.
                       'url'     → load from a remote URL (CSV format).
        dataset (str): The seaborn dataset name (e.g. 'penguins'),
                       the path to a .csv file, or a full https:// URL.

    Actions:
        Reads the dataset into a DataFrame. Prints shape for a quick sanity check.

    Output:
        pd.DataFrame — the loaded dataset, unmodified.

    Raises:
        ValueError        if origin is not 'seaborn', 'csv', or 'url'.
        FileNotFoundError if a CSV path does not exist on disk.
    """
    print(f"\n[LOAD] origin='{origin}' | dataset='{dataset}'")

    if origin == "seaborn":
        df = sns.load_dataset(dataset)
        print(f"[LOAD] Seaborn dataset '{dataset}' loaded successfully.")

    elif origin == "csv":
        csv_path = Path(dataset)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path.resolve()}")
        df = pd.read_csv(csv_path)
        print(f"[LOAD] CSV loaded from '{dataset}'.")

    elif origin == "url":
        # pandas.read_csv accepts HTTP/HTTPS URLs directly.
        # This covers the remote datasets used in Lab 02 (EUR/USD, Breast Cancer).
        df = pd.read_csv(dataset)
        print(f"[LOAD] CSV loaded from remote URL.")

    else:
        raise ValueError(f"Unknown origin '{origin}'. Valid options: 'seaborn', 'csv', 'url'.")

    print(f"[LOAD] Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# JSON UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def safe_json(obj):
    """
    Recursively convert any object to a JSON-safe Python type.
    Numpy integers/floats are cast to Python natives.
    NaN and Inf become null (None → JSON null) so the output is always valid JSON.

    Inputs:
        obj: Any Python/numpy/pandas object.

    Actions:
        Recursively walks dicts and lists; converts scalars.

    Output:
        A JSON-serializable copy of obj.
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    # Handle pandas NA/NaT/None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def write_artifact(filename: str, data: dict, phase: str) -> Path:
    """
    Write a JSON artifact to ARTIFACTS_DIR with an immutable naming convention.
    Files are NEVER overwritten — a '__PHASE__TIMESTAMP' suffix is always added.

    Inputs:
        filename (str): Base name, e.g. '00_raw_profile.json'.
        data     (dict): Data to serialize.
        phase    (str):  Current phase label (e.g. 'OBSERVE').

    Actions:
        Converts data to safe JSON, writes the file, prints the path.

    Output:
        Path — the full path of the written artifact.
    """
    stem      = Path(filename).stem
    ext       = Path(filename).suffix or ".json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{stem}__{phase}__{timestamp}{ext}"
    out_path  = ARTIFACTS_DIR / safe_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(data), f, indent=2, ensure_ascii=False)

    print(f"  [WRITE] {out_path.name}")
    return out_path


def log_operation(op_name: str, params: dict, output_path: str, phase: str):
    """
    Append one entry to run_log.json. This is the shared audit trail for the
    entire lab — it always lives at LAB_DIR/artifacts/run_log.json, regardless
    of whether a --dataset-label sub-folder is in use.

    This means all datasets within the same lab share one traceable history,
    which makes it easy to review what was run and when across the full lab.

    Inputs:
        op_name     (str):  Operation name (e.g. 'profile_dataset').
        params      (dict): Parameters used (column names, methods, etc.).
        output_path (str):  Path to the artifact or plot produced.
        phase       (str):  Current phase.

    Actions:
        Reads existing run_log.json (or creates it), appends the new entry,
        and writes it back atomically.

    Output:
        None (side effect: run_log.json updated on disk).
    """
    # run_log always lives at the lab-level artifacts root, never inside a dataset subfolder.
    log_path = LAB_DIR / "artifacts" / "run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log = {"runs": []}
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = {"runs": []}

    log["runs"].append({
        "timestamp": datetime.now().isoformat(),
        "phase":     phase,
        "operation": op_name,
        "params":    safe_json(params),
        "output":    str(output_path)
    })

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG — OBSERVE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, phase: str) -> dict:
    """
    Generate a high-level snapshot of the dataset: shape, column names, dtypes,
    and memory footprint. This is always the very first operation.

    Inputs:
        df    (pd.DataFrame): The raw loaded dataset.
        phase (str):          Current phase label.

    Actions:
        Computes shape, column metadata, memory usage.
        Writes artifact 00_raw_profile.json.

    Output:
        dict — the profile summary.
    """
    print("\n[OP] profile_dataset")
    print("  ↳ Why: Before touching any data we need to know its dimensions,")
    print("         column names, and types — the 'shape of the problem'.")

    profile = {
        "n_rows":         int(df.shape[0]),
        "n_cols":         int(df.shape[1]),
        "columns":        list(df.columns),
        "dtypes":         {col: str(df[col].dtype) for col in df.columns},
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }

    path = write_artifact("00_raw_profile.json", profile, phase)
    log_operation("profile_dataset", {}, str(path), phase)
    return profile


def infer_schema(df: pd.DataFrame, phase: str) -> dict:
    """
    Classify every column into a semantic type: numeric, categorical, boolean,
    or datetime. Also flags columns with very high cardinality.

    Inputs:
        df    (pd.DataFrame): The raw loaded dataset.
        phase (str):          Current phase label.

    Actions:
        Iterates columns, assigns semantic_type, counts unique values,
        flags high-cardinality categoricals (>20 unique values).
        Writes artifact 01_schema.json.

    Output:
        dict — {column_name: {dtype, semantic_type, n_unique, high_cardinality}}
    """
    print("\n[OP] infer_schema")
    print("  ↳ Why: The semantic type of a column determines which operations")
    print("         are valid. You cannot compute a mean on a city name.")

    schema = {}
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))

        if pd.api.types.is_bool_dtype(df[col]):
            col_type = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        else:
            col_type = "categorical"

        schema[col] = {
            "dtype":           str(df[col].dtype),
            "semantic_type":   col_type,
            "n_unique":        n_unique,
            "high_cardinality": n_unique > 20 and col_type == "categorical"
        }

    path = write_artifact("01_schema.json", schema, phase)
    log_operation("infer_schema", {}, str(path), phase)
    return schema


def missing_report(df: pd.DataFrame, phase: str) -> dict:
    """
    Audit every column for missing values. Reports counts and percentages.

    Inputs:
        df    (pd.DataFrame): The raw loaded dataset.
        phase (str):          Current phase label.

    Actions:
        Counts NaN per column, computes overall missingness rate.
        Writes artifact 02_missing_report.json.

    Output:
        dict — {total_cells, total_missing, pct_missing_overall, by_column: {...}}
    """
    print("\n[OP] missing_report")
    print("  ↳ Why: Missing data is not random noise — it can introduce systematic")
    print("         bias. We must quantify it before any statistic is computed.")

    total_cells = int(df.shape[0] * df.shape[1])
    total_missing = int(df.isna().sum().sum())

    by_col = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        by_col[col] = {
            "n_missing":   n,
            "pct_missing": round(n / df.shape[0] * 100, 2)
        }

    report = {
        "total_cells":          total_cells,
        "total_missing":        total_missing,
        "pct_missing_overall":  round(total_missing / total_cells * 100, 2),
        "by_column":            by_col
    }

    path = write_artifact("02_missing_report.json", report, phase)
    log_operation("missing_report", {}, str(path), phase)
    return report


def duplicates_report(df: pd.DataFrame, phase: str) -> dict:
    """
    Detect fully duplicated rows in the dataset.

    Inputs:
        df    (pd.DataFrame): The raw loaded dataset.
        phase (str):          Current phase label.

    Actions:
        Uses pandas .duplicated() to find all repeated rows.
        Writes artifact 03_duplicates_report.json.

    Output:
        dict — {n_duplicates, pct_duplicates, duplicate_indices}
    """
    print("\n[OP] duplicates_report")
    print("  ↳ Why: Duplicate rows inflate counts and skew all statistics.")
    print("         Knowing they exist lets us decide whether to remove them.")

    dup_mask    = df.duplicated()
    n_dups      = int(dup_mask.sum())
    dup_indices = df[dup_mask].index.tolist()

    report = {
        "n_duplicates":    n_dups,
        "pct_duplicates":  round(n_dups / df.shape[0] * 100, 2),
        "duplicate_indices": dup_indices
    }

    path = write_artifact("03_duplicates_report.json", report, phase)
    log_operation("duplicates_report", {}, str(path), phase)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG — DESCRIBE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def numeric_summary(df: pd.DataFrame, phase: str) -> dict:
    """
    Compute descriptive statistics for every numeric column:
    mean, median, std, min, max, Q1, Q3, IQR, skewness, and kurtosis.

    Inputs:
        df    (pd.DataFrame): Dataset (may include NaNs; they are dropped per column).
        phase (str):          Current phase label.

    Actions:
        Selects numeric columns, computes per-column statistics.
        Writes artifact 04a_numeric_summary.json.

    Output:
        dict — {column: {count, mean, median, std, min, max, q1, q3, iqr, skewness, kurtosis}}
    """
    print("\n[OP] numeric_summary")
    print("  ↳ Why: Mean/median reveal the center of the distribution.")
    print("         Std/IQR reveal its spread. Skewness tells us if the")
    print("         distribution has a long tail — common in real-world data.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary = {}

    for col in numeric_cols:
        s  = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        summary[col] = {
            "count":    int(s.count()),
            "mean":     float(s.mean()),
            "median":   float(s.median()),
            "std":      float(s.std()),
            "min":      float(s.min()),
            "max":      float(s.max()),
            "q1":       q1,
            "q3":       q3,
            "iqr":      round(q3 - q1, 4),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurt())
        }

    path = write_artifact("04a_numeric_summary.json", summary, phase)
    log_operation("numeric_summary", {}, str(path), phase)
    return summary


def categorical_summary(df: pd.DataFrame, phase: str) -> dict:
    """
    Compute frequency tables for every categorical / object / boolean column.

    Inputs:
        df    (pd.DataFrame): Dataset.
        phase (str):          Current phase label.

    Actions:
        For each categorical column: counts unique values, computes value_counts
        and proportions (including NaN as a category).
        Writes artifact 04b_categorical_summary.json.

    Output:
        dict — {column: {n_unique, top_value, value_counts, proportions}}
    """
    print("\n[OP] categorical_summary")
    print("  ↳ Why: Frequency tables reveal class imbalances and dominant categories")
    print("         that will influence any group comparison or model training.")

    cat_cols = df.select_dtypes(include=["str", "category", "bool"]).columns.tolist()
    summary  = {}

    for col in cat_cols:
        vc  = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(normalize=True, dropna=False)
        summary[col] = {
            "n_unique":     int(df[col].nunique(dropna=True)),
            "top_value":    str(vc.index[0]) if len(vc) > 0 else None,
            "value_counts": {str(k): int(v) for k, v in vc.items()},
            "proportions":  {str(k): round(float(v), 4) for k, v in pct.items()}
        }

    path = write_artifact("04b_categorical_summary.json", summary, phase)
    log_operation("categorical_summary", {}, str(path), phase)
    return summary


def crosstab(df: pd.DataFrame, col_a: str, col_b: str, phase: str) -> dict:
    """
    Build a contingency table (cross-tabulation) between two categorical columns.
    This is the raw material for a chi-square independence test.

    Inputs:
        df    (pd.DataFrame): Dataset.
        col_a (str):          First categorical column (index).
        col_b (str):          Second categorical column (columns).
        phase (str):          Current phase label.

    Actions:
        Calls pd.crosstab. Writes artifact 04c_crosstab_{col_a}_{col_b}.json.

    Output:
        dict — {index_col, columns_col, table: {col_a_val: {col_b_val: count}}}
    """
    print(f"\n[OP] crosstab(a='{col_a}', b='{col_b}')")
    print(f"  ↳ Why: A contingency table shows how two categorical variables co-occur.")
    print(f"         High off-diagonal counts hint at an association between them.")

    ct = pd.crosstab(df[col_a], df[col_b])
    result = {
        "index_col":   col_a,
        "columns_col": col_b,
        "table":       safe_json(ct.to_dict())
    }

    fname = f"04c_crosstab_{col_a}_{col_b}.json"
    path  = write_artifact(fname, result, phase)
    log_operation("crosstab", {"a": col_a, "b": col_b}, str(path), phase)
    return result


def correlation_matrix(df: pd.DataFrame, method: str, phase: str) -> dict:
    """
    Compute a pairwise correlation matrix for all numeric columns.

    Three methods are supported — each with different assumptions:
      - 'pearson'  : measures LINEAR relationships; assumes normality.
      - 'spearman' : rank-based; robust to outliers and non-linearity.
      - 'kendall'  : rank-based; better for small samples.

    Inputs:
        df     (pd.DataFrame): Dataset.
        method (str):          'pearson', 'spearman', or 'kendall'.
        phase  (str):          Current phase label.

    Actions:
        Selects numeric columns, computes corr matrix, writes JSON artifact.

    Output:
        dict — {method, matrix: {col: {col: corr_value}}}
    """
    print(f"\n[OP] correlation_matrix(method='{method}')")
    print(f"  ↳ Why: Correlation quantifies the strength of relationships between")
    print(f"         numeric variables. Use Pearson for linear, Spearman for rank-based.")

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method=method)

    result = {
        "method": method,
        "matrix": safe_json(corr.to_dict())
    }

    path = write_artifact(f"04d_correlation_{method}.json", result, phase)
    log_operation("correlation_matrix", {"method": method}, str(path), phase)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG — PLOT OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _save_plot(fig, plot_name: str, phase: str, registry: list) -> str:
    """
    Save a matplotlib figure to PLOTS_DIR with an immutable timestamped filename,
    then register it in the visual registry list.

    Inputs:
        fig       (Figure):  The matplotlib Figure to save.
        plot_name (str):     Descriptive snake_case name (no extension).
        phase     (str):     Current phase label.
        registry  (list):    The running visual registry (mutated in place).

    Actions:
        Saves the figure as PNG at 120 DPI, closes the figure, appends metadata
        to registry, logs the operation.

    Output:
        str — the path where the plot was saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{plot_name}__{phase}__{timestamp}.png"
    out_path  = PLOTS_DIR / filename

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    entry = {
        "plot_id":   plot_name,
        "phase":     phase,
        "path":      str(out_path),
        "timestamp": timestamp
    }
    registry.append(entry)
    print(f"  [PLOT] Saved: {filename}")
    log_operation("save_plot", {"plot_name": plot_name}, str(out_path), phase)
    return str(out_path)


def plot_count(df: pd.DataFrame, x: str, phase: str, registry: list,
               hue: str = None) -> str:
    """
    Produce a count bar chart for a categorical column, optionally split by a hue.

    Inputs:
        df       (pd.DataFrame): Dataset.
        x        (str):          Categorical column to count.
        phase    (str):          Current phase label.
        registry (list):         Visual registry list (mutated).
        hue      (str, optional): Column used to color-split bars.

    Actions:
        Creates a seaborn countplot, saves via _save_plot.

    Output:
        str — path to the saved PNG.
    """
    print(f"\n[OP] plot_count(x='{x}', hue={hue})")
    print(f"  ↳ Why: Count plots reveal category frequencies and class imbalances at a glance.")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=x, hue=hue, ax=ax, palette="Set2")
    ax.set_title(f"Count of {x}" + (f" by {hue}" if hue else ""))
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    fig.tight_layout()

    name = f"count_{x}" + (f"_by_{hue}" if hue else "")
    return _save_plot(fig, name, phase, registry)


def plot_hist(df: pd.DataFrame, x: str, phase: str, registry: list,
              groupby: str = None) -> str:
    """
    Produce a histogram for a numeric column, optionally overlaid per group.

    Inputs:
        df       (pd.DataFrame): Dataset.
        x        (str):          Numeric column to plot.
        phase    (str):          Current phase label.
        registry (list):         Visual registry list (mutated).
        groupby  (str, optional): Categorical column to overlay distributions.

    Actions:
        Plots overlapping semi-transparent histograms per group (or single),
        saves via _save_plot.

    Output:
        str — path to the saved PNG.
    """
    print(f"\n[OP] plot_hist(x='{x}', groupby={groupby})")
    print(f"  ↳ Why: A histogram reveals distribution shape: is it normal, right-skewed,")
    print(f"         bimodal? Each shape implies different statistical tests.")

    fig, ax = plt.subplots(figsize=(8, 5))

    if groupby and groupby in df.columns:
        for group, gdf in df.groupby(groupby):
            gdf[x].dropna().plot.hist(ax=ax, alpha=0.55, bins=20, label=str(group))
        ax.legend(title=groupby)
    else:
        df[x].dropna().plot.hist(ax=ax, bins=20, color="steelblue", edgecolor="white")

    ax.set_title(f"Distribution of {x}" + (f" by {groupby}" if groupby else ""))
    ax.set_xlabel(x)
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    name = f"hist_{x}" + (f"_by_{groupby}" if groupby else "")
    return _save_plot(fig, name, phase, registry)


def plot_box(df: pd.DataFrame, x: str, y: str, phase: str, registry: list) -> str:
    """
    Produce a box plot of a numeric variable (y) grouped by a categorical variable (x).

    Inputs:
        df       (pd.DataFrame): Dataset.
        x        (str):          Categorical column (groups on x-axis).
        y        (str):          Numeric column (values on y-axis).
        phase    (str):          Current phase label.
        registry (list):         Visual registry list (mutated).

    Actions:
        Creates a seaborn boxplot, saves via _save_plot.

    Output:
        str — path to the saved PNG.
    """
    print(f"\n[OP] plot_box(x='{x}', y='{y}')")
    print(f"  ↳ Why: Box plots show median, IQR, and outliers per group.")
    print(f"         Large differences in medians suggest groups truly differ.")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y, ax=ax, palette="Set2")
    ax.set_title(f"{y} by {x}")
    fig.tight_layout()

    return _save_plot(fig, f"box_{y}_by_{x}", phase, registry)


def plot_scatter(df: pd.DataFrame, x: str, y: str, phase: str, registry: list,
                 hue: str = None) -> str:
    """
    Produce a scatter plot between two numeric variables, optionally colored by category.

    Inputs:
        df       (pd.DataFrame): Dataset.
        x        (str):          Numeric column for the x-axis.
        y        (str):          Numeric column for the y-axis.
        phase    (str):          Current phase label.
        registry (list):         Visual registry list (mutated).
        hue      (str, optional): Categorical column for color encoding.

    Actions:
        Creates a seaborn scatterplot, saves via _save_plot.

    Output:
        str — path to the saved PNG.
    """
    print(f"\n[OP] plot_scatter(x='{x}', y='{y}', hue={hue})")
    print(f"  ↳ Why: Scatter plots show the shape of a bivariate relationship.")
    print(f"         Are the points in a line? A curve? Clustered by color?")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, palette="Set2", alpha=0.7)
    ax.set_title(f"{x} vs {y}" + (f"  (color: {hue})" if hue else ""))
    fig.tight_layout()

    name = f"scatter_{x}_vs_{y}" + (f"_hue_{hue}" if hue else "")
    return _save_plot(fig, name, phase, registry)


def plot_heatmap_corr(df: pd.DataFrame, method: str, phase: str,
                      registry: list) -> str:
    """
    Produce a color-coded heatmap of the full correlation matrix.

    Inputs:
        df       (pd.DataFrame): Dataset.
        method   (str):          Correlation method ('pearson', 'spearman', 'kendall').
        phase    (str):          Current phase label.
        registry (list):         Visual registry list (mutated).

    Actions:
        Computes corr matrix, creates annotated seaborn heatmap, saves via _save_plot.

    Output:
        str — path to the saved PNG.
    """
    print(f"\n[OP] plot_heatmap_corr(method='{method}')")
    print(f"  ↳ Why: A heatmap gives a simultaneous view of ALL pairwise correlations.")
    print(f"         Red = strong positive, Blue = strong negative. Use it to prioritize")
    print(f"         which pairs deserve a scatter plot.")

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title(f"Correlation Heatmap — {method.capitalize()}")
    fig.tight_layout()

    return _save_plot(fig, f"heatmap_corr_{method}", phase, registry)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_observe(df: pd.DataFrame, phase: str):
    """
    Execute all OBSERVE-phase operations sequentially.
    Goal: characterize the raw dataset without computing any statistics yet.

    Inputs:
        df    (pd.DataFrame): The loaded dataset.
        phase (str):          Phase label ('OBSERVE').

    Actions:
        Calls profile_dataset → infer_schema → missing_report → duplicates_report.
        All artifacts are written to ARTIFACTS_DIR.

    Output:
        dict — combined results from all four operations.
    """
    print("\n" + "=" * 62)
    print("  PHASE: OBSERVE")
    print("  Goal: Understand the raw structure before touching any numbers.")
    print("=" * 62)

    results = {}
    results["profile"]    = profile_dataset(df, phase)
    results["schema"]     = infer_schema(df, phase)
    results["missing"]    = missing_report(df, phase)
    results["duplicates"] = duplicates_report(df, phase)

    print("\n[OBSERVE COMPLETE]")
    print("  ▶ Copy the artifacts content into the {ARTIFACTS_CONTENT} field of")
    print("    the agent prompt and set phase = OBSERVE to get the agent's plan.")
    return results


def run_describe(df: pd.DataFrame, phase: str):
    """
    Execute all DESCRIBE-phase operations: summaries + an adaptive set of plots.
    Plots are chosen based on the schema (not hardcoded) to avoid selection bias.

    Selection rules:
      - Count plots:  up to 3 low-cardinality (≤10 unique) categorical columns.
      - Histograms:   up to 3 numeric columns with the highest variance.
      - Box plots:    up to 2 numeric columns vs the first low-cardinality categorical.
      - Scatter:      top-2 most correlated numeric pairs (Pearson absolute value).
      - Heatmap:      always generated if ≥2 numeric columns exist.
    Total cap: ≤10 plots.

    Inputs:
        df    (pd.DataFrame): The loaded dataset.
        phase (str):          Phase label ('DESCRIBE').

    Actions:
        Computes numeric_summary, categorical_summary, two correlation matrices,
        generates adaptive plots, writes 04_descriptive_stats.json and
        05_visual_registry.json.

    Output:
        dict — combined stats and visual registry.
    """
    print("\n" + "=" * 62)
    print("  PHASE: DESCRIBE")
    print("  Goal: Compute statistics and generate exploratory visualizations.")
    print("=" * 62)

    np.random.seed(RANDOM_SEED)
    visual_registry = []
    results         = {}

    # ── Summaries ──────────────────────────────────────────────────────────────
    results["numeric_summary"]    = numeric_summary(df, phase)
    results["categorical_summary"] = categorical_summary(df, phase)
    results["correlation_pearson"]  = correlation_matrix(df, "pearson",  phase)
    results["correlation_spearman"] = correlation_matrix(df, "spearman", phase)
    results["correlation_kendall"]  = correlation_matrix(df, "kendall",  phase)

    # ── Identify column sets ───────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    low_card_cat = [c for c in df.select_dtypes(include=["str", "category"]).columns
                    if df[c].nunique() <= 10]

    plot_count_used   = 0
    plot_hist_used    = 0
    plot_box_used     = 0
    plot_scatter_used = 0

    # Count plots (≤3)
    for col in low_card_cat:
        if plot_count_used >= 3:
            break
        plot_count(df, col, phase, visual_registry)
        plot_count_used += 1

    # Histograms for most-variable numeric cols (≤3)
    if numeric_cols:
        variances   = df[numeric_cols].var().sort_values(ascending=False)
        top_numeric = variances.index[:3].tolist()
        groupby_col = low_card_cat[0] if low_card_cat else None
        for col in top_numeric:
            if plot_hist_used >= 3:
                break
            plot_hist(df, col, phase, visual_registry, groupby=groupby_col)
            plot_hist_used += 1

    # Box plots (≤2 numerics vs first categorical)
    if numeric_cols and low_card_cat:
        cat_col = low_card_cat[0]
        for num_col in numeric_cols[:2]:
            if plot_box_used >= 2:
                break
            plot_box(df, cat_col, num_col, phase, visual_registry)
            plot_box_used += 1

    # Scatter for top-correlated pairs (≤2)
    if len(numeric_cols) >= 2:
        corr_abs_df   = df[numeric_cols].corr(method="pearson").abs()
        corr_abs_arr  = corr_abs_df.to_numpy().copy()
        np.fill_diagonal(corr_abs_arr, 0)
        corr_abs = pd.DataFrame(corr_abs_arr,
                                index=corr_abs_df.index,
                                columns=corr_abs_df.columns)

        seen_pairs = set()
        hue_col    = low_card_cat[0] if low_card_cat else None

        for col in corr_abs.columns:
            if plot_scatter_used >= 2:
                break
            partner = corr_abs[col].idxmax()
            pair    = tuple(sorted([col, partner]))
            if pair not in seen_pairs:
                plot_scatter(df, col, partner, phase, visual_registry, hue=hue_col)
                seen_pairs.add(pair)
                plot_scatter_used += 1

    # Heatmap (always if ≥2 numeric cols)
    if len(numeric_cols) >= 2:
        plot_heatmap_corr(df, "pearson", phase, visual_registry)

    # ── Write consolidated artifacts ───────────────────────────────────────────
    write_artifact("05_visual_registry.json", {"plots": visual_registry}, phase)
    write_artifact("04_descriptive_stats.json", safe_json(results), phase)

    print("\n[DESCRIBE COMPLETE]")
    print("  ▶ Copy artifacts/04_* and artifacts/05_* into the {ARTIFACTS_CONTENT}")
    print("    field of the agent prompt and set phase = DESCRIBE.")
    return results


def run_hypothesize(df: pd.DataFrame, phase: str, ops: list):
    """
    Execute only the specific operations requested by the agent in Phase 3.
    Unlike run_describe(), this does NOT run the full pipeline — only the
    operations listed in --ops are executed.

    Supported op strings:
      crosstab(col_a, col_b)
      correlation_matrix(method)
      plot_count(x)  |  plot_count(x, hue)
      plot_hist(x)   |  plot_hist(x, groupby)
      plot_box(x, y)
      plot_scatter(x, y)  |  plot_scatter(x, y, hue)
      plot_heatmap_corr(method)

    Inputs:
        df    (pd.DataFrame): The loaded dataset.
        phase (str):          Phase label ('HYPOTHESIZE').
        ops   (list[str]):    Op strings from --ops, e.g.
                              ['crosstab(species,island)', 'plot_box(sex,body_mass_g)']

    Output:
        None (artifacts written to disk; visual registry written if any plots produced).
    """
    import re

    print("\n" + "=" * 62)
    print("  PHASE: HYPOTHESIZE")
    print("  Goal: Run targeted operations requested by the agent.")
    print("=" * 62)

    visual_registry = []

    for op_str in ops:
        match = re.match(r"(\w+)\((.*)\)$", op_str.strip())
        if not match:
            print(f"  [SKIP] Cannot parse op: '{op_str}'")
            continue

        op_name  = match.group(1)
        raw_args = [a.strip() for a in match.group(2).split(",") if a.strip()]

        if op_name == "crosstab" and len(raw_args) == 2:
            crosstab(df, raw_args[0], raw_args[1], phase)

        elif op_name == "correlation_matrix" and len(raw_args) == 1:
            correlation_matrix(df, raw_args[0], phase)

        elif op_name == "plot_count":
            hue = raw_args[1] if len(raw_args) >= 2 else None
            plot_count(df, raw_args[0], phase, visual_registry, hue=hue)

        elif op_name == "plot_hist":
            groupby = raw_args[1] if len(raw_args) >= 2 else None
            plot_hist(df, raw_args[0], phase, visual_registry, groupby=groupby)

        elif op_name == "plot_box" and len(raw_args) == 2:
            plot_box(df, raw_args[0], raw_args[1], phase, visual_registry)

        elif op_name == "plot_scatter" and len(raw_args) >= 2:
            hue = raw_args[2] if len(raw_args) >= 3 else None
            plot_scatter(df, raw_args[0], raw_args[1], phase, visual_registry, hue=hue)

        elif op_name == "plot_heatmap_corr" and len(raw_args) == 1:
            plot_heatmap_corr(df, raw_args[0], phase, visual_registry)

        else:
            print(f"  [SKIP] Unknown op or wrong number of args: '{op_str}'")

    if visual_registry:
        write_artifact("05_visual_registry_hypothesize.json",
                       {"plots": visual_registry}, phase)

    print("\n[HYPOTHESIZE COMPLETE]")
    print("  ▶ Paste the new artifacts into the agent prompt to proceed.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible Data Observatory Runner — Moscoso_80 | 2026-1\n"
            "Each lab writes to its own isolated folder under:\n"
            "  semestre2026-1/Estudiantes/Moscoso_80/Laboratorios/{lab}/"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--lab",
        required=True,
        help=(
            "Lab folder name. Use a consistent, descriptive name.\n"
            "Examples: Lab_01_pinguinos | Lab_02_pandas | Lab_03_regresion"
        )
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        dest="dataset_label",
        help=(
            "Optional: isolate artifacts into a sub-folder when a lab uses\n"
            "multiple datasets. Omit for single-dataset labs.\n"
            "Examples: eurusd | breast_cancer | titanic"
        )
    )
    parser.add_argument(
        "--phase",
        choices=["OBSERVE", "DESCRIBE", "HYPOTHESIZE"],
        required=True,
        help=(
            "Execution phase:\n"
            "  OBSERVE     — raw profiling only\n"
            "  DESCRIBE    — stats + plots\n"
            "  HYPOTHESIZE — targeted extra operations requested by the agent\n"
            "                (requires --ops)"
        )
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=None,
        dest="ops",
        help=(
            "Specific operations for HYPOTHESIZE phase. Examples:\n"
            "  --ops crosstab(species,island)\n"
            "  --ops crosstab(species,sex) plot_box(sex,body_mass_g)\n"
            "  --ops plot_scatter(bill_depth_mm,bill_length_mm,species)"
        )
    )
    parser.add_argument(
        "--origin",
        choices=["seaborn", "csv", "url"],
        required=True,
        help=(
            "Dataset source:\n"
            "  seaborn — built-in seaborn dataset (e.g. penguins, titanic)\n"
            "  csv     — local .csv file path\n"
            "  url     — remote CSV URL (e.g. raw GitHub, UCI repository)"
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Seaborn name, local CSV path, or full https:// URL"
    )

    args = parser.parse_args()

    # Set module-level path globals BEFORE any other function is called.
    _resolve_paths(args.lab, args.dataset_label)

    # Print the resolved paths so the student can verify output location.
    label_info = f" / dataset: {args.dataset_label}" if args.dataset_label else ""
    print(f"\n[CONFIG] Lab: {args.lab}{label_info}")
    print(f"[CONFIG] Artifacts → {ARTIFACTS_DIR}")
    print(f"[CONFIG] Plots     → {PLOTS_DIR}")

    setup_directories()
    df = load_dataset(args.origin, args.dataset)

    if args.phase == "OBSERVE":
        run_observe(df, args.phase)
    elif args.phase == "DESCRIBE":
        run_describe(df, args.phase)
    elif args.phase == "HYPOTHESIZE":
        if not args.ops:
            print("[ERROR] --phase HYPOTHESIZE requires --ops. Example:")
            print("  --ops crosstab(species,island) plot_box(sex,body_mass_g)")
            sys.exit(1)
        run_hypothesize(df, args.phase, args.ops)


if __name__ == "__main__":
    main()