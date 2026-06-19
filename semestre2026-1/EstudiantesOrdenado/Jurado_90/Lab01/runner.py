"""
Runner — Observatorio de Datos Reproducible (Pingüinos)
=======================================================
Ejecuta operaciones de observación, descripción e hipótesis,
y escribe artifacts reproducibles en JSON/PNG.

Uso:
    python runner.py --phase OBSERVE
    python runner.py --phase DESCRIBE
    python runner.py --phase HYPOTHESIZE_AND_CONCLUDE
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def setup_directories():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def safe_json(obj):
    """Convierte objetos a tipos serializables JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def write_artifact(filename, data):
    path = ARTIFACTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_json(data), f, ensure_ascii=False, indent=2)
    print(f"  [artifact] {path}")


def load_dataset():
    try:
        df = sns.load_dataset("penguins")
        print(f"Dataset cargado desde seaborn: {df.shape}")
    except (ImportError, OSError, ValueError):
        url = (
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
        )
        df = pd.read_csv(url)
        print(f"Dataset cargado desde CSV remoto: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────
# OPERACIONES DEL CATÁLOGO
# ─────────────────────────────────────────────────────────


def profile_dataset(df):
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def infer_schema(df):
    schema = {}
    for c in df.columns:
        nunique = df[c].nunique()
        dtype = str(df[c].dtype)
        if dtype in ("object", "category"):
            role = "categorical"
        elif nunique <= 10:
            role = "categorical_low_cardinality"
        else:
            role = "numeric"
        schema[c] = {
            "dtype": dtype,
            "role": role,
            "n_unique": nunique,
            "sample_values": df[c].dropna().unique()[:5].tolist(),
        }
    return schema


def missing_report(df):
    report = {}
    for c in df.columns:
        n_miss = int(df[c].isna().sum())
        report[c] = {
            "n_missing": n_miss,
            "pct_missing": round(n_miss / len(df) * 100, 2),
        }
    return report


def duplicates_report(df):
    n_dup = int(df.duplicated().sum())
    return {
        "n_duplicates": n_dup,
        "pct_duplicates": round(n_dup / len(df) * 100, 2),
    }


def numeric_summary(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    summary = {}
    for c in num_cols:
        s = df[c].dropna()
        summary[c] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
        }
    return summary


def categorical_summary(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    summary = {}
    for c in cat_cols:
        counts = df[c].value_counts(dropna=False)
        summary[c] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "pct": {str(k): round(v / len(df) * 100, 2) for k, v in counts.items()},
        }
    return summary


def crosstab_op(df, a, b):
    ct = pd.crosstab(df[a], df[b])
    return {
        "variables": [a, b],
        "table": {
            str(idx): {str(col): int(ct.loc[idx, col]) for col in ct.columns}
            for idx in ct.index
        },
    }


def correlation_matrix(df, method="pearson"):
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)
    return {
        "method": method,
        "matrix": {
            c: {r: round(float(corr.loc[r, c]), 4) for r in corr.index}
            for c in corr.columns
        },
    }


# ─────────────────────────────────────────────────────────
# OPERACIONES DE PLOT
# ─────────────────────────────────────────────────────────


def _save_plot(fig, name):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(SCRIPT_DIR))


def plot_count(df, x, hue=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=x, hue=hue, ax=ax)
    ax.set_title(f"Conteo de {x}" + (f" por {hue}" if hue else ""))
    name = f"count_{x}" + (f"_by_{hue}" if hue else "")
    return _save_plot(fig, name), name


def plot_hist(df, x, groupby=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if groupby:
        for g, grp in df.groupby(groupby):
            ax.hist(grp[x].dropna(), alpha=0.5, label=str(g), bins=20)
        ax.legend()
    else:
        ax.hist(df[x].dropna(), bins=20, edgecolor="black", alpha=0.7)
    ax.set_title(f"Histograma de {x}" + (f" por {groupby}" if groupby else ""))
    ax.set_xlabel(x)
    ax.set_ylabel("Frecuencia")
    name = f"hist_{x}" + (f"_by_{groupby}" if groupby else "")
    return _save_plot(fig, name), name


def plot_box(df, x, y):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"Boxplot de {y} por {x}")
    name = f"box_{y}_by_{x}"
    return _save_plot(fig, name), name


def plot_scatter(df, x, y, hue=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, ax=ax)
    ax.set_title(f"Scatter {x} vs {y}" + (f" (color: {hue})" if hue else ""))
    name = f"scatter_{x}_vs_{y}" + (f"_hue_{hue}" if hue else "")
    return _save_plot(fig, name), name


def plot_heatmap_corr(df, method="pearson"):
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"Heatmap correlación ({method})")
    name = f"heatmap_corr_{method}"
    return _save_plot(fig, name), name


# ─────────────────────────────────────────────────────────
# FASES
# ─────────────────────────────────────────────────────────


def run_observe(df):
    print("\n=== FASE OBSERVE ===")
    profile = profile_dataset(df)
    schema = infer_schema(df)
    missing = missing_report(df)
    duplicates = duplicates_report(df)

    raw_profile = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "schema": schema,
        "missing": missing,
        "duplicates": duplicates,
    }
    write_artifact("00_raw_profile.json", raw_profile)
    return raw_profile


def run_describe(df):
    print("\n=== FASE DESCRIBE ===")
    schema = infer_schema(df)
    num_sum = numeric_summary(df)
    cat_sum = categorical_summary(df)

    cat_cols = [
        c
        for c, v in schema.items()
        if v["role"] in ("categorical", "categorical_low_cardinality")
    ]
    num_cols = [c for c, v in schema.items() if v["role"] == "numeric"]

    # Crosstabs para pares categóricos
    crosstabs = {}
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                key = f"{cat_cols[i]}_x_{cat_cols[j]}"
                crosstabs[key] = crosstab_op(df, cat_cols[i], cat_cols[j])

    # Correlaciones
    corr_pearson = correlation_matrix(df, "pearson")
    corr_spearman = correlation_matrix(df, "spearman")

    desc_stats = {
        "timestamp": datetime.now().isoformat(),
        "numeric_summary": num_sum,
        "categorical_summary": cat_sum,
        "crosstabs": crosstabs,
        "correlations": {"pearson": corr_pearson, "spearman": corr_spearman},
    }
    write_artifact("04_descriptive_stats.json", desc_stats)

    # Gráficos exploratorios
    visual_registry = []

    # 1) Conteos de categóricas
    for c in cat_cols:
        path, name = plot_count(df, c)
        visual_registry.append(
            {"type": "count", "variable": c, "path": path, "id": name}
        )

    # 2) Conteos con hue (species si existe)
    hue_col = "species" if "species" in cat_cols else None
    if hue_col:
        for c in cat_cols:
            if c != hue_col:
                path, name = plot_count(df, c, hue=hue_col)
                visual_registry.append(
                    {
                        "type": "count_hue",
                        "variable": c,
                        "hue": hue_col,
                        "path": path,
                        "id": name,
                    }
                )

    # 3) Histogramas de numéricas (top por varianza)
    variances = {c: float(df[c].var()) for c in num_cols if df[c].var() == df[c].var()}
    sorted_num = sorted(variances, key=variances.get, reverse=True)
    for c in sorted_num[:4]:
        path, name = plot_hist(df, c, groupby=hue_col)
        visual_registry.append(
            {
                "type": "histogram",
                "variable": c,
                "groupby": hue_col,
                "path": path,
                "id": name,
            }
        )

    # 4) Boxplots numérica por categórica
    if hue_col and len(sorted_num) >= 2:
        for c in sorted_num[:3]:
            path, name = plot_box(df, hue_col, c)
            visual_registry.append(
                {"type": "boxplot", "x": hue_col, "y": c, "path": path, "id": name}
            )

    # 5) Scatter entre pares de mayor correlación
    if len(num_cols) >= 2:
        path, name = plot_scatter(df, sorted_num[0], sorted_num[1], hue=hue_col)
        visual_registry.append(
            {
                "type": "scatter",
                "x": sorted_num[0],
                "y": sorted_num[1],
                "hue": hue_col,
                "path": path,
                "id": name,
            }
        )

    # 6) Heatmaps
    path, name = plot_heatmap_corr(df, "pearson")
    visual_registry.append(
        {"type": "heatmap", "method": "pearson", "path": path, "id": name}
    )
    path, name = plot_heatmap_corr(df, "spearman")
    visual_registry.append(
        {"type": "heatmap", "method": "spearman", "path": path, "id": name}
    )

    write_artifact(
        "05_visual_registry.json",
        {
            "timestamp": datetime.now().isoformat(),
            "plots": visual_registry,
        },
    )
    return desc_stats


def run_hypothesize(df):
    print("\n=== FASE HYPOTHESIZE_AND_CONCLUDE ===")

    # Verificar que existen artifacts previos
    for req in [
        "00_raw_profile.json",
        "04_descriptive_stats.json",
        "05_visual_registry.json",
    ]:
        if not (ARTIFACTS_DIR / req).exists():
            print(f"  [ERROR] Falta artifact: {req}. Ejecute fases anteriores primero.")
            sys.exit(1)

    with open(ARTIFACTS_DIR / "04_descriptive_stats.json", encoding="utf-8") as f:
        desc = json.load(f)

    num_cols = list(desc["numeric_summary"].keys())
    cat_cols = list(desc["categorical_summary"].keys())

    # Hipótesis basadas en correlaciones y distribuciones
    hypotheses = []

    # H1: correlación flipper_length vs body_mass
    if "flipper_length_mm" in num_cols and "body_mass_g" in num_cols:
        hypotheses.append(
            {
                "id": "H1",
                "origin": "ai_observer",
                "statement": "flipper_length_mm se asocia positivamente con body_mass_g",
                "variables": {"x": "flipper_length_mm", "y": "body_mass_g"},
                "suggested_tests": ["pearson_correlation", "spearman_correlation"],
                "evidence_refs": [
                    "04_descriptive_stats.json:correlations.pearson",
                    "05_visual_registry.json:scatter_body_mass_g_vs_flipper_length_mm",
                ],
                "decision_needed": "human",
            }
        )

    # H2: bill_length difiere por species
    if "bill_length_mm" in num_cols and "species" in cat_cols:
        hypotheses.append(
            {
                "id": "H2",
                "origin": "ai_observer",
                "statement": "bill_length_mm difiere significativamente entre species",
                "variables": {"x": "bill_length_mm", "group": "species"},
                "suggested_tests": ["kruskal_wallis", "one_way_anova"],
                "evidence_refs": [
                    "04_descriptive_stats.json:numeric_summary.bill_length_mm",
                    "05_visual_registry.json:box_bill_length_mm_by_species",
                ],
                "decision_needed": "human",
            }
        )

    # H3: species se asocia con island
    if "species" in cat_cols and "island" in cat_cols:
        hypotheses.append(
            {
                "id": "H3",
                "origin": "ai_observer",
                "statement": "La distribución de species se asocia con island",
                "variables": {"x": "species", "y": "island"},
                "suggested_tests": ["chi_squared"],
                "evidence_refs": [
                    "04_descriptive_stats.json:crosstabs.species_x_island",
                    "05_visual_registry.json:count_island_by_species",
                ],
                "decision_needed": "human",
            }
        )

    # H4: body_mass difiere por sex
    if "body_mass_g" in num_cols and "sex" in cat_cols:
        hypotheses.append(
            {
                "id": "H4",
                "origin": "ai_observer",
                "statement": "body_mass_g difiere entre sexos (Male vs Female)",
                "variables": {"x": "body_mass_g", "group": "sex"},
                "suggested_tests": ["mann_whitney_u", "t_test_independent"],
                "evidence_refs": [
                    "04_descriptive_stats.json:numeric_summary.body_mass_g",
                    "04_descriptive_stats.json:categorical_summary.sex",
                ],
                "decision_needed": "human",
            }
        )

    write_artifact(
        "06_hypotheses_log.json",
        {
            "timestamp": datetime.now().isoformat(),
            "hypotheses": hypotheses,
        },
    )

    # Ejecutar pruebas estadísticas
    tests_results = []

    # Test H1: Pearson y Spearman para flipper vs body_mass
    if "flipper_length_mm" in num_cols and "body_mass_g" in num_cols:
        clean = df[["flipper_length_mm", "body_mass_g"]].dropna()
        r_p, p_p = stats.pearsonr(clean["flipper_length_mm"], clean["body_mass_g"])
        r_s, p_s = stats.spearmanr(clean["flipper_length_mm"], clean["body_mass_g"])
        tests_results.append(
            {
                "hypothesis_id": "H1",
                "test": "pearson_correlation",
                "statistic": round(float(r_p), 4),
                "p_value": float(p_p),
                "supports_hypothesis": p_p < 0.05,
            }
        )
        tests_results.append(
            {
                "hypothesis_id": "H1",
                "test": "spearman_correlation",
                "statistic": round(float(r_s), 4),
                "p_value": float(p_s),
                "supports_hypothesis": p_s < 0.05,
            }
        )

    # Test H2: Kruskal-Wallis para bill_length por species
    if "bill_length_mm" in num_cols and "species" in cat_cols:
        groups = [g["bill_length_mm"].dropna().values for _, g in df.groupby("species")]
        stat_kw, p_kw = stats.kruskal(*groups)
        tests_results.append(
            {
                "hypothesis_id": "H2",
                "test": "kruskal_wallis",
                "statistic": round(float(stat_kw), 4),
                "p_value": float(p_kw),
                "supports_hypothesis": p_kw < 0.05,
            }
        )

    # Test H3: Chi-cuadrado species x island
    if "species" in cat_cols and "island" in cat_cols:
        ct = pd.crosstab(df["species"], df["island"])
        chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
        tests_results.append(
            {
                "hypothesis_id": "H3",
                "test": "chi_squared",
                "statistic": round(float(chi2), 4),
                "p_value": float(p_chi),
                "dof": int(dof),
                "supports_hypothesis": p_chi < 0.05,
            }
        )

    # Test H4: Mann-Whitney U para body_mass por sex
    if "body_mass_g" in num_cols and "sex" in cat_cols:
        male = df[df["sex"] == "Male"]["body_mass_g"].dropna()
        female = df[df["sex"] == "Female"]["body_mass_g"].dropna()
        if len(male) > 0 and len(female) > 0:
            stat_mw, p_mw = stats.mannwhitneyu(male, female, alternative="two-sided")
            tests_results.append(
                {
                    "hypothesis_id": "H4",
                    "test": "mann_whitney_u",
                    "statistic": round(float(stat_mw), 4),
                    "p_value": float(p_mw),
                    "supports_hypothesis": p_mw < 0.05,
                }
            )

    write_artifact(
        "08_tests.json",
        {
            "timestamp": datetime.now().isoformat(),
            "tests": tests_results,
        },
    )

    # Conclusiones
    conclusions = {
        "timestamp": datetime.now().isoformat(),
        "descriptive_findings": [
            {
                "finding": "El dataset contiene 344 observaciones de 3 especies de pingüinos con 7 variables",
                "evidence_refs": [
                    "00_raw_profile.json:profile.n_rows",
                    "00_raw_profile.json:profile.n_cols",
                ],
            },
            {
                "finding": "Existen valores faltantes concentrados en sex y variables de medición corporal",
                "evidence_refs": ["00_raw_profile.json:missing"],
            },
            {
                "finding": "Las variables numéricas presentan rangos y dispersiones diferentes según especie",
                "evidence_refs": ["04_descriptive_stats.json:numeric_summary"],
            },
        ],
        "visual_patterns": [
            {
                "pattern": "Los histogramas de bill_length_mm muestran distribuciones diferenciadas por especie",
                "evidence_refs": [
                    "05_visual_registry.json:hist_bill_length_mm_by_species"
                ],
            },
            {
                "pattern": "El scatter de body_mass_g vs flipper_length_mm muestra agrupaciones por especie",
                "evidence_refs": [
                    "05_visual_registry.json:scatter_body_mass_g_vs_flipper_length_mm_hue_species"
                ],
            },
        ],
        "next_hypotheses": [
            "Explorar si bill_depth_mm también difiere por species controlando por sex",
            "Analizar si la relación flipper_length-body_mass varía por island",
        ],
    }
    write_artifact("07_conclusions.json", conclusions)

    # Preguntas para el humano
    questions = {
        "timestamp": datetime.now().isoformat(),
        "questions": [
            "¿Cómo deben tratarse los valores faltantes en sex? ¿Eliminar filas o imputar?",
            "¿Se debería controlar el análisis por species al comparar sexos?",
            "¿Existe un factor temporal (año) que pueda afectar las mediciones?",
        ],
    }
    write_artifact("09_questions.json", questions)

    # Reporte markdown
    report = _generate_report()
    report_path = ARTIFACTS_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [artifact] {report_path}")

    return conclusions


def _generate_report():
    sections = []
    sections.append("# Observatorio de Datos — Pingüinos\n")
    sections.append(f"Generado: {datetime.now().isoformat()}\n")

    # Cargar artifacts
    for fname, title in [
        ("00_raw_profile.json", "## Observación inicial"),
        ("04_descriptive_stats.json", "## Descripción"),
        ("06_hypotheses_log.json", "## Hipótesis"),
        ("08_tests.json", "## Pruebas estadísticas"),
        ("07_conclusions.json", "## Conclusiones"),
    ]:
        fpath = ARTIFACTS_DIR / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            sections.append(f"\n{title}\n")
            if fname == "00_raw_profile.json":
                p = data.get("profile", {})
                sections.append(f"- Filas: {p.get('n_rows')}")
                sections.append(f"- Columnas: {p.get('n_cols')}")
                sections.append(f"- Variables: {', '.join(p.get('columns', []))}")
                m = data.get("missing", {})
                sections.append("\nValores faltantes:")
                for col, info in m.items():
                    if info["n_missing"] > 0:
                        sections.append(
                            f"  - {col}: {info['n_missing']} ({info['pct_missing']}%)"
                        )
            elif fname == "08_tests.json":
                for t in data.get("tests", []):
                    sections.append(
                        f"- **{t['hypothesis_id']}** ({t['test']}): "
                        f"stat={t['statistic']}, p={t['p_value']:.2e} "
                        f"→ {'Apoya' if t['supports_hypothesis'] else 'No apoya'} hipótesis"
                    )
            elif fname == "06_hypotheses_log.json":
                for h in data.get("hypotheses", []):
                    sections.append(f"- **{h['id']}**: {h['statement']}")
                    sections.append(
                        f"  - Tests sugeridos: {', '.join(h['suggested_tests'])}"
                    )
            elif fname == "07_conclusions.json":
                sections.append("\n### Hallazgos descriptivos")
                for f_ in data.get("descriptive_findings", []):
                    sections.append(
                        f"- {f_['finding']} (refs: {', '.join(f_['evidence_refs'])})"
                    )
                sections.append("\n### Patrones visuales")
                for p_ in data.get("visual_patterns", []):
                    sections.append(
                        f"- {p_['pattern']} (refs: {', '.join(p_['evidence_refs'])})"
                    )

    # Gráficos
    vr_path = ARTIFACTS_DIR / "05_visual_registry.json"
    if vr_path.exists():
        with open(vr_path, encoding="utf-8") as f:
            vr = json.load(f)
        sections.append("\n## Gráficos generados\n")
        for plot in vr.get("plots", []):
            sections.append(f"- [{plot['id']}]({plot['path']})")

    return "\n".join(sections)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Runner del Observatorio de Datos")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["OBSERVE", "DESCRIBE", "HYPOTHESIZE_AND_CONCLUDE", "ALL"],
        help="Fase a ejecutar",
    )
    args = parser.parse_args()

    setup_directories()
    df = load_dataset()

    if args.phase == "OBSERVE":
        run_observe(df)
    elif args.phase == "DESCRIBE":
        run_describe(df)
    elif args.phase == "HYPOTHESIZE_AND_CONCLUDE":
        run_hypothesize(df)
    elif args.phase == "ALL":
        run_observe(df)
        run_describe(df)
        run_hypothesize(df)

    print("\n✓ Fase completada.")


if __name__ == "__main__":
    main()
