# -*- coding: utf-8 -*-
"""
Runner - Fase 2 del laboratorio (arquitectura Agente/Runner).
El agente (chat) propone que funcion llamar e interpreta los artifacts.
Este script es el unico que calcula y escribe artifacts en JSON/PNG.
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sstats

ART_DIR = "artifacts"


def cargar_datos(path):
    os.makedirs(ART_DIR, exist_ok=True)
    return pd.read_csv(path)


def guardar_artifact(nombre, data):
    ruta = os.path.join(ART_DIR, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"Artifact guardado: {ruta}")


# ---------- FASE OBSERVE ----------

def perfil_crudo(df):
    return {
        "n_filas": df.shape[0],
        "n_columnas": df.shape[1],
        "columnas": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulos_por_columna": df.isnull().sum().to_dict(),
        "duplicados": int(df.duplicated().sum()),
        "cardinalidad": df.nunique().to_dict(),
    }


def correr_observe(df):
    perfil = perfil_crudo(df)
    guardar_artifact("00_raw_profile.json", perfil)
    return perfil


# ---------- FASE DESCRIBE ----------

def guardar_png(nombre_archivo):
    ruta = os.path.join(ART_DIR, nombre_archivo)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    return ruta


def stats_numericas(df, columnas=None):
    if columnas is None:
        num = df.select_dtypes(include=["number"])
    else:
        num = df[columnas]

    return {
        "describe": num.describe().round(3).to_dict(),
        "missing": num.isnull().sum().to_dict(),
        "pearson": num.corr(method="pearson").round(3).to_dict(),
        "spearman": num.corr(method="spearman").round(3).to_dict(),
    }


def stats_categoricas(df, columnas):
    out = {}
    for col in columnas:
        vc = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(normalize=True, dropna=False) * 100
        out[col] = {
            "conteos": vc.to_dict(),
            "porcentajes": pct.round(2).to_dict(),
            "nulos": int(df[col].isnull().sum()),
        }
    return out


def tablas_cruzadas(df, pares):
    out = {}
    for c1, c2 in pares:
        tabla = pd.crosstab(df[c1], df[c2])
        tabla_pct = pd.crosstab(df[c1], df[c2], normalize="index").round(3)
        out[f"{c1}_x_{c2}"] = {
            "conteos": tabla.to_dict(),
            "porcentajes_fila": tabla_pct.to_dict(),
        }
    return out


def graficos_categoricas(df, columnas):
    registry = {}
    for col in columnas:
        filename = f"05_{col}_bar.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        df[col].value_counts(dropna=False).plot(kind="bar", color="steelblue", edgecolor="black", ax=ax)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Conteo")
        registry[f"{col}_bar"] = filename
        guardar_png(filename)
    return registry


def graficos_numericas(df, columnas):
    registry = {}
    for col in columnas:
        filename = f"05_{col}_hist.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        df[col].dropna().hist(bins=20, color="darkorange", edgecolor="black", ax=ax)
        ax.set_title(f"Histograma de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        registry[f"{col}_hist"] = filename
        guardar_png(filename)
    return registry


def boxplot_body_mass_by_species(df):
    filename = "05_body_mass_g_by_species_boxplot.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="species", y="body_mass_g", palette="Set2", ax=ax)
    ax.set_title("body_mass_g por species")
    ax.set_xlabel("species")
    ax.set_ylabel("body_mass_g")
    guardar_png(filename)
    return {"body_mass_g_by_species_boxplot": filename}


def boxplot_bill_length_by_island(df):
    filename = "05_bill_length_mm_by_island_boxplot.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="island", y="bill_length_mm", palette="Set3", ax=ax)
    ax.set_title("bill_length_mm por island")
    ax.set_xlabel("island")
    ax.set_ylabel("bill_length_mm")
    guardar_png(filename)
    return {"bill_length_mm_by_island_boxplot": filename}


def scatter_bill_length_vs_flipper_by_species(df):
    filename = "05_bill_length_mm_vs_flipper_length_mm_by_species_scatter.png"
    sub = df[["bill_length_mm", "flipper_length_mm", "species"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=sub,
        x="bill_length_mm",
        y="flipper_length_mm",
        hue="species",
        palette="deep",
        s=50,
        ax=ax,
    )
    ax.set_title("bill_length_mm vs flipper_length_mm coloreado por species")
    ax.set_xlabel("bill_length_mm")
    ax.set_ylabel("flipper_length_mm")
    ax.legend(title="species")
    guardar_png(filename)
    return {"bill_length_mm_vs_flipper_length_mm_by_species_scatter": filename}


def heatmap_correlacion(df, columnas):
    filename = "05_numeric_correlation_heatmap.png"
    corr = df[columnas].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlación (Pearson)")
    guardar_png(filename)
    return {"numeric_correlation_heatmap": filename}


def correr_describe(df, cat_cols, num_cols, pares_cruzados):
    stats = {
        "numericas": stats_numericas(df, num_cols),
        "categoricas": stats_categoricas(df, cat_cols),
        "cruzadas": tablas_cruzadas(df, pares_cruzados),
    }
    guardar_artifact("04_descriptive_stats.json", stats)

    visual_registry = {}
    visual_registry.update(graficos_categoricas(df, cat_cols))
    visual_registry.update(graficos_numericas(df, num_cols))
    visual_registry.update(boxplot_body_mass_by_species(df))
    visual_registry.update(boxplot_bill_length_by_island(df))
    visual_registry.update(scatter_bill_length_vs_flipper_by_species(df))
    visual_registry.update(heatmap_correlacion(df, num_cols))
    guardar_artifact("05_visual_registry.json", visual_registry)
    return {"stats": stats, "registro_visual": visual_registry}


# ---------- FASE HYPOTHESIZE_AND_CONCLUDE ----------
# Estas funciones solo calculan. El texto de hipotesis/conclusiones lo
# redacta el agente en el chat, a partir de lo que estas funciones guardan.

def prueba_pearson(df, col1, col2):
    sub = df[[col1, col2]].dropna()
    r, p = sstats.pearsonr(sub[col1], sub[col2])
    return {"tipo": "pearson", "col1": col1, "col2": col2, "r": round(r, 3), "p_valor": p}


def prueba_anova(df, col_grupo, col_valor):
    sub = df[[col_grupo, col_valor]].dropna()
    grupos = [g[col_valor].values for _, g in sub.groupby(col_grupo)]
    f, p = sstats.f_oneway(*grupos)
    return {"tipo": "anova", "col_grupo": col_grupo, "col_valor": col_valor, "f": round(f, 3), "p_valor": p}


def prueba_chi2(df, col1, col2):
    tabla = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, _ = sstats.chi2_contingency(tabla)
    return {"tipo": "chi2", "col1": col1, "col2": col2, "chi2": round(chi2, 3), "p_valor": p, "dof": dof}


def correr_tests(resultados):
    guardar_artifact("08_tests.json", resultados)
    return resultados


if __name__ == "__main__":
    df = cargar_datos("penguins - penguins.csv")
    correr_observe(df)
    correr_describe(
        df,
        ["species", "island", "sex"],
        ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
        [("species", "island"), ("species", "sex"), ("island", "sex")],
    )

    resultados = [
        prueba_anova(df, "species", "body_mass_g"),
        prueba_chi2(df, "species", "island"),
        prueba_pearson(df, "bill_length_mm", "flipper_length_mm"),
    ]
    correr_tests(resultados)