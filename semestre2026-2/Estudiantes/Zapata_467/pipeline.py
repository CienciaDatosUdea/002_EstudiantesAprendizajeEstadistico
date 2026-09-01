import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    os.makedirs("artifacts", exist_ok=True)

    df = sns.load_dataset("penguins")

    numeric_vars = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_vars = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_cardinality_vars = [col for col in categorical_vars if df[col].nunique(dropna=False) < 4]

    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>Observatorio</title></head><body>")
    parts.append("<h1>Reporte estadístico</h1>")

    num_rows, num_columns = df.shape
    parts.append(f"<p>El dataset tiene {num_rows} filas y {num_columns} columnas.</p>")
    parts.append(f"<p>Variables numéricas: {', '.join(numeric_vars)}</p>")
    parts.append(f"<p>Variables categóricas: {', '.join(categorical_vars)}</p>")
    parts.append("<h2>Valores faltantes</h2>")
    parts.append(df.isnull().sum().to_frame("faltantes").to_html())

    estadisticas_numericas = pd.DataFrame({
        "Media": df[numeric_vars].mean(),
        "Mediana": df[numeric_vars].median(),
        "Desviación estándar": df[numeric_vars].std(),
        "Rango intercuartílico": df[numeric_vars].quantile(0.75) - df[numeric_vars].quantile(0.25),
    })
    parts.append("<h2>Estadísticas numéricas</h2>")
    parts.append(estadisticas_numericas.to_html())

    parts.append("<h2>Variables categóricas</h2>")
    for variable in categorical_vars:
        conteos = df[variable].value_counts(dropna=False)
        porcentajes = df[variable].value_counts(normalize=True, dropna=False).mul(100)
        resumen = pd.DataFrame({"Conteo": conteos, "Porcentaje (%)": porcentajes.round(2)})
        parts.append(f"<h3>{variable}</h3>")
        parts.append(resumen.to_html())

    parts.append("<h2>Tablas cruzadas</h2>")
    for a, b in combinations(categorical_vars, 2):
        parts.append(f"<h3>{a} x {b}</h3>")
        parts.append(pd.crosstab(df[a], df[b], dropna=False).to_html())

    parts.append("<h2>Correlaciones</h2>")
    parts.append(df[numeric_vars].corr(method="pearson").round(3).to_html())
    parts.append(df[numeric_vars].corr(method="spearman").round(3).to_html())

    fig, axes = plt.subplots(len(numeric_vars), 1, figsize=(10, 4 * len(numeric_vars)))
    axes = np.atleast_1d(axes)
    for eje, variable in zip(axes, numeric_vars):
        sns.histplot(data=df, x=variable, kde=True, bins=20, ax=eje)
        eje.set_title(f"Histograma de {variable}")
    plt.tight_layout()
    hist_path = os.path.join("artifacts", "histogramas.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    with open(hist_path, "rb") as f:
        hist_b64 = __import__("base64").b64encode(f.read()).decode("ascii")
    parts.append(f"<h2>Histogramas</h2><img src='data:image/png;base64,{hist_b64}' alt='Histogramas'>")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="species", y="bill_length_mm", hue="species", legend=False)
    box_path = os.path.join("artifacts", "boxplot.png")
    plt.tight_layout()
    plt.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close()
    with open(box_path, "rb") as f:
        box_b64 = __import__("base64").b64encode(f.read()).decode("ascii")
    parts.append(f"<h2>Boxplot</h2><img src='data:image/png;base64,{box_b64}' alt='Boxplot'>")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", s=70, alpha=0.8)
    scatter_path = os.path.join("artifacts", "scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    with open(scatter_path, "rb") as f:
        scatter_b64 = __import__("base64").b64encode(f.read()).decode("ascii")
    parts.append(f"<h2>Scatter</h2><img src='data:image/png;base64,{scatter_b64}' alt='Scatter'>")

    plt.figure(figsize=(10, 7))
    sns.heatmap(df[numeric_vars].corr(method="pearson"), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    heatmap_path = os.path.join("artifacts", "heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    with open(heatmap_path, "rb") as f:
        heatmap_b64 = __import__("base64").b64encode(f.read()).decode("ascii")
    parts.append(f"<h2>Heatmap</h2><img src='data:image/png;base64,{heatmap_b64}' alt='Heatmap'>")

    parts.append("</body></html>")

    with open("artifacts/observatorio.html", "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
