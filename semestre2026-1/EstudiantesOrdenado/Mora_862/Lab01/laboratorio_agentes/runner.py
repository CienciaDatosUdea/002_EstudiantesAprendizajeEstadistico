"""Runner Generalizado Versión 2.0 - Fase 2 Agentes."""
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def ensure_dirs():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def write_json(filename, payload, source):
    path = ARTIFACTS_DIR / filename
    data = {
        "metadata": {"generated_at": datetime.utcnow().isoformat(), "source": source},
        "payload": payload
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float64, np.floating)) else int(o) if isinstance(o, (np.int64, np.integer)) else str(o))
    print(f" Artifact: {filename}")

# --- FASE: OBSERVE ---
def run_observe(df, source):
    payload = {
        "summary": {"rows": len(df), "cols": len(df.columns)},
        "missing": df.isna().sum().to_dict(),
        "columns": {"numeric": df.select_dtypes(include="number").columns.tolist(),
                    "categorical": df.select_dtypes(exclude="number").columns.tolist()}
    }
    write_json("00_raw_profile.json", payload, source)

# --- FASE: DESCRIBE ---
def run_describe(df, source):
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(exclude="number")
    
    payload = {
        "numeric": num.describe().to_dict(),
        "categorical": {c: df[c].value_counts(normalize=True).to_dict() for c in cat.columns},
        "correlations": num.corr().to_dict()
    }
    write_json("04_descriptive_stats.json", payload, source)

    # Visuales Generalizados
    plots = []
    if not num.empty:
        plt.figure(figsize=(8,6)); sns.heatmap(num.corr(), annot=True, cmap='coolwarm')
        p = "05_correlation_heatmap.png"; plt.savefig(ARTIFACTS_DIR / p); plt.close(); plots.append(p)
    
    if len(cat.columns) > 0 and len(num.columns) > 0:
        plt.figure(figsize=(8,6)); sns.boxplot(data=df, x=cat.columns[0], y=num.columns[0])
        p = "05_boxplot_exploratory.png"; plt.savefig(ARTIFACTS_DIR / p); plt.close(); plots.append(p)
    
    write_json("05_visual_registry.json", {"plots": plots}, source)

# --- FASE: HYPOTHESIZE & CONCLUDE (Versión Completa de 3 Hipótesis) ---
def run_hypothesize(df, source):
    df_c = df.dropna()
    num_cols = df_c.select_dtypes(include="number").columns.tolist()
    cat_cols = df_c.select_dtypes(exclude="number").columns.tolist()

    test_results = {}
    hypotheses = []
    conclusions = []

    # --- HIPÓTESIS 1: Diferencia de grupos (ANOVA) ---
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        c1, n1 = cat_cols[0], num_cols[0]
        groups = [g[n1].values for _, g in df_c.groupby(c1)]
        f, p = stats.f_oneway(*groups)
        
        test_results["anova"] = {"var_cat": c1, "var_num": n1, "f": f, "p": p}
        hypotheses.append({"id": "H1", "statement": f"El valor de {n1} varía según la categoría {c1}."})
        conclusions.append({
            "hypothesis_id": "H1",
            "supported": bool(p < 0.05),
            "evidence": f"Prueba ANOVA: F={f:.2f}, p-valor={p:.4f}. Se rechaza la hipótesis nula si p < 0.05."
        })

    # --- HIPÓTESIS 2: Relación entre medidas (Pearson) ---
    if len(num_cols) >= 2:
        n1, n2 = num_cols[0], num_cols[1]
        r, p_r = stats.pearsonr(df_c[n1], df_c[n2])
        
        test_results["pearson"] = {"vars": [n1, n2], "r": r, "p": p_r}
        hypotheses.append({"id": "H2", "statement": f"Existe una asociación lineal entre {n1} y {n2}."})
        conclusions.append({
            "hypothesis_id": "H2",
            "supported": bool(p_r < 0.05),
            "evidence": f"Correlación de Pearson: r={r:.2f}, p-valor={p_r:.4f}. Asociación significativa detectada."
        })

    # --- HIPÓTESIS 3: Asociación entre categorías (Chi-cuadrado) ---
    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[0], cat_cols[1]
        contingency = pd.crosstab(df_c[c1], df_c[c2])
        chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
        
        test_results["chi_square"] = {"vars": [c1, c2], "chi2": chi2, "p": p_chi}
        hypotheses.append({"id": "H3", "statement": f"La variable {c1} está asociada con la variable {c2}."})
        conclusions.append({
            "hypothesis_id": "H3",
            "supported": bool(p_chi < 0.05),
            "evidence": f"Prueba Chi-cuadrado: estadístico={chi2:.2f}, p-valor={p_chi:.4f}."
        })

    # Guardar todos los artifacts requeridos por la guía
    write_json("08_tests.json", test_results, source)
    write_json("06_hypotheses_log.json", {"hypotheses": hypotheses}, source)
    write_json("07_conclusions.json", {"conclusions": conclusions}, source)
    
    questions = [
        {"prompt": "¿Existen sesgos en la recolección de datos que expliquen las asociaciones encontradas?", "ref": "07_conclusions.json"},
        {"prompt": "¿Cómo afecta el desbalance de categorías observado en 04_descriptive_stats a las pruebas H1 y H3?", "ref": "04_descriptive_stats.json"}
    ]
    write_json("09_questions.json", {"questions": questions}, source)
    print(" Todos los artifacts de la Fase 3 generados (06, 07, 08, 09)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--phase", choices=["observe", "describe", "hypothesize"], required=True)
    args = parser.parse_args()
    
    ensure_dirs()
    df = pd.read_csv(args.file)
    if args.phase == "observe": run_observe(df, args.file)
    elif args.phase == "describe": run_describe(df, args.file)
    elif args.phase == "hypothesize": run_hypothesize(df, args.file)

if __name__ == "__main__":
    main()