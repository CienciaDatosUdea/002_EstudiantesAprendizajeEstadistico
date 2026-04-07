import pandas as pd
import seaborn as sns
import json
import os
import argparse
import matplotlib.pyplot as plt
from scipy import stats 

# --- CONFIGURACIÓN ---
def load_data():
    return sns.load_dataset('penguins')

def save_artifact(data, filename):
    json_str = pd.Series(data).to_json(default_handler=str)
    clean_data = json.loads(json_str)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)
    print(f"✅ Artifact generado: {filename}")


# --- FUNCIONES DE LA FASE: OBSERVE ---

def run_observe(df):
    print("🔎 Ejecutando operaciones de OBSERVE...")
    
    # 1. Profile (Artifact 00)
    profile = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns)
    }
    save_artifact(profile, "artifacts/00_raw_profile.json")

    # 2. Schema Inference (Artifact 01) - REQUERIDO POR CLAUDE
    schema = {
        "column_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns)
    }
    save_artifact(schema, "artifacts/01_schema.json")
    
    # 3. Missing Report (Artifact 02)
    missing = df.isnull().sum().to_dict()
    missing = {k: int(v) for k, v in missing.items()}
    save_artifact(missing, "artifacts/02_missing_report.json")
    
    # 4. Duplicates Report (Artifact 03)
    dups = {"total_duplicates": int(df.duplicated().sum())}
    save_artifact(dups, "artifacts/03_duplicates_report.json")

# --- OPERACIONES DESCRIPTIVAS (Se mantienen iguales) ---
def numeric_summary(df, cols, output):
    res = df[cols].describe().to_dict()
    save_artifact(res, output)

def categorical_summary(df, cols, output):
    res = {col: df[col].value_counts().to_dict() for col in cols}
    save_artifact(res, output)

def correlation_matrix(df, cols, method, output):
    res = df[cols].corr(method=method).to_dict()
    save_artifact(res, output)

def crosstab_op(df, a, b, output):
    res = pd.crosstab(df[a], df[b]).to_dict()
    save_artifact(res, output)

# --- OPERACIONES VISUALES (Se mantienen iguales) ---
def plot_count(df, x, output_registry, hue=None):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=x, hue=hue)
    path = f"artifacts/plots/count_{x}_{hue if hue else 'none'}.png"
    plt.savefig(path); plt.close()
    return {"op": "plot_count", "file": path}

def plot_hist(df, x, output_registry, groupby=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=x, hue=groupby, kde=True)
    path = f"artifacts/plots/hist_{x}_{groupby if groupby else 'none'}.png"
    plt.savefig(path); plt.close()
    return {"op": "plot_hist", "file": path}

def plot_box(df, x, y, output_registry):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y)
    path = f"artifacts/plots/box_{x}_{y}.png"
    plt.savefig(path); plt.close()
    return {"op": "plot_box", "file": path}

def plot_scatter(df, x, y, output_registry, hue=None):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    path = f"artifacts/plots/scatter_{x}_{y}.png"
    plt.savefig(path); plt.close()
    return {"op": "plot_scatter", "file": path}

def plot_heatmap_corr(df, output_registry):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    path = "artifacts/plots/heatmap_correlation.png"
    plt.savefig(path); plt.close()
    return {"op": "plot_heatmap_corr", "file": path}

# --- FASE: DESCRIBE ---
def run_describe(df):
    print("📊 Ejecutando bloque DESCRIBE...")
    stats_results = {
        "numeric": df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].describe().to_dict(),
        "categorical": {col: df[col].value_counts().to_dict() for col in ["species", "island", "sex"]},
        "correlation": df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].corr().to_dict(),
        "crosstab": pd.crosstab(df["species"], df["island"]).to_dict()
    }
    save_artifact(stats_results, "artifacts/04_descriptive_stats.json")

    viz_registry = {"visualizations": []}
    viz_registry["visualizations"].append(plot_count(df, "species", "artifacts/05_visual_registry.json"))
    viz_registry["visualizations"].append(plot_count(df, "island", "artifacts/05_visual_registry.json", hue="species"))
    viz_registry["visualizations"].append(plot_hist(df, "bill_length_mm", "artifacts/05_visual_registry.json", groupby="species"))
    viz_registry["visualizations"].append(plot_hist(df, "body_mass_g", "artifacts/05_visual_registry.json", groupby="species"))
    viz_registry["visualizations"].append(plot_box(df, "species", "flipper_length_mm", "artifacts/05_visual_registry.json"))
    viz_registry["visualizations"].append(plot_scatter(df, "bill_length_mm", "body_mass_g", "artifacts/05_visual_registry.json", hue="species"))
    viz_registry["visualizations"].append(plot_heatmap_corr(df, "artifacts/05_visual_registry.json"))
    save_artifact(viz_registry, "artifacts/05_visual_registry.json")

# --- FASE: HYPOTHESIZE AND CONCLUDE (NUEVA) ---
def run_hypothesize(df):
    print("🧪 Ejecutando bloque HYPOTHESIZE AND CONCLUDE...")
    
    # 1. Crear el Log de Hipótesis (Artifact 06) según definición de Claude
    hypotheses_log = {
        "hypotheses": [
            {"id": "H1", "statement": "Flipper length and body mass exhibit strong monotonic linear association."},
            {"id": "H2", "statement": "Physical measurements differ significantly across species groups."},
            {"id": "H3", "statement": "Species and island membership are significantly associated."},
            {"id": "H4", "statement": "Bill depth and flipper length exhibit moderate negative association."},
            {"id": "H5", "statement": "Sex distribution is balanced."}
        ]
    }
    save_artifact(hypotheses_log, "artifacts/06_hypotheses_log.json")

    # Limpieza de nulos para tests estadísticos
    df_clean = df.dropna()
    results = {}

    # Test H1: Pearson & Spearman (Flipper vs Body Mass)
    r_p, p_p = stats.pearsonr(df_clean['flipper_length_mm'], df_clean['body_mass_g'])
    r_s, p_s = stats.spearmanr(df_clean['flipper_length_mm'], df_clean['body_mass_g'])
    results["H1"] = {
        "pearson": {"statistic": r_p, "p_value": p_p, "result": "significant" if p_p < 0.05 else "not_significant"},
        "spearman": {"statistic": r_s, "p_value": p_s, "result": "significant" if p_s < 0.05 else "not_significant"}
    }

    # Test H2: ANOVA (Body Mass por Especie)
    groups = [df_clean[df_clean['species'] == s]['body_mass_g'] for s in df_clean['species'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    results["H2"] = {
        "anova": {"statistic": f_stat, "p_value": p_val, "result": "significant" if p_val < 0.05 else "not_significant"}
    }

    # Test H3: Chi-Square (Species vs Island)
    contingency = pd.crosstab(df['species'], df['island'])
    chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
    results["H3"] = {
        "chi_square": {"statistic": chi2, "p_value": p_chi, "result": "significant" if p_chi < 0.05 else "not_significant"}
    }

    # Test H4: Spearman (Bill Depth vs Flipper Length)
    r_s4, p_s4 = stats.spearmanr(df_clean['bill_depth_mm'], df_clean['flipper_length_mm'])
    results["H4"] = {
        "spearman": {"statistic": r_s4, "p_value": p_s4, "result": "significant" if p_s4 < 0.05 else "not_significant"}
    }

    # Test H5: Chi-Square Goodness of Fit (Sex distribution)
    sex_counts = df_clean['sex'].value_counts()
    chi_gof, p_gof = stats.chisquare(sex_counts)
    results["H5"] = {
        "chi_square_gof": {"statistic": chi_gof, "p_value": p_gof, "result": "significant" if p_gof < 0.05 else "not_significant"}
    }

    save_artifact(results, "artifacts/08_tests.json")

# --- FASE: CONCLUDE  ---
def run_conclude():
    print("📋 Generando Reporte Final y Conclusiones...")
    
    # Datos dictados por Claude en su análisis final
    conclusions_data = {
        "descriptive_findings": [
            {"finding": "Dataset con 342 observaciones y 3 especies. Datos faltantes mínimos."},
            {"finding": "Especie Adelie es la más prevalente (152), seguida de Gentoo (124) y Chinstrap (68)."},
            {"finding": "Distribución de sexo balanceada (p=0.869)."}
        ],
        "test_interpretations": {
            "H1": "SOPORTADA: Relación aleta-peso extremadamente fuerte.",
            "H2": "SOPORTADA: Diferencias físicas significativas entre especies.",
            "H3": "SOPORTADA: Asociación crítica entre especie e isla.",
            "H4": "SOPORTADA: Relación negativa entre profundidad de pico y aleta.",
            "H5": "CONFIRMADA: El ratio de sexo es equitativo."
        }
    }
    
    # Guardar Artifact 07
    save_artifact(conclusions_data, "artifacts/07_conclusions.json")

    # Generar Reporte Markdown (report.md)
    report_content = """# Informe de Análisis: Penguin Dataset
## Resumen Ejecutivo
Análisis completo del dataset de pingüinos utilizando el flujo OBSERVE -> DESCRIBE -> HYPOTHESIZE.

## Hallazgos Clave
1. **Morfología**: Las aletas y el peso están altamente integrados (r=0.87).
2. **Ecología**: Las especies están segregadas por islas (Gentoo solo en Biscoe, Chinstrap en Dream).
3. **Estadística**: Todas las hipótesis de diferencia morfológica fueron validadas (p < 0.001).

## Conclusión
El dataset muestra una estructura biológica clara donde la especie y el hábitat determinan las características físicas de los individuos.
"""
    with open("artifacts/report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("✅ Reporte final generado en: artifacts/report.md")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str)
    args = parser.parse_args()
    df = load_data()

    if args.phase == 'OBSERVE':
        run_observe(df)
    elif args.phase == 'DESCRIBE':
        run_describe(df)
    elif args.phase == 'HYPOTHESIZE':
        run_hypothesize(df)
    elif args.phase == 'CONCLUDE': 
        run_conclude()
    else:
        print("⚠️ Fase no reconocida.")

if __name__ == "__main__":
    os.makedirs('artifacts/plots', exist_ok=True)
    main()