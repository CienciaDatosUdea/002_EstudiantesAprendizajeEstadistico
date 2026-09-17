import json
import os
import pandas as pd
import seaborn as sns

os.makedirs("artifacts", exist_ok=True)
df = sns.load_dataset("penguins")

# 1. Generar 00_raw_profile.json
profile = {
    "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "columns": {
        col: {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
        }
        for col in df.columns
    },
    "duplicates": int(df.duplicated().sum()),
}

with open("artifacts/00_raw_profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=4)

# 2. Generar 04_descriptive_stats.json
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
num_stats = {}
for col in num_cols:
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    num_stats[col] = {
        "mean": round(float(df[col].mean()), 2),
        "median": round(float(df[col].median()), 2),
        "std": round(float(df[col].std()), 2),
        "iqr": round(float(q75 - q25), 2),
    }

with open("artifacts/04_descriptive_stats.json", "w", encoding="utf-8") as f:
    json.dump({"numerical_summary": num_stats}, f, indent=4)

print("¡Artefactos 00 y 04 generados correctamente en /artifacts!")
