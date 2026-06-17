import os
import json
import seaborn as sns
import pandas as pd

# =====================================
# Runner para Fase 2 - Enfoque con Agentes
# =====================================
# Ejecuta: python runner.py
# Genera: artifacts/00_raw_profile.json
#
# Este script debe ejecutarse desde la carpeta fase2_agentes para mantener separados los artifacts de la Fase 2.
# =====================================

def ensure_artifacts_dirs():
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('artifacts/plots', exist_ok=True)

def load_dataset():
    try:
        df = sns.load_dataset('penguins')
    except Exception:
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'
        df = pd.read_csv(url)
    return df

def profile_dataset(df):
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'head': df.head(5).to_dict(orient='list')
    }
    return profile

def infer_schema(df):
    schema = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            tipo = 'numeric'
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
            tipo = 'categorical'
        else:
            tipo = str(df[col].dtype)
        schema[col] = tipo
    return schema

def missing_report(df):
    return df.isnull().sum().to_dict()

def duplicates_report(df):
    return {'n_duplicates': int(df.duplicated().sum())}

def main():
    ensure_artifacts_dirs()
    df = load_dataset()
    profile = profile_dataset(df)
    schema = infer_schema(df)
    missing = missing_report(df)
    duplicates = duplicates_report(df)
    raw_profile = {
        'profile': profile,
        'schema': schema,
        'missing': missing,
        'duplicates': duplicates
    }
    with open('artifacts/00_raw_profile.json', 'w', encoding='utf-8') as f:
        json.dump(raw_profile, f, ensure_ascii=False, indent=2)
    print('artifacts/00_raw_profile.json generado correctamente.')
    print('\nPara ejecutar este runner, usa en terminal:')
    print('    python runner.py')
    print('Asegúrate de estar en la carpeta fase2_agentes para mantener los artifacts organizados.')

if __name__ == '__main__':
    main()
