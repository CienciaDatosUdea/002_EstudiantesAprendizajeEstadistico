import pandas as pd
import numpy as np
import os    
import json  

def get_initial_observation(df):
    """
    Performs structural analysis (Parte A).
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "types": {"numerical": numerical_cols, "categorical": categorical_cols},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "low_cardinality": [col for col in df.columns if df[col].nunique() < 10]
    }

def get_statistical_description(df):
    """
    Calculates moments of distribution and correlations (Parte B).
    """
    # 1. Numerical Summary
    num_df = df.select_dtypes(include=[np.number])
    summary = num_df.describe().to_dict()
    for col in num_df.columns:
        summary[col]["iqr"] = float(num_df[col].quantile(0.75) - num_df[col].quantile(0.25))

    # 2. Categorical Summary
    cat_df = df.select_dtypes(exclude=[np.number])
    cat_summary = {col: {"counts": cat_df[col].value_counts().to_dict(), 
                         "percentages": (cat_df[col].value_counts(normalize=True)*100).to_dict()} 
                   for col in cat_df.columns}

    # 3. Correlations
    pearson = num_df.corr(method='pearson').to_dict()
    spearman = num_df.corr(method='spearman').to_dict()

    return {
        "numerical": summary,
        "categorical": cat_summary,
        "correlations": {"pearson": pearson, "spearman": spearman}
    }

def log_hypotheses(hypotheses_list, filename="06_hypotheses_log.json"):
    """
    Persists a list of falsifiable hypotheses as a JSON artifact.
    """
    # Ensure the path is relative to the project root
    path = os.path.join("..", "artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump({"hypotheses": hypotheses_list}, f, indent=4)
    return path

def save_final_report(report_data, filename="07_conclusions.json"):
    """
    Persists the final conclusions and researcher questions as a JSON artifact.
    """
    path = os.path.join("..", "artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    return path