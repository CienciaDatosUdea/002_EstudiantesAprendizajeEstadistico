import pandas as pd
from scipy import stats
import os
import json

def run_correlation_test(df, var1, var2):
    """Performs a Pearson correlation test."""
    # Remove NaNs for the specific pair
    data = df[[var1, var2]].dropna()
    r, p_val = stats.pearsonr(data[var1], data[var2])
    return {"statistic": float(r), "p_value": float(p_val)}

def run_anova_test(df, numerical_col, categorical_col):
    """Performs a One-Way ANOVA test."""
    groups = [group[numerical_col].dropna() for name, group in df.groupby(categorical_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    return {"statistic": float(f_stat), "p_value": float(p_val)}

def run_chi2_test(df, var1, var2):
    """Performs a Chi-squared test of independence."""
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    return {"statistic": float(chi2), "p_value": float(p_val)}

def save_test_results(results, filename="08_tests.json"):
    """Saves the statistical results as an artifact."""
    path = os.path.join("../artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    return path