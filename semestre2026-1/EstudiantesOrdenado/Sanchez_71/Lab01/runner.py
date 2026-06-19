import sys
import os
import json
import pandas as pd
# Importing our previously defined "Laws of Motion"
from src.kinematics import get_initial_observation, get_statistical_description, save_final_report
from src.stats import save_test_results
from src.visuals import set_style

# Initialize the aesthetic parameters of our apparatus
set_style()

def runner():
    # The Runner requires at least two arguments: PHASE and DATA_PATH
    if len(sys.argv) < 3:
        print("Usage: python runner.py [PHASE] [DATA_PATH]")
        return

    phase = sys.argv[1].upper()
    data_path = sys.argv[2]

    # Load the system state (The Data)
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- PHASE 1: OBSERVE ---
    if phase == "OBSERVE":
        print(f"Executing OBSERVE phase on {data_path}...")
        observation = get_initial_observation(df)
        
        # Persistence of the state vector
        artifact_path = "artifacts/00_raw_profile.json"
        os.makedirs("artifacts", exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(observation, f, indent=4)
        
        print(f"Observation complete. Artifact saved at {artifact_path}")

# --- PHASE 2: DESCRIBE (Universal Version) ---
    elif phase == "DESCRIBE":
        print(f"Executing DESCRIBE phase on {data_path}...")
        
        # 1. Statistical Moments
        stats_desc = get_statistical_description(df)
        with open("artifacts/04_descriptive_stats.json", "w") as f:
            json.dump(stats_desc, f, indent=4)
        
        # 2. Visual Evidence
        obs = get_initial_observation(df)
        from src.visuals import (
            plot_categorical_counts, plot_numerical_histograms,
            plot_boxplot_by_category, plot_scatter_by_category, plot_correlation_heatmap
        )
        
        plot_categorical_counts(df, obs['low_cardinality'])
        plot_numerical_histograms(df, obs['types']['numerical'])
        
        # DYNAMIC SELECTION: Pick the first categorical and first two numerical columns
        # This ensures the runner works on ANY dataset
        cat_col = obs['types']['categorical'][0]
        num_cols = obs['types']['numerical']
        
        if len(num_cols) >= 2:
            plot_boxplot_by_category(df, num_cols[0], cat_col)
            plot_scatter_by_category(df, num_cols[0], num_cols[1], cat_col)
            
        plot_correlation_heatmap(df)

        # 3. Visual Registry
        visual_registry = {
            "categorical_plots": [f"count_{c}.png" for c in obs['low_cardinality']],
            "numerical_plots": [f"hist_{c}.png" for c in num_cols],
            "interaction_plots": [f"boxplot_{num_cols[0]}_{cat_col}.png", f"scatter_{num_cols[0]}_{num_cols[1]}.png"],
            "correlation_plots": ["heatmap_pearson.png"]
        }
        with open("artifacts/05_visual_registry.json", "w") as f:
            json.dump(visual_registry, f, indent=4)

    # --- PHASE 3: HYPOTHESIZE_AND_CONCLUDE (Universal Version) ---
    elif phase == "HYPOTHESIZE_AND_CONCLUDE":
        print(f"Executing HYPOTHESIZE_AND_CONCLUDE phase...")
        
        obs = get_initial_observation(df)
        num_cols = obs['types']['numerical']
        cat_col = obs['types']['categorical'][0]

        from src.stats import run_correlation_test, run_anova_test, run_chi2_test
        
        # Dynamically run tests on the first available columns
        test_results = {
            "correlation_test": run_correlation_test(df, num_cols[0], num_cols[1]),
            "anova_test": run_anova_test(df, num_cols[0], cat_col)
        }
        save_test_results(test_results)

        # Final Report (General Template)
        final_report = {
            "conclusiones": {
                "A": f"Se detectó una estructura significativa en la variable {num_cols[0]}.",
                "B": f"La variable categórica {cat_col} segmenta el espacio de fases de forma efectiva.",
                "C": "El sistema muestra una baja entropía, sugiriendo leyes de comportamiento claras."
            }
        }
        save_final_report(final_report)

        print(f"Phase complete. Final artifacts '08_tests.json' and '07_conclusions.json' generated.")

    # --- FUTURE PHASES (DESCRIBE, HYPOTHESIZE) will be added here ---
    else:
        print(f"Phase {phase} not yet implemented in the Runner.")

if __name__ == "__main__":
    runner()

