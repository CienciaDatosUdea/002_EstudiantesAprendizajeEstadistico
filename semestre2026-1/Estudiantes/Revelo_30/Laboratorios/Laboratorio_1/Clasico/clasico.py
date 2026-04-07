import polars as pl
import polars.selectors as cs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
import json

# Load penguins dataset from seaborn import load_dataset
penguins = pl.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")


path = Path('/home/mrevelo/Projects/U/Semestre 2026-1/Fisica computacional 2/002_EstudiantesAprendizajeEstadistico/semestre2026-1/Estudiantes/Revelo_30/Laboratorios/Laboratorio_1/Clasico/artifacts')

def runner(dataset: pl.DataFrame, dataset_name: str):

    artifacts_path = Path(f"{path}/{dataset_name}")

    artifacts_plots_path = artifacts_path / "plots"

    if not artifacts_path.exists():
        artifacts_path.mkdir(parents=True, exist_ok=True)

    if not artifacts_plots_path.exists():
        artifacts_plots_path.mkdir(parents=True, exist_ok=True)

    # Part A: Exploration

    # 1. Number of rows and columns

    num_rows, num_cols = dataset.shape

    rows_cols_artifact = pl.DataFrame({
        "num_rows": [num_rows],
        "num_cols": [num_cols]
    })

    rows_cols_artifact.write_json(f"{artifacts_path}/rows_cols.json")

    print("1. Number of rows and columns:")
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    print("--" * 20)


    # 2. Which columns are numeric and which are categorical?

    numeric_cols = dataset.select(pl.selectors.numeric()).columns

    categorical_cols = dataset.select(pl.selectors.string()).columns

    numeric_cols_artifact = pl.DataFrame({
        "numeric_columns": numeric_cols
    })
    categorical_cols_artifact = pl.DataFrame({
        "categorical_columns": categorical_cols
    })

    numeric_cols_artifact.write_json(f"{artifacts_path}/numeric_cols.json")
    categorical_cols_artifact.write_json(f"{artifacts_path}/categorical_cols.json")

    print("2. Which columns are numeric and which are categorical?")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print("--" * 20)


    # 3. Null values in each column

    null_counts = dataset.null_count()

    if not null_counts.is_empty():
        null_counts.write_json(f"{artifacts_path}/null_counts.json")

    print("3. Null values in each column:")
    print("Null values in each column:")
    print(null_counts)
    print("--" * 20)

    # Drop rows with null values for the rest of the analysis
    dataset = dataset.drop_nulls().drop_nans()


    # 4. Duplicated rows?

    duplicated_rows = dataset.is_duplicated().sum()

    duplicated_rows_artifact = pl.DataFrame({
        "duplicated_rows": [duplicated_rows]
    })
    duplicated_rows_artifact.write_json(f"{artifacts_path}/duplicated_rows.json")

    print("4. Duplicated rows:")
    print(f"Number of duplicated rows: {duplicated_rows}")
    print("--" * 20)


    # 5. Low cardinality columns (less than 10 unique values)

    low_cardinality_cols = [col for col in dataset.columns if dataset[col].n_unique() < 10]

    low_cardinality_cols_artifact = pl.DataFrame({
        "low_cardinality_columns": low_cardinality_cols
    })
    low_cardinality_cols_artifact.write_json(f"{artifacts_path}/low_cardinality_cols.json")

    print("5. Low cardinality columns (less than 10 unique values):")
    print(f"Low cardinality columns: {low_cardinality_cols}")
    print("--" * 20)


    # Part B: Description

    # 6. Mean, median, standard deviation and interquartile range of numeric columns as a table

    summary = pl.DataFrame({
        "column": numeric_cols,
        "mean": [dataset.select(pl.col(c).mean()).item() for c in numeric_cols],
        "median": [dataset.select(pl.col(c).median()).item() for c in numeric_cols],
        "std": [dataset.select(pl.col(c).std()).item() for c in numeric_cols],
        "iqr": [
            dataset.select(pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)).item()
            for c in numeric_cols
        ],
    })

    summary.write_json(f"{artifacts_path}/numeric_summary.json")

    print("6. Mean, median, standard deviation and interquartile range of numeric columns:")
    print(summary)

    print("--" * 20)


    # 7. Count and percentage of each category in categorical columns as a table 

    result = pl.concat(
        [
            dataset
            .group_by(col)
            .len()
            .with_columns(
                (pl.col("len") / dataset.height * 100).round(2).alias("percentage"),
                pl.lit(col).alias("column")
            )
            .rename({col: "category", "len": "count"})
            .select(["column", "category", "count", "percentage"])
            .sort("column", "count", descending=[False, True])
            for col in categorical_cols
        ]
    )

    result.write_json(f"{artifacts_path}/categorical_counts.json")

    print("7. Count and percentage of each category in categorical columns:")
    print(result)
    print("--" * 20)


    # 8. Cross tables for relevant categorical columns

    # Species per island

    cross_table_1 = dataset.pivot(
        values="species",
        index="island",
        on="species",
        aggregate_function="len"
    )

    cross_table_1.write_json(f"{artifacts_path}/cross_table_species_island.json")

    # Species per sex

    cross_table_2 = dataset.pivot(
        values="species",
        index="species",
        on="sex",
        aggregate_function="len"
    )

    cross_table_2.write_json(f"{artifacts_path}/cross_table_species_sex.json")

    # Sex per island

    cross_table_3 = dataset.pivot(
        values="sex",
        index="island",
        on="sex",
        aggregate_function="len"
    )

    cross_table_3.write_json(f"{artifacts_path}/cross_table_sex_island.json")

    categorical_cols

    print("8. Cross tables for relevant categorical columns:")
    print("Species per island:")
    print(cross_table_1)
    print("Species per sex:")
    print(cross_table_2)
    print("Sex per island:")
    print(cross_table_3)
    print("--" * 20)


    # 9. Pearson and Spearman correlation matrix for numeric columns

    dataset_pd = dataset.to_pandas()

    numeric_df = dataset_pd.select_dtypes(include=["number"])

    pearson_corr = numeric_df.corr(method="pearson")
    spearman_corr = numeric_df.corr(method="spearman")

    pearson_corr_polars = (
        pl.from_pandas(pearson_corr.reset_index())
        .rename({"index": "variable"})
    )

    pearson_corr_polars.write_json(f"{artifacts_path}/pearson_corr.json")

    spearman_corr_polars = (
        pl.from_pandas(spearman_corr.reset_index())
        .rename({"index": "variable"})
    )

    spearman_corr_polars.write_json(f"{artifacts_path}/spearman_corr.json")

    print("Pearson correlation matrix:")
    print(pearson_corr_polars)

    print("\nSpearman correlation matrix:")
    print(spearman_corr_polars)


    # Part C: Visualization

    # 10. Count plots for categorical columns with low cardinality

    low_cardinality_cols = [
        col for col in categorical_cols
        if dataset[col].n_unique() <= 10
    ]

    print("10. Count plots for categorical columns with low cardinality:")

    for col in low_cardinality_cols:
        figure = plt.figure(figsize=(8, 5))
        (
            dataset_pd[col]
            .value_counts(dropna=False)
            .plot.bar(title=f"Count plot of {col}", figsize=(8, 5))
        )
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        figure.savefig(f"{artifacts_plots_path}/categorical_count_plot_{col}.png")


    # 11. Histograms for numeric columns and description of their distribution

    print("11. Histograms for numeric columns and description of their distribution:")
    histogram_descriptions = {}
    for col in numeric_cols:
        hist = dataset.select(pl.col(col)).to_pandas().plot.hist(bins=20, figsize=(8, 5))
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.show()
        hist.get_figure().savefig(f"{artifacts_plots_path}/histogram_{col}.png")
        # Shape of the distribution
            ## Simetric, or skewed to the right or left, unimodal or multimodal, etc.

        #calculate the skewness of the distribution
        skewness = dataset.select(pl.col(col).skew()).item()
        if skewness > 0:
            skewness_description = "skewed to the right"
        elif skewness < 0:
            skewness_description = "skewed to the left"
        else:
            skewness_description = "symmetric"

        # Center
            ## Mean, median, or mode
            ## Centered around mean or median?

        #calculate the mean and median of the distribution
        mean = dataset.select(pl.col(col).mean()).item()
        median = dataset.select(pl.col(col).median()).item()
        if abs(mean - median) < 0.1 * abs(mean):
            center_description = "centered around the mean"
        else:        
            center_description = "centered around the median"

        # Spread
            ## Standard deviation, interquartile range, etc.
            ## Is the spread large or small compared to the center?

        #calculate the standard deviation and interquartile range of the distribution
        std = dataset.select(pl.col(col).std()).item()
        iqr = dataset.select(pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).item()
        if std < 0.5 * abs(mean):
            spread_description = "small spread compared to the center"
        else:
            spread_description = "large spread compared to the center"

        # Outliers
            ## Are there any outliers? How do they affect the distribution?
            ## Gaps in the distribution?
            ## Clusters in the distribution?

        #calculate the number of outliers using the IQR method
        q1 = dataset.select(pl.col(col).quantile(0.25)).item()
        q3 = dataset.select(pl.col(col).quantile(0.75)).item()
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = dataset.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).select(col)
        if len(outliers) > 0:
            outliers_description = f"there are {len(outliers)} outliers that may affect the distribution"
        else:
            outliers_description = "there are no outliers that may affect the distribution"

        #calculate the number of gaps in the distribution using a histogram with 20 bins
        counts, bin_edges = np.histogram(dataset.select(pl.col(col)).to_pandas(), bins=20)
        gaps = np.sum(counts == 0)
        if gaps > 0:
            gaps_description = f"there are {gaps} gaps in the distribution"
        else:
            gaps_description = "there are no gaps in the distribution"
        
        # Maximum and minimum values
        max_value = dataset.select(pl.col(col).max()).item()
        min_value = dataset.select(pl.col(col).min()).item()

        print("Description:")
        print(f"The distribution of {col} is {skewness_description}, {center_description}, {spread_description}, {outliers_description}, and {gaps_description}.The mean of the distribution is {mean} and the median is {median}. The maximum value is {max_value} and the minimum value is {min_value}.")

        histogram_descriptions[col] = {
            "skewness": skewness_description,
            "center": center_description,
            "spread": spread_description,
            "outliers": outliers_description,
            "gaps": gaps_description,
            "mean": mean,
            "median": median,
            "max": max_value,
            "min": min_value
        }

    with open(f"{artifacts_path}/histogram_descriptions.json", "w") as f:
        json.dump(histogram_descriptions, f, indent=4)


    # 12. Box plots for numerc culumns per categorical column
    print("12. Box plots for numeric columns per categorical column:")
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            figure = plt.figure(figsize=(8, 5))
            sns.boxplot(x=cat_col, y=num_col, data=dataset)
            plt.title(f"Box plot of {num_col} by {cat_col}")
            plt.xlabel(cat_col)
            plt.ylabel(num_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            figure.savefig(f"{artifacts_plots_path}/boxplot_{num_col}_by_{cat_col}.png")


    # 13. Scatter plots for numeric columns colored by categorical columns
    print("13. Scatter plots for numeric columns colored by categorical columns:")
    for num_col1 in numeric_cols:
        for num_col2 in numeric_cols:
            if num_col1 != num_col2:
                for cat_col in categorical_cols:
                    figure = plt.figure(figsize=(8, 5))
                    sns.scatterplot(x=num_col1, y=num_col2, hue=cat_col, data=dataset)
                    plt.title(f"Scatter plot of {num_col1} vs {num_col2} colored by {cat_col}")
                    plt.xlabel(num_col1)
                    plt.ylabel(num_col2)
                    plt.legend(title=cat_col)
                    plt.tight_layout()
                    plt.show()
                    figure.savefig(f"{artifacts_plots_path}/scatterplot_{num_col1}_vs_{num_col2}_by_{cat_col}.png")


    # 14. Heatmap of the correlation matrix for numeric columns
    print("14. Heatmap of the correlation matrix for numeric columns:")
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap of Pearson Correlation Matrix")
    plt.tight_layout()
    plt.show()
    figure.savefig(f"{artifacts_plots_path}/heatmap_pearson_corr.png")

    # Part D: Hipothesis (whithout causality)

    # 15. Propose at least 3 falseable hypothesis based on the descriptive analysis and visualizations of the data. 

    print("15. Propose at least 3 falseable hypothesis based on the descriptive analysis and visualizations of the data.")
    count = 0
    already_selected = []
    hypotheses = {}
    for col1 in pearson_corr.columns:
        already_selected.append(col1)
        for col2 in pearson_corr.columns:
            if col1 != col2 and col2 not in already_selected:
                corr_value = pearson_corr.loc[col1, col2]
                if abs(corr_value) < 0.5:
                    print(f"Hypothesis: There is a correlation between {col1} and {col2}.")
                    hypotheses[f"false.{col1}:{col2}"] = f"There is a correlation between {col1} and {col2}."
                    count += 1
                    if count >= 3:
                        break
                else:
                    print(f"Hypothesis: There is no correlation between {col1} and {col2}.")
                    hypotheses[f"true.{col1}:{col2}"] = f"There is no correlation between {col1} and {col2}."
                    count += 1
                    if count >= 3:
                        break
        if count >= 3:
            break
    
    with open(f"{artifacts_path}/hypotheses.json", "w") as f:
        json.dump(hypotheses, f, indent=4)



    # Part E: Statistical tests

    # 16. Select an adecuate statistical test (Pearson/Spearman, ANOVA/Kruskal, chi-square) to evaluate each of the proposed hypothesis.

    print("16. Select an adecuate statistical test (Pearson/Spearman, ANOVA/Kruskal, chi-square) to evaluate each of the proposed hypothesis.")

    for key, value in hypotheses.items():
        print(f"To evaluate the hypothesis '{value}', we can use the Pearson correlation test.")

    # 17. Statistical report and p-value for each hypothesis

    # 18. Indicate if the hypothesis is supported or refuted based on the p-value and the proposed hypothesis.

    print("17. Statistical report and p-value for each hypothesis\nand\n 18. Indicate if the hypothesis is supported or refuted based on the p-value and the proposed hypothesis:")

    alpha = 0.05
    statistical_results = {}
    for key, value in hypotheses.items():
        print("Hypothesis:")
        print(value)
        print("Statistical test: Pearson correlation test")

        col1, col2 = key.split(".")[1].split(":")
        df_pair = dataset.select([col1, col2]).drop_nulls()

        x = df_pair[col1].to_numpy()
        y = df_pair[col2].to_numpy()

        corr_value, p_value = pearsonr(x, y)

        print(f"Correlation coefficient: {corr_value:.2f}")
        print(f"P-value: {p_value}")

        abs_r = abs(corr_value)
        if abs_r < 0.2:
            strength = "very weak"
        elif abs_r < 0.4:
            strength = "weak"
        elif abs_r < 0.6:
            strength = "moderate"
        elif abs_r < 0.8:
            strength = "strong"
        else:
            strength = "very strong"

        if corr_value > 0:
            direction = "positive"
        else:
            direction = "negative"

        print(f"Interpretation: {strength} {direction} correlation.")

        significant = p_value < alpha
        if significant:
            print("Conclusion: Reject H0 (no correlation).")
        else:
            print("Conclusion: Fail to reject H0 (no evidence of correlation).")

        proposed_no_corr = "There is no correlation" in value

        if proposed_no_corr and significant:
            print("Outcome: Hypothesis refuted.")
        elif proposed_no_corr and not significant:
            print("Outcome: Hypothesis supported.")
        elif not proposed_no_corr and significant:
            print("Outcome: Hypothesis supported.")
        else:
            print("Outcome: Hypothesis refuted.")

        print("---" * 10)
        statistical_results[key.split(".")[1].replace(":", "-")] = {
            "hypothesis": value,
            "test": "Pearson correlation test",
            "correlation_coefficient": corr_value,
            "p_value": p_value,
            "interpretation": f"{strength} {direction} correlation",
            "significant": str(significant),
            "conclusion": "Reject H0" if significant else "Fail to reject H0",
            "outcome": "Hypothesis supported" if (proposed_no_corr and not significant) or (not proposed_no_corr and significant) else "Hypothesis refuted"
        }
    
    with open(f"{artifacts_path}/statistical_results.json", "w") as f:
        json.dump(statistical_results, f, indent=4)

    # 19) Redacte conclusiones en tres capas:
    #     A) Hallazgos descriptivos (con evidencia).
    #     B) Patrones visuales (con evidencia).
    #     C) Próximas hipótesis a probar.
    # 20) Incluya preguntas para un investigador humano (p. ej., manejo de la variable `sex`, control por `species`).


if __name__ == "__main__":
    runner(dataset=penguins, dataset_name="penguins")