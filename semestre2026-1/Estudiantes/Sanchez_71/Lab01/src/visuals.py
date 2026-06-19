import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def save_plot(filename, show=True):
    """Saves the plot and optionally displays it in the notebook."""
    path = os.path.join("../artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show() # This renders the plot in the notebook
    plt.close()    # This clears the memory
    return path

def plot_categorical_counts(df, columns):
    for col in columns:
        plt.figure()
        sns.countplot(data=df, x=col, hue=col, palette="viridis", legend=False)
        plt.title(f"Distribution of {col}")
        save_plot(f"count_{col}.png")

def plot_numerical_histograms(df, columns):
    for col in columns:
        plt.figure()
        sns.histplot(data=df, x=col, kde=True, color="skyblue")
        plt.title(f"Histogram of {col}")
        save_plot(f"hist_{col}.png")

def plot_boxplot_by_category(df, numerical_col, categorical_col):
    plt.figure()
    sns.boxplot(data=df, x=categorical_col, y=numerical_col, hue=categorical_col, palette="Set2", legend=False)
    plt.title(f"{numerical_col} by {categorical_col}")
    save_plot(f"boxplot_{numerical_col}_{categorical_col}.png")

def plot_scatter_by_category(df, x_col, y_col, hue_col):
    plt.figure()
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7)
    plt.title(f"Interaction: {x_col} vs {y_col} (Colored by {hue_col})")
    save_plot(f"scatter_{x_col}_{y_col}.png")

def plot_correlation_heatmap(df, method="pearson"):
    plt.figure()
    corr = df.select_dtypes(include=['number']).corr(method=method)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Correlation Heatmap ({method.capitalize()})")
    save_plot(f"heatmap_{method}.png")