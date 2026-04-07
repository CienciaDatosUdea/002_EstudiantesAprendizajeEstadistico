import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

# 1. Read the dataset
df = pl.read_csv("https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/Pandas_data_historical_dataEURUSD.csv")

# 2. Set the 'time' column as the index. 
# Note: Polars does not have a built-in index like pandas, but we can sort the DataFrame by the 'time' column to ensure it is in the correct order for time series analysis.
df = df.drop("")  # Drop the unnecessary column

# Set the 'time' column as a datetime type
df = df.with_columns(
    pl.col("time").str.to_datetime()
)

df = df.sort("time")

# 3. Dataset information
print((df.schema))

(df.describe()).show()

# 4. Null and nan values
null_count = df.null_count().sum()
nan_count = df.select(cs.numeric().is_nan().sum()).sum()

print(f"Null values: {null_count}")
print(f"NaN values: {nan_count}")

# 5. Use PascalCase for column names
columns = df.columns
new_columns = [col.title().replace("_", "") if "MeanCloseOpen" not in col else col.replace("_", "") for col in columns]

df = df.rename(dict(zip(columns, new_columns)))

print(df.columns)

# 6. Add new column named 'DiffPrice' that contains the difference between 'Close' and 'Open' prices.

df = df.with_columns(
    (pl.col("Close") - pl.col("Open")).alias("DiffPrice")
).select(["Time", "Close", "DiffPrice"])

# Distribution of 'DiffPrice'

plt.hist(df["DiffPrice"], bins=30, edgecolor="black")
plt.title("Distribution of DiffPrice")
plt.xlabel("DiffPrice")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.show()

# Determine the best distribution for 'DiffPrice'

data = df["DiffPrice"].to_numpy()

distributions = (get_common_distributions() + [dist for dist in ['gamma', 'lognorm', "beta", "burr" "norm"] if dist not in get_common_distributions()])  

f = Fitter(data,
           distributions=distributions)
f.fit()
f.summary()

f.get_best(method="sumsquare_error")

f.fitted_param["beta"]

print(f"The best distribution for 'DiffPrice' is: {f.get_best(method='sumsquare_error')}")

# 7. Select data from 2023

df_2023 = df.filter(pl.col("Time").dt.year() == 2023).sort("Time", descending=False)

# 8. Mean with periodicity of 15 days, a week, and a month

mean_15d = (
    df_2023
    .sort("Time")
    .group_by_dynamic("Time", every="15d", start_by="datapoint")
    .agg(pl.col("DiffPrice").mean().alias("Mean_DiffPrice_15d"))
)

mean_1w = (
    df_2023
    .sort("Time")
    .group_by_dynamic("Time", every="1w", start_by="datapoint")
    .agg(pl.col("DiffPrice").mean().alias("Mean_DiffPrice_1w"))
)

mean_1mo = (
    df_2023
    .sort("Time")
    .group_by_dynamic("Time", every="1mo", start_by="datapoint")
    .agg(pl.col("DiffPrice").mean().alias("Mean_DiffPrice_1mo"))
)

print(mean_15d)
print(mean_1w)
print(mean_1mo)

# 9. Histogram for each month of 2023

df_2023_grouped = (
    df_2023
    .sort("Time")
    .group_by(pl.col("Time").dt.month(), maintain_order=True)
)

for (month,), group in df_2023_grouped:
    plt.hist(group["DiffPrice"], bins=30, edgecolor="black")
    plt.title(f"Distribution of DiffPrice for Month {month}")
    plt.xlabel("DiffPrice")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()

