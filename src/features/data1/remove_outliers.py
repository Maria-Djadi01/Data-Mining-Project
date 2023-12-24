import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
import sys
import seaborn as sns

# Change the working directory
sys.path.append("../../../../Data-Mining-Project")

from src.utils import (
    mark_outliers_iqr,
    plot_binary_outliers,
    calculate_quartiles,
    box_plot,
)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------

df = pd.read_csv("../../../data/interim/processed_static_dataset.csv")
df = df.drop(columns=["Unnamed: 0"])

outlier_columns = list(df.columns[:13])

# ---------------------------------------------------------------
# Plotting outliers
# ---------------------------------------------------------------

for column in outlier_columns:
    df[[column, "Fertility"]].boxplot(by="Fertility", figsize=(20, 10))
    plt.title(f"Boxplot for {column}")
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", True)

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# test on a single column
dataset = mark_outliers_iqr(df, "P")
Min, Q1, Q2, Q3, Max = calculate_quartiles(dataset["P"])
IQR = Q3 - Q1
if (dataset["P_outlier"] == True).any():
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    dataset["P"] = dataset["P"].clip(lower=lower_fence, upper=upper_fence)
dataset.iloc[279]


# Create a loop
dataset = df.copy()
for col in outlier_columns:
    dataset = mark_outliers_iqr(dataset, col)
    Min, Q1, Q2, Q3, Max = calculate_quartiles(dataset[col])
    IQR = Q3 - Q1
    if (dataset[col + "_outlier"] == True).any():
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        dataset[col] = dataset[col].clip(lower=lower_fence, upper=upper_fence)
        dataset.drop(columns=[col + "_outlier"], inplace=True)

outliers_removed_df = dataset

# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
# fig.suptitle("Fertility vs. Other Variables", y=1.02)

# for i, column in enumerate(outlier_columns):
#     sns.boxplot(
#         x="Fertility", y=column, data=outliers_removed_df, ax=axes[i // 4, i % 4]
#     )

# plt.tight_layout()
# plt.show()

for col in outlier_columns:
    dataset = mark_outliers_iqr(outliers_removed_df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", True)

box_plot(outliers_removed_df)

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_csv(
    "../../../data/interim/02_static_dataset_processed_outliers_removed.csv"
)
