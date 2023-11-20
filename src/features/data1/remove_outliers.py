import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy


from src.utils import mark_outliers_chauvenet, mark_outliers_iqr, plot_binary_outliers

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
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", True)


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Chauvenets criteron
# test on a single column
dataset = mark_outliers_chauvenet(df, "P")
# number of missing values
dataset.isnull().sum()
dataset[dataset["P" + "_outlier"] == True] = np.nan
# drop the null raws
dataset = dataset.dropna(how="all")
dataset.drop(columns=["P" + "_outlier"], inplace=True)

# Create a loop
dataset = df.copy()
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(dataset, col)
    dataset[dataset[col + "_outlier"] == True] = np.nan
    dataset = dataset.dropna(how="all")
    n_outliers = len(dataset) - len(dataset[col].dropna(how="all"))
    dataset.drop(columns=[col + "_outlier"], inplace=True)
    print(f"{col} - {n_outliers} outliers removed")

outliers_removed_df = dataset


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_csv(
    "../../../data/interim/02_static_dataset_processed_outliers_removed.csv"
)
