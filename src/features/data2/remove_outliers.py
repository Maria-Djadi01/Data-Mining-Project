import sys

# Specify the directory where your data is located
project_dir = "D:/2M/D.Mining/Data-Mining-Project/"

# Change the working directory
sys.path.append(project_dir)

import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
import numpy as np
import warnings

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------#
# Load data
# ----------------------------------------------------------------#

# df = pd.read_csv("../../../data/interim/temp_dataset_processed.csv")
# df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
# # Load data #
# # ----------------------------------------------------------------#

df = pd.read_csv("../../../data/interim/02_temp_dataset_fill_miss_val.csv", index_col=0)
# df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)

# --------------------------------------------------------------#
# Plotting outliers
# --------------------------------------------------------------#
selected_columns = ['case count', 'test count', 'positive tests']
df_selected_columns = df[selected_columns]
for column in df.columns:
    if column in selected_columns:
        df[[column, "zcta"]].boxplot(by="zcta", figsize=(20, 10))
        plt.title(f"Boxplot for {column}")
        plt.savefig(f"../../visualization/outliers/{column}_boxplot.png")
        plt.show()

row_to_drop = df[df['positive tests'] == 35000]
df.drop(row_to_drop.index, inplace=True)

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["no outlier " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------


# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_fence) | (
        dataset[col] > upper_fence
    )

    return dataset


# Loop over all columns
for col in df_selected_columns.columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", True)

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------
# test on a single column
dataset = mark_outliers_iqr(df, "case count")
# number of missing values
dataset.isnull().sum()
dataset[dataset["case count" + "_outlier"] == True] = np.nan
# drop the null raws
dataset = dataset.dropna(how="all")
dataset.drop(columns=["case count" + "_outlier"], inplace=True)

# Create a loop
dataset = df.copy()
for col in selected_columns:
    dataset = mark_outliers_iqr(dataset, col)
    print(f'Number of outliers for {col} is {dataset[dataset[col + "_outlier"] == True][col + "_outlier"].sum()}')
    dataset[dataset[col + "_outlier"] == True] = np.nan
    dataset = dataset.dropna(how="all")
    n_outliers = len(dataset) - len(dataset[col].dropna(how="all"))
    dataset.drop(columns=[col + "_outlier"], inplace=True)

outliers_removed_df = dataset
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
# convert zcta to int
outliers_removed_df["zcta"] = outliers_removed_df["zcta"].astype(int)
outliers_removed_df.to_csv(
    "../../../data/interim/03_temp_dataset_processed_outliers_removed.csv"
)

outliers_removed_df.info()
