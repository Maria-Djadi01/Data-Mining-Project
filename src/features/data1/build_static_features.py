import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import sys
import seaborn as sns

sys.path.append("../../../../Data-Mining-Project")

from src.utils import correlation_plots


# from src.utils import qq_plot

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_csv(
    "../../../data/interim/02_static_dataset_processed_outliers_removed.csv"
)
df = df.drop(columns=["Unnamed: 0"])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

selected_columns = list(df.columns[:13])

for column in selected_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


for col in selected_columns:
    column_name = col
    num_missing_values = df[col].isnull().sum()

    description = (
        f"Column name: {column_name} | Number of missing values: {num_missing_values}\n"
    )

    print(description)


# --------------------------------------------------------------
# Data Reduction
# --------------------------------------------------------------

# Removal of Horizontal Redundancies

df = df.drop_duplicates()


# Removal of Correlated Attributes
df_without_fertility = df.drop(columns=["Fertility"])
correlation_matrix = df_without_fertility.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

df_without_fertility = df_without_fertility.drop(columns=["OC"])


# --------------------------------------------------------------
# Data Normalization
# --------------------------------------------------------------
def qq_plot(df):
    num_cols = len(df.columns)
    num_rows = int(
        np.ceil(num_cols / 3)
    )  # Adjust the number of columns per row as needed

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 3 * num_rows))
    fig.suptitle("QQ Plots for Columns ", y=1.02)

    # Iterate over each pair of columns and create QQ plots
    for i, (col1, col2) in enumerate(zip(df.columns[:-1:2], df.columns[1::2])):
        ax = axes[i // 3, i % 3]
        sm.qqplot_2samples(df[col1], df[col2], ax=ax, line="45")
        ax.set_title(f"QQ Plot - {col1} vs {col2}")

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Min-Max Normalization

minMax_df = (df_without_fertility - df_without_fertility.min()) / (
    df_without_fertility.max() - df_without_fertility.min()
)


qq_plot(minMax_df)

# Z-Score Normalization

zscore_df = (
    df_without_fertility - df_without_fertility.mean()
) / df_without_fertility.std()

qq_plot(zscore_df)
