import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import sys
import seaborn as sns

sys.path.append("../../../../Data-Mining-Project")

from src.utils import correlation_plots, plot_distribution


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

df_without_fertility = df_without_fertility.drop(columns=["OM"])


# --------------------------------------------------------------
# Data Normalization
# --------------------------------------------------------------


# Min-Max Normalization

minMax_df = (df_without_fertility - df_without_fertility.min()) / (
    df_without_fertility.max() - df_without_fertility.min()
)


plot_distribution(minMax_df)

# Z-Score Normalization

zscore_df = (
    df_without_fertility - df_without_fertility.mean()
) / df_without_fertility.std()

plot_distribution(zscore_df)


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

minMax_df["Fertility"] = df["Fertility"]
minMax_df.to_csv("../../../data/interim/03_static_dataset_features_built.csv")
