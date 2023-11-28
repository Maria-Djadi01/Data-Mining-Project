import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import seaborn as sns
import math

# Specify the directory where your data is located
project_dir = "../../../../Data-Mining-Project"

# Change the working directory
sys.path.append(project_dir)
from src.utils import equal_width_discretization, equal_frequency_discretization

warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------#
# Load data #
# ----------------------------------------------------------------#

df = pd.read_csv("../../../data/interim/static_dataset3_processed.csv", index_col=0)
df.info()
df.head()

# ---------------------------------------------------------------- #
# Descritization
# ---------------------------------------------------------------- #

cols = ["Temperature", "Humidity", "Rainfall"]
for col in cols:
    plt.figure(figsize=(12, 8))
    plt.title("Original Data Distribution of " + col)
    sns.histplot(df[col], kde=False)
    plt.xlabel("Numeric Column Values")
    plt.ylabel("Frequency")
    plt.show()

# Visualize the impact of different discretization methods
df_disc = df.copy()

for col in cols:
    df_disc[col + "_freq_disc"] = equal_frequency_discretization(df[col])
    df_disc[col + "_width_disc"] = equal_width_discretization(df[col])


# ---------------------------------------------------------------- #
# Compare the distribution of the original and discretized data
# ---------------------------------------------------------------- #
# Set up the figure and axis
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(
    wspace=0.4, hspace=0.6
)  # Adjust the width and height space between subplots

for col in cols:
    # Plot the original data
    sns.histplot(df[col], kde=False, ax=axes[cols.index(col), 0])
    axes[cols.index(col), 0].set_title("Original Distribution")

    # Plot the discretized freq data
    sns.histplot(df_disc[col + "_freq_disc"], kde=False, ax=axes[cols.index(col), 1])
    axes[cols.index(col), 1].set_title("Discretized Distribution (Frequency)")

# Rotate x-axis labels
for ax in axes.flatten():
    ax.tick_params(axis="x", labelrotation=45)

plt.show()
fig.savefig("../../../reports/figures/03_EDA/Discretized_Distribution_Frequency")

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(
    wspace=0.4, hspace=0.6
)  # Adjust the width and height space between subplots

for col in cols:
    # Plot the original data
    sns.histplot(df[col], kde=False, ax=axes[cols.index(col), 0])
    axes[cols.index(col), 0].set_title("Original Distribution")

    # Plot the discretized freq data
    sns.histplot(df_disc[col + "_width_disc"], kde=False, ax=axes[cols.index(col), 1])
    axes[cols.index(col), 1].set_title("Discretized Distribution (Width)")

# Rotate x-axis labels
for ax in axes.flatten():
    ax.tick_params(axis="x", labelrotation=45)

plt.show()
fig.savefig("../../../reports/figures/03_EDA/Discretized_Distribution_Width")
# ---------------------------------------------------------------- #
# Save the dataset
# ---------------------------------------------------------------- #
df_disc.to_csv("../../../data/processed/static_dataset3_discretized.csv")
