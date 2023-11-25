import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import seaborn as sns

# Specify the directory where your data is located
project_dir = "D:/2M/D.Mining/Data-Mining-Project/"

# Change the working directory
sys.path.append(project_dir)
from src.utils import  equal_width_discretization, equal_frequency_discretization

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
num_classes_freq = 10
num_classes_width = 10

df_disc = df.copy()

for col in cols:
    df_disc[col + "_freq_disc"] = equal_frequency_discretization(df[col], num_classes_freq)
    df_disc[col + "_width_disc"] = equal_width_discretization(df[col], num_classes_width)


# ---------------------------------------------------------------- #
# Save the dataset
# ---------------------------------------------------------------- #
df_disc.to_csv("../../../data/processed/static_dataset3_discretized.csv")



