import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_csv("../../../data/interim/.csv")
df = df.drop(columns=["Unnamed: 0"])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df = df.dropna()

# --------------------------------------------------------------
# Data Reduction
# --------------------------------------------------------------


# Removal of Horizontal Redundancies

df = df.dropduplicates()
# Removal of Vertical Redundancies

df = df.loc[:, (df != df.iloc[0]).any()]
# Removal of Correlated Attributes


# --------------------------------------------------------------
# Data Normalization
# --------------------------------------------------------------

# Min-Max Normalization

minMax_df = (df - df.min()) / (df.max() - df.min())

# Z-Score Normalization

zscore_df = (df - df.mean()) / df.std()
