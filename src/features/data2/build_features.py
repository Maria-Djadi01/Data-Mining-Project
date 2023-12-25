import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------#
# Load data #
# ----------------------------------------------------------------#

df = pd.read_csv("../../../data/interim/temp_dataset_processed.csv", index_col=0)
df.info()
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# Number of missing values in each column
for col in df.columns:
    print(col, df[col].isnull().sum())

# We can fill the missing values in test count by looking to test rate if we know the population
df_fill_test_count = df.copy()
for row in df_fill_test_count[df_fill_test_count["test count"].isnull()].index:
    df_fill_test_count.loc[row, "test count"] = (
        df_fill_test_count.loc[row, "test rate"]
        * df_fill_test_count.loc[row, "population"]
        / 100
    ).astype(int)
df_fill_test_count["test count"].isnull().sum()

# We can fill the positive tests by looking to test count and positivity rate
df_fill_positive_tests = df_fill_test_count.copy()
for row in df_fill_positive_tests[
    df_fill_positive_tests["positive tests"].isnull()
].index:
    df_fill_positive_tests.loc[row, "positive tests"] = (
        df_fill_positive_tests.loc[row, "test count"]
        * df_fill_positive_tests.loc[row, "positivity rate"]
        / 100
    ).astype(int)
df_fill_positive_tests["positive tests"].isnull().sum()

# We can fill the case count by looking at the case rate and population
df_fill_case_count = df_fill_positive_tests.copy()
for row in df_fill_case_count[df_fill_case_count["case count"].isnull()].index:
    df_fill_case_count.loc[row, "case count"] = (
        df_fill_case_count.loc[row, "population"]
        * df_fill_case_count.loc[row, "case rate"]
        / 100
    ).astype(int)
df_fill_case_count["case count"].isnull().sum()

df_fill_miss = df_fill_case_count
for col in df_fill_miss.columns:
    print(col, df_fill_miss[col].isnull().sum())

# ----------------------------------------------------------------#
# Save file
# ----------------------------------------------------------------#
df_fill_miss.to_csv("../../../data/interim/02_temp_dataset_fill_miss_val.csv")
