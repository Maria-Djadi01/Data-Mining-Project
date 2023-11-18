import sys

# Specify the directory where your data is located
project_dir = "D:/2M/D.Mining/Data-Mining-Project/"

# Change the working directory
sys.path.append(project_dir)

import pandas as pd
from src.utils import central_tendances


# --------------------------------------------------------------
# Read  Static file
# --------------------------------------------------------------

file_id = "1zcNeHEEoi9XmpPYcUKTqRB_VngeQGZEK"

download_link = f"https://drive.google.com/uc?id={file_id}"

static_dataset_df = pd.read_csv(download_link)
static_dataset_df.to_csv("../../data/raw/static_dataset.csv")

# Convert the P column to numeric type
static_dataset_df["P"] = pd.to_numeric(static_dataset_df["P"], errors="coerce")

static_dataset_df.to_csv("../../data/interim/processed_static_dataset.csv")


# --------------------------------------------------------------
# Read  Time Series file
# --------------------------------------------------------------

file_id = "1CQMBEUzL8g39Cpost4D5_6a_mraOlFNP"

# download_link = f"https://drive.google.com/uc?id={file_id}"

# tempo_dataset_df_processed = pd.read_csv(download_link)
# tempo_dataset_df_processed.to_csv("../../data/raw/temp_dataset.csv")

tempo_dataset_df_processed = pd.read_csv("../../data/raw/temp_dataset.csv")

# convert the start date and end date objects to date object, by exploring the
# whole df we noticed two types of dates. The first one is like '10/31/2020'
# and the second one is like '12-Jan'. We need to convert both of them to date
# object.

# The issue with the second format is that it doesn't have the year, so we need
# to add it.

df_time_fixed = tempo_dataset_df_processed.copy()
# Filter rows with and without a '/' in the 'Start date' column
filtered_df_with_year = df_time_fixed[df_time_fixed["Start date"].str.contains("/")]
filtered_df_without_year = df_time_fixed[
    ~df_time_fixed["Start date"].str.contains("/")
][["Start date", "end date", "time_period"]]

# Convert the 'Start date' column to datetime format using .loc
filtered_df_with_year.loc[:, "year"] = pd.to_datetime(
    filtered_df_with_year["Start date"]
).dt.year

# Select the relevant columns
start_date_column_with_year = filtered_df_with_year[
    ["year", "Start date", "end date", "time_period"]
]

for index, row in filtered_df_without_year.iterrows():
    if row["time_period"] not in start_date_column_with_year["time_period"].values:
        df_time_fixed.drop(index, inplace=True)
        continue
    time_period = row["time_period"]
    year = start_date_column_with_year.loc[
        start_date_column_with_year["time_period"] == time_period, "year"
    ].iloc[0]

    df_time_fixed["Start date"].loc[index] = pd.to_datetime(
        df_time_fixed["Start date"].loc[index] + "-" + str(int(year))
    )
    df_time_fixed["end date"].loc[index] = pd.to_datetime(
        df_time_fixed["end date"].loc[index] + "-" + str(int(year))
    )

for index, row in filtered_df_with_year.iterrows():
    df_time_fixed["Start date"].loc[index] = pd.to_datetime(
        df_time_fixed["Start date"].loc[index]
    )
    df_time_fixed["end date"].loc[index] = pd.to_datetime(
        df_time_fixed["end date"].loc[index]
    )

tempo_dataset_df_processed = df_time_fixed

column_order = [
    "zcta",
    "time_period",
    "Start date",
    "end date",
    "population",
    "test count",
    "positive tests",
    "case count",
    "test rate",
    "case rate",
    "positivity rate",
]

# To improve data readability, we will reorder columns and the rows according
# to Start date and the state zip code
tempo_dataset_df_processed = tempo_dataset_df_processed[column_order].sort_values(
    by=["zcta", "Start date"]
)
# turn zcta into object

tempo_dataset_df_processed.to_csv("../../data/interim/temp_dataset_processed.csv")
