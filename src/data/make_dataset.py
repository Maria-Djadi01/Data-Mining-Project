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

# download_link = f"https://drive.google.com/uc?id={file_id}"

# static_dataset_df = pd.read_csv(download_link)
# static_dataset_df.to_csv("../../data/raw/static_dataset.csv")


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
# to add it. We will convert the start date of the first format and pick the
# most frequent year and use it to fill the year of the second format.

# create a seperate column that contains the "%m/%d/%Y" format
start_date_column = pd.DataFrame()
for idx, date in enumerate(tempo_dataset_df_processed["Start date"]):
    if "/" in date:
        start_date_column.at[idx, "Start date"] = pd.to_datetime(
            date, format="%m/%d/%Y"
        )

# pick the most frequent year
year_mode = central_tendances(start_date_column["Start date"].dt.year)[2]
# Convert 'Start date' column based on date format
for idx, date in enumerate(tempo_dataset_df_processed["Start date"]):
    if "/" in date:
        tempo_dataset_df_processed.at[idx, "Start date"] = pd.to_datetime(
            date, format="%m/%d/%Y"
        )
    else:
        tempo_dataset_df_processed.at[idx, "Start date"] = pd.to_datetime(
            date + "-2021", format="%d-%b-%Y"
        )

# Repeat the same process for 'End date' column if needed
for idx, date in enumerate(tempo_dataset_df_processed["end date"]):
    if "/" in date:
        tempo_dataset_df_processed.at[idx, "end date"] = pd.to_datetime(
            date, format="%m/%d/%Y"
        )
    else:
        tempo_dataset_df_processed.at[idx, "end date"] = pd.to_datetime(
            date + "-2021", format="%d-%b-%Y"
        )

column_order = [
    "zcta",
    "Start date",
    "end date",
    "time_period",
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
df = tempo_dataset_df_processed[column_order].sort_values(by=["zcta", "Start date"])
# turn zcta into object

tempo_dataset_df_processed.to_csv("../../data/interim/temp_dataset_processed.csv")
