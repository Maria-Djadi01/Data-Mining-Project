# -*- coding: utf-8 -*-
import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Read  Static file
# --------------------------------------------------------------

file_id = "1zcNeHEEoi9XmpPYcUKTqRB_VngeQGZEK"

download_link = f"https://drive.google.com/uc?id={file_id}"

static_dataset_df = pd.read_csv(download_link)
static_dataset_df.to_csv("../../data/raw/static_dataset.csv")


static_dataset_df["P"] = pd.to_numeric(static_dataset_df["P"], errors="coerce")

static_dataset_df.to_csv("../../data/interim/processed_static_dataset.csv")


# --------------------------------------------------------------
# Read  Time Series file
# --------------------------------------------------------------

file_id = "1CQMBEUzL8g39Cpost4D5_6a_mraOlFNP"

download_link = f"https://drive.google.com/uc?id={file_id}"

tempo_dataset_df = pd.read_csv(download_link)
tempo_dataset_df.to_csv("../../data/raw/temp_dataset.csv")
