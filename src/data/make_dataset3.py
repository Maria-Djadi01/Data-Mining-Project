# import sys
# project_dir = "D:/2M/D.Mining/Data-Mining-Project/"
# sys.path.append(project_dir)

import pandas as pd
import numpy as np
from src.utils import equal_frequency_discretization, equal_width_discretization


# --------------------------------------------------------------
# Read  Static file
# --------------------------------------------------------------

sheet_id = "1DDUltRw8dDUIuNUTaSxdChr4Xdi0FMyV/edit#gid=750806171"
sheet_name = "Dataset3"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

static_dataset3_df = pd.read_csv(url)
static_dataset3_df.to_csv("../../data/raw/static_dataset3.csv")

# --------------------------------------------------------------
# Read the dataset
# --------------------------------------------------------------
df = pd.read_csv("../../data/raw/static_dataset3.csv")
df.info()
df.head()

# change the columns type to float
# Remove commas and change the columns type to float
df["Temperature"] = df["Temperature"].str.replace(',', '.').astype(float)
df["Humidity"] = df["Humidity"].str.replace(',', '.').astype(float)
df["Rainfall"] = df["Rainfall"].str.replace(',', '.').astype(float)

# view final df
df.info()
df.head()

# ---------------------------------------------------------------- #
# Save the dataset
# ---------------------------------------------------------------- #
df.to_csv("../../data/interim/static_dataset3_processed.csv")