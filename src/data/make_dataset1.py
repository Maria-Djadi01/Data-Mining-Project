import sys

project_dir = "../../../Data-Mining-Project"

# Change the working directory
sys.path.append(project_dir)

import pandas as pd


# --------------------------------------------------------------
# Read  Static file
# --------------------------------------------------------------

file_id = "1zcNeHEEoi9XmpPYcUKTqRB_VngeQGZEK"

download_link = f"https://drive.google.com/uc?id={file_id}"

static_dataset_df = pd.read_csv(download_link)
static_dataset_df.to_csv("../../data/raw/static_dataset.csv")
