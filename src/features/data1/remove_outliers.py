import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------

df = pd.read_csv("../../../data/interim/processed_static_dataset.csv")
df = df.drop(columns=["Unnamed: 0"])


# ---------------------------------------------------------------
# Plotting outliers
# ---------------------------------------------------------------

for column in df.columns:
    if column not in ["Fertility"]:
        df[[column]].boxplot(figsize=(20, 10))
        plt.title(f"Boxplot for {column}")
        plt.show()
