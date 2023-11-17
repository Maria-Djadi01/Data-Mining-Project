import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------#
# Load data #
# ----------------------------------------------------------------#

df = pd.read_csv("../../../data/interim/temp_dataset_processed.csv", index_col=0)
# df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)

# --------------------------------------------------------------#
# Plotting outliers
# --------------------------------------------------------------#
selected_columns = [
    col
    for col in df.columns
    if col not in ["zcta", "Start date", "end date", "time_period", "population"]
]
df_selected_columns = df[selected_columns]
for column in df.columns:
    if column not in ["zcta", "Start date", "end date", "time_period", "population"]:
        df[[column, "zcta"]].boxplot(by="zcta", figsize=(20, 10))
        plt.title(f"Boxplot for {column}")
        plt.show()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["no outlier " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------


# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_fence) | (
        dataset[col] > upper_fence
    )

    return dataset


# Loop over all columns
for col in df_selected_columns.columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", True)

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------


# TODO: Understand the function of chauvenets
# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i, (index, row) in enumerate(dataset.iterrows()):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[index]) - scipy.special.erf(low[index]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Loop over all columns
for col in selected_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", True)


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------
# test on a single column
dataset = mark_outliers_chauvenet(df, "case count")
# number of missing values
dataset.isnull().sum()
dataset[dataset['case count' + "_outlier"] == True] = np.nan
# drop the null raws
dataset = dataset.dropna(how='all')
dataset.drop(columns=['case count' + "_outlier"], inplace=True)

# Create a loop
dataset = df.copy()
for col in selected_columns:
    dataset = mark_outliers_chauvenet(dataset, col)
    dataset[dataset[col + "_outlier"] == True] = np.nan
    dataset = dataset.dropna(how='all')
    n_outliers = len(dataset) - len(dataset[col].dropna(how='all'))
    print(f'{col} - {n_outliers} outliers removed')

outliers_removed_df = dataset
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_csv(
    "../../../data/interim/02_temp_dataset_processed_outliers_removed.csv"
)
