import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy
import statsmodels.api as sm
from scipy import stats
import math

# --------------------------------------------------------------
# Central Tendencies (distribution based)
# --------------------------------------------------------------


def central_tendances(column):
    """
    Calculate measures of central tendency for a given numerical column.

    Parameters:
    - column (iterable): A list or iterable containing numerical data.

    Returns:
    - mean (float): The arithmetic mean of the values in the column.
    - median (float): The median value in the column.
    - mode (list): A list of mode(s) if they exist; otherwise, an empty list.

    This function computes the mean, median, and mode of a given dataset.
    - Mean is the average of all values.
    - Median is the middle value when the data is sorted.
    - Mode is the most frequently occurring value(s) in the dataset.

    If there are multiple modes, a list of modes is returned.

    Example:
    >>> data = [1, 2, 2, 4, 5, 4]
    >>> central_tendences(data)
    (2.3333333333333335, 3.0, [2, 4])
    """

    # Mean
    sum = 0
    for i in column:
        sum = sum + i
    mean = sum / len(column)

    # Median
    sorted_col = np.sort(column)
    if len(sorted_col) % 2 == 0:
        i = int(len(sorted_col) / 2)
        median = sorted_col[i]
    else:
        i = int(len(sorted_col) / 2)
        median = (sorted_col[i] + sorted_col[i + 1]) / 2

    # Mode
    freq_list = {}
    for x in column:
        if x in freq_list:
            freq_list[x] += 1
        else:
            freq_list[x] = 1
    max_freq = max(freq_list.values())
    mode = [key for key, value in freq_list.items() if value == max_freq]

    return mean, median, mode


# --------------------------------------------------------------
# Quartiles (distribution based)
# --------------------------------------------------------------


def calculate_quartiles(column):
    """
    Calculate quartiles for a given numerical column.

    Parameters:
    - column (iterable): A list or iterable containing numerical data.

    Returns:
    - min_value (float): The minimum value in the column.
    - Q1 (float): The first quartile (25th percentile) of the data.
    - Q2 (float): The second quartile (50th percentile or median) of the data.
    - Q3 (float): The third quartile (75th percentile) of the data.
    - max_value (float): The maximum value in the column.

    This function computes the quartiles of a dataset. Quartiles are statistical measures that divide a dataset into four equal parts, each containing 25% of the data. The quartiles are used to understand the spread and distribution of the data.

    Example:
    >>> data = [10, 15, 20, 25, 30, 35]
    >>> calculate_quartiles(data)
    (10, 17.5, 25.0, 32.5, 35)
    """

    nrows = len(column)
    sorted_column = np.sort(column)
    Q1_index = int(0.25 * (nrows - 1))
    Q2_index = int(0.5 * (nrows - 1))
    Q3_index = int(0.75 * (nrows - 1))
    if nrows % 2 == 0:
        Q1 = (sorted_column[Q1_index] + sorted_column[Q1_index + 1]) / 2
        Q2 = (sorted_column[Q2_index] + sorted_column[Q2_index + 1]) / 2
        Q3 = (sorted_column[Q3_index] + sorted_column[Q3_index + 1]) / 2
    else:
        Q1 = sorted_column[Q1_index]
        Q2 = sorted_column[Q2_index]
        Q3 = sorted_column[Q3_index]
    return (min(sorted_column), Q1, Q2, Q3, max(sorted_column))


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------


# TODO: Understand the function of chauvenets criterion
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


# --------------------------------------------------------------
# Histogram Plots
# --------------------------------------------------------------


def histogram_plot(df):
    """
    Generate a grid of histogram plots for each column in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing numerical data.

    This function creates a grid of histogram plots, with each subplot corresponding to a column in the DataFrame. It uses Matplotlib to visualize the distribution of numerical data in each column.

    The number of rows in the subplot grid is determined based on the number of columns in the DataFrame, and the figure size is set to (12, 12) by default. The function automatically calculates the appropriate number of bins for each histogram based on the data's range.

    Example:
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt

    >>> # Create a sample DataFrame
    >>> data = {'A': [10, 15, 20, 25, 30],
    ...         'B': [5, 10, 15, 20, 25],
    ...         'C': [15, 20, 25, 30, 35]}
    >>> df = pd.DataFrame(data)

    >>> # Plot histograms for each column in the DataFrame
    >>> plot_histograms(df)

    Note: You need to have Matplotlib and Pandas installed to use this function.
    """
    fig, axs = plt.subplots(int(df.shape[1] / 3) + 1, 3, figsize=(12, 12))

    for i, column in enumerate(df.columns):
        ax = axs[i // 3, i % 3]
        ax.hist(
            df[column],
            # bins=range(int(min(df[column])), int(max(df[column])) + 1),
            edgecolor="black",
        )
        ax.set_title(f"Histogram for {column}")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Bar plots
# --------------------------------------------------------------


def bar_plot(df):
    fig, axs = plt.subplots(int(df.shape[1] / 4) + 1, 4, figsize=(12, 12))

    for i, column in enumerate(df.columns):
        ax = axs[i // 4, i % 4]
        ax.bar(
            df[column],
            # bins=range(int(min(df[column])), int(max(df[column])) + 1),
            edgecolor="black",
        )
        ax.set_title(f"Histogram for {column}")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Box plots
# --------------------------------------------------------------


def box_plot(df):
    fig, axs = plt.subplots(int(df.shape[1] / 4) + 1, 4, figsize=(12, 12))

    for i, column in enumerate(df.columns):
        ax = axs[i // 4, i % 4]

        # Filter out nan values before plotting
        data_to_plot = df[column].dropna().values

        ax.boxplot(
            data_to_plot,
        )
        ax.set_title(f"Box Plot for {column}")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Correlation plots
# --------------------------------------------------------------


def correlation_plots(df):
    """
    Visualize data using scatter plots and a correlation heatmap.

    Parameters:
    - df (DataFrame): The input DataFrame containing numerical data.

    This function creates scatter plots for all pairs of numerical columns in the DataFrame
    and displays a correlation heatmap for the entire DataFrame.

    Example:
    >>> import pandas as pd
    >>> # Assuming your DataFrame is named 'df'
    >>> visualize_data(df)
    """
    sns.set(style="whitegrid")

    sns.pairplot(df)
    plt.show()

    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.show()


# --------------------------------------------------------------
# Quartiles Plots
# --------------------------------------------------------------


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


# Q-Q Plot
def qq_plot(df):
    fig, axes = plt.subplots(
        nrows=len(df.columns),
        figsize=(8, 2 * len(df.columns)),
    )

    for i, column in enumerate(df.columns):
        ax = axes[i]
        sm.qqplot(df[column], line="45", ax=ax)
        ax.set_title(f"QQ Plot - {column}")

    plt.tight_layout()
    plt.show()

def class_discretization(column):
    """
    Discretizes a numerical column into classes with means as labels.

    Args:
    - column (pandas Series): Numerical column to be discretized.

    Returns:
    - pd.Series: Discretized column with labels.

    Example:
    ```python
    result = class_discretization(data['numeric_column'])
    ```

    Dependencies: math, pandas
    """
    k = int((1 + 10/3) * math.log(len(column), 10))
    print(f'k = {k}')

    width = (max(column) - min(column)) / k

    bins = [min(column) + i * width for i in range(k+1)]
    return pd.cut(column, bins, labels=[f'{((bins[i] + bins[i+1]) / 2):.1f}' for i in range(k)])

def equal_frequency_discretization(column):
    num_classes = int((1 + 10/3) * math.log(len(column), 10))  
    
    # Perform equal-frequency discretization
    discretized = pd.qcut(column, q=num_classes, labels=False, duplicates='drop')

    # Get the bin edges
    _, bin_edges = pd.qcut(column, q=num_classes, retbins=True, duplicates='drop')

    # Create custom labels based on the bin edges
    labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]

    # Assign the custom labels to the discretized column
    discretized_with_labels = discretized.map(dict(enumerate(labels)))

    return discretized_with_labels

def equal_width_discretization(column):
    num_classes = int((1 + 10/3) * math.log(len(column), 10))
    # Perform equal-width discretization
    discretized = pd.cut(column, bins=num_classes, labels=False, duplicates='drop')

    # Get the bin edges
    _, bin_edges = pd.cut(column, bins=num_classes, retbins=True, duplicates='drop')

    # Create custom labels based on the bin edges
    labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]

    # Assign the custom labels to the discretized column
    discretized_with_labels = discretized.map(dict(enumerate(labels)))

    return discretized_with_labels