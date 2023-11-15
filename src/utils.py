import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    Q1_index = int(0.25 * (nrows + 1))
    Q2_index = int(0.5 * (nrows + 1))
    Q3_index = int(0.75 * (nrows + 1))
    if nrows % 2 == 0:
        Q1 = (sorted_column[Q1_index] + sorted_column[Q1_index + 1]) / 2
        Q2 = (sorted_column[Q2_index] + sorted_column[Q2_index + 1]) / 2
        Q3 = (sorted_column[Q3_index] + sorted_column[Q3_index + 1]) / 2
    else:
        Q1 = sorted_column[Q1_index]
        Q2 = sorted_column[Q2_index]
        Q3 = sorted_column[Q3_index]
    return (min(sorted_column), Q1, Q2, Q3, max(sorted_column))


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
    fig, axs = plt.subplots(int(df.shape[1] / 4) + 1, 4, figsize=(12, 12))

    for i, column in enumerate(df.columns):
        ax = axs[i // 4, i % 4]
        ax.hist(
            df[column],
            # bins=range(int(min(df[column])), int(max(df[column])) + 1),
            edgecolor="black",
        )
        ax.set_title(f"Histogram for {column}")

    plt.tight_layout()
    plt.show()


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


def correlation_plots(static_df):
    """
    Visualize data using scatter plots and a correlation heatmap.

    Parameters:
    - static_df (DataFrame): The input DataFrame containing numerical data.

    This function creates scatter plots for all pairs of numerical columns in the DataFrame
    and displays a correlation heatmap for the entire DataFrame.

    Example:
    >>> import pandas as pd
    >>> # Assuming your DataFrame is named 'static_df'
    >>> visualize_data(static_df)
    """
    sns.set(style="whitegrid")

    sns.pairplot(static_df)
    plt.show()

    correlation_matrix = static_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.show()
