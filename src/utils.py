import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split

# import statsmodels.api as sm
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

    Min, Q1, Q2, Q3, Max = calculate_quartiles(dataset[col])
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_fence) | (
        dataset[col] > upper_fence
    )

    return dataset


# --------------------------------------------------------------
# Marking Outliers
# --------------------------------------------------------------


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

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
# Central Distribution Plots
# --------------------------------------------------------------
def plot_distribution(df):
    """
    Generate distribution plots for all columns in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.

    This function creates distribution plots using Seaborn for all columns in the DataFrame.
    """
    # Set the style
    sns.set(style="whitegrid")

    # Determine the number of rows and columns for subplots
    num_rows = len(df.columns) // 4
    num_cols = min(4, len(df.columns))

    # Create subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 4 * num_rows))

    # Plot distribution plots
    for i, column in enumerate(df.columns):
        ax = axs[i // num_cols, i % num_cols]
        sns.histplot(data=df, x=column, ax=ax, kde=True)
        ax.set_title(f"Distribution of {column}")

    # Adjust layout
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Class Discretization
# --------------------------------------------------------------


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
    k = int((1 + 10 / 3) * math.log(len(column), 10))
    print(f"k = {k}")

    width = (max(column) - min(column)) / k

    bins = [min(column) + i * width for i in range(k + 1)]
    return pd.cut(
        column, bins, labels=[f"{((bins[i] + bins[i+1]) / 2):.1f}" for i in range(k)]
    )


# --------------------------------------------------------------
# Equal-Frequency Discretization
# --------------------------------------------------------------


def equal_frequency_discretization(column):
    num_classes = int((1 + 10 / 3) * math.log(len(column), 10))

    # Perform equal-frequency discretization
    discretized = pd.qcut(column, q=num_classes, labels=False, duplicates="drop")

    # Get the bin edges
    _, bin_edges = pd.qcut(column, q=num_classes, retbins=True, duplicates="drop")

    # Create custom labels based on the bin edges
    # labels = [f"{column.name}_class_{i}" for i in range(len(bin_edges) - 1)]
    labels = [
        f"{column.name} [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]"
        for i in range(len(bin_edges) - 1)
    ]

    # Assign the custom labels to the discretized column
    discretized_with_labels = discretized.map(dict(enumerate(labels)))

    return discretized_with_labels


# --------------------------------------------------------------
# Equal-Width Discretization
# --------------------------------------------------------------


def equal_width_discretization(column):
    num_classes = int((1 + 10 / 3) * math.log(len(column), 10))
    # Perform equal-width discretization
    discretized = pd.cut(column, bins=num_classes, labels=False, duplicates="drop")

    # Get the bin edges
    _, bin_edges = pd.cut(column, bins=num_classes, retbins=True, duplicates="drop")

    # Create custom labels based on the bin edges
    # labels = [f"{column.name}_class_{i}" for i in range(len(bin_edges) - 1)]
    labels = [
        f"{column.name} [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]"
        for i in range(len(bin_edges) - 1)
    ]

    # Assign the custom labels to the discretized column
    discretized_with_labels = discretized.map(dict(enumerate(labels)))

    return discretized_with_labels


# --------------------------------------------------------------
# Apriori Plots
# --------------------------------------------------------------


def plot_apriori_results(experiment_results, min_supp_range, min_conf_range):
    """
    Plot line charts for Min_Supp versus Frequent_Items_Count, Min_Supp versus Rules_Count,
    and Min_Conf versus Rules_Count.

    Parameters:
    - experiment_results (list): List of dictionaries containing experiment results.
    """
    sns.set(style="whitegrid")

    min_supp_values = [result["Min_Supp"] for result in experiment_results]
    frequent_items_count_values = sorted(
        set(result["Total_L_Count"] for result in experiment_results)
    )
    result_count_values = sorted(
        set(result["Result_Count"] for result in experiment_results)
    )

    # Create a single figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))

    # Plot line chart for Min_Supp versus Frequent_Items_Count
    sns.lineplot(
        x=min_supp_values,
        y=frequent_items_count_values,
        marker="o",
        ax=ax1,
        label="Frequent Items Count",
    )
    ax1.set_xlabel("Min_Supp")
    ax1.set_ylabel("Frequent_Items_Count")
    ax1.set_title("Min_Supp vs Frequent_Items_Count")
    ax1.legend()

    # Plot line chart for Min_Supp versus Rules_Count
    sns.lineplot(
        x=min_supp_values,
        y=result_count_values,
        marker="o",
        ax=ax2,
        label="Rules Count",
    )
    ax2.set_xlabel("Min_Supp")
    ax2.set_ylabel("Rules_Count")
    ax2.set_title("Min_Supp vs Rules_Count")
    ax2.legend()


def minConf_plot(experiment_results, min_supp_range, min_conf_range):
    # Plot line chart for Min_Conf versus

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    # min_conf_values = sorted(set(result["Min_Conf"] for result in experiment_results))
    for min_supp_value in min_supp_range:
        result_subset = [
            result
            for result in experiment_results
            if result["Min_Supp"] == min_supp_value
        ]
        result_subset = sorted(result_subset, key=lambda x: x["Min_Conf"])

        conf_values = [result["Min_Conf"] for result in result_subset]
        result_count_values = [result["Result_Count"] for result in result_subset]

        # Plot the line graph for the current Min_Supp value
        ax.plot(
            conf_values,
            result_count_values,
            label=f"Min_Supp={min_supp_value}",
            marker="o",
        )

    # Add labels and title
    ax.set_xlabel("Min_Conf")
    ax.set_ylabel("Result_Count")
    ax.set_title("Result_Count vs Min_Conf (grouped by Min_Supp)")
    ax.legend(title="Min_Supp", loc="upper left", bbox_to_anchor=(1, 1))

    # Show the plot
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Split Data
# --------------------------------------------------------------


def split_data(df):
    """
    Split a DataFrame into training and test sets.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    - train_df (DataFrame): The training set.
    - test_df (DataFrame): The test set.

    This function splits a DataFrame into training and test sets using the specified test size. The default test size is 0.2 (20% of the dataset).

    Example:
    >>> import pandas as pd
    >>> # Assuming your DataFrame is named 'df'
    >>> train_df, test_df = split_data(df)
    """
    X = df.drop(columns=["Fertility"]).values
    y = df["Fertility"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------#
# Metrics
# ----------------------------------------------------------------#


# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes))
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


# Accuracy
def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.diag(cm)) / np.sum(cm)


# Recall
def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis=1)


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis=0)


# f1_score
def f1_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = np.sum(cm, axis=0) - np.diag(cm)
    tn = np.sum(cm) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))

    return fp / (fp + tn)


def specifity(f1_score):
    return 1 - f1_score


def compute_metrics(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    rec = recall(y_test, y_pred)
    prec = precision(y_test, y_pred)
    fpr = f1_score(y_test, y_pred)
    spe = specifity(fpr)
    # print("Confusion Matrix: \n", conf_mat)
    # print("Accuracy: ", acc.mean())
    # print("Recall: ", rec.mean())
    # print("Precision: ", prec.mean())
    # print("F1 score: ", fpr.mean())
    # print("Specifity: ", spe.mean())
    return {
        "accuracy": acc.mean(),
        "precision": prec.mean(),
        "recall": rec.mean(),
        "f1_score": fpr.mean(),
        "specificity": spe.mean(),
    }


# ----------------------------------------------------------------#
# Plot Confusion Matrix
# ----------------------------------------------------------------#


def plot_confusion_matrix(conf_mat):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
    fig.suptitle("Confusion matrix", c="b")
    sns.heatmap(
        conf_mat / np.sum(conf_mat), ax=axes[0], annot=True, fmt=".2%", cmap="Blues"
    )
    axes[0].set_xlabel("Predicted labels")
    axes[0].set_ylabel("Actual labels")

    sns.heatmap(conf_mat, ax=axes[1], annot=True, cmap="Blues", fmt="")
    axes[1].set_xlabel("Predicted labels")
    axes[1].set_ylabel("Actual labels")
    plt.show()

# ----------------------------------------------------------------
# Silhouette Score
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Silhouette Score
# ----------------------------------------------------------------
# silhouette_score from scratch
def silhouette_score(X, labels):
    X = np.array(X)
    n_samples = X.shape[0]
    silhouette_values = np.zeros(n_samples)

    for i in range(n_samples):
        a_i = silhouette_a(i, labels, X)
        b_i = silhouette_b(i, labels, X)
        silhouette_values[i] = silhouette(a_i, b_i)

    # return the mean silhouette score
    return np.mean(silhouette_values)

def silhouette_a(sample_idx, labels, X):
    cluster_idx = labels[sample_idx]
    cluster_points = np.where(labels == cluster_idx)[0]
    cluster_points = np.delete(cluster_points, np.where(cluster_points == sample_idx))
    a_i = np.mean([euclidean_distance(X[sample_idx], X[i]) for i in cluster_points])
    return a_i

def silhouette_b(sample_idx, labels, X):
    b_i = []
    for cluster_idx in np.unique(labels):
        if cluster_idx != labels[sample_idx]:
            cluster_points = np.where(labels == cluster_idx)[0]
            b_i.append(np.mean([euclidean_distance(X[sample_idx], X[i]) for i in cluster_points]))
    return min(b_i) if b_i else 0

def silhouette(a_i, b_i):
    return (b_i - a_i) / max(a_i, b_i)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))