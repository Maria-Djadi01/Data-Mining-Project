import numpy as np


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
