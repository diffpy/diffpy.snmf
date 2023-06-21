import numpy as np


def pearson(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    data = np.asarray(data)
    x = data[:, 0]
    y = data[:, 1]
    correlation_matrix = np.corrcoef(x, y)
    return correlation_matrix[0, 1]
