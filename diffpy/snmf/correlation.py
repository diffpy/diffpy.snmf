import numpy as np


def pearson(data1, data2):
    """
    Returns the pearson correlation coefficient between two data PDF/XRD patterns

    Parameters
    ----------
    data1: ndarray
        An array of the values of a XRD or PDF pattern
    data2: ndarray
        An array of the values of a XRD or PDF pattern
    Returns
    -------
    float
        The pearson correlation coefficient between data1 and data2

    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    correlation_matrix = np.corrcoef(data1, data2)
    return correlation_matrix[0, 1]
