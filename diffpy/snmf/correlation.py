import numpy as np


def pearson(data1, data2):
    """
    Returns the pearson correlation coefficient between two data PDF/XRD patterns

    Parameters
    ----------
    data1: 1d array like
        An array/list of the values of a XRD or PDF pattern
    data2: 1d array like
        An array/list of the values of a XRD or PDF pattern

    Returns
    -------
    float
        The pearson correlation coefficient between data1 and data2

    Raises
    ------
    ValueError
        If data1 and data2 are not the same shape

    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    pearson_coefficient = np.corrcoef(data1, data2)[0, 1]
    return pearson_coefficient
