import numpy as np
import scipy.optimize


def lsqnonneg(coefficient, target):
    """

    Parameters
    ----------
    coefficient: ndarray
        The coefficient matrix of the least squares problem
    target: ndarray
        The target vector for the least squares problem

    Returns
    -------
    ndarray
        The solution to the least squares problem
    """
    coefficient = np.asarray(coefficient)
    target = np.asarray(target)
    return scipy.optimize.nnls(coefficient, target)[0]
