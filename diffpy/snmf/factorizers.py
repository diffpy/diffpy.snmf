import numpy as np
import scipy.optimize


def lsqnonneg(coefficient, target):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    Parameters
    ----------
    coefficient: ndarray or list of lists
        The coefficient matrix of the least squares problem
    target: ndarray or list of lists
        The target vector for the least squares problem

    Returns
    -------
    ndarray
        The solution vector to the least squares problem
    """
    coefficient = np.asarray(coefficient)
    target = np.asarray(target)
    return scipy.optimize.nnls(coefficient, target)[0]
