import numpy as np
import scipy.optimize


def lsqnonneg(coefficient, target):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    Parameters
    ----------
    coefficient: 2-dimensional ndarray or list
        The coefficient matrix of the least squares problem
    target: 1-dimensional ndarray or list
        The target vector for the least squares problem

    Returns
    -------
    1-dimensional ndarray
        The solution vector to the least squares problem
    """
    coefficient = np.asarray(coefficient)
    target = np.asarray(target)
    return scipy.optimize.nnls(coefficient, target)[0]
