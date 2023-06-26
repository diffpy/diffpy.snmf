import numpy as np
import scipy.optimize


def lsqnonneg(coefficient, target):
    """
    Solves ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    Parameters
    ----------
    coefficient: 2d array like
        A 2-dimensional array like object representing the coefficient matrix 'A' of the least square problem.
        In context, A represents a matrix of the stretched components given by the getAfun function at a certain moment.
        Each column corresponds with an individual stretched component. The number of rows is the number of independent
        variable values. The matrix does not need to be nonnegative.

    target: 1d array like
        A 1d array like vector representing the target vector 'b' which contains a series of unstretched PDF/XRD values.
        The array does not need to be nonnegative. Contains only row elements, shape(N,).

    Returns
    -------
    1d array like
        The solution vector (x) to the least squares problem

    Raises
    ------
    ValueError
        If the coefficient or target matrices are not the correct shape

    """
    coefficient = np.asarray(coefficient)
    target = np.asarray(target)
    return scipy.optimize.nnls(coefficient, target)[0]
