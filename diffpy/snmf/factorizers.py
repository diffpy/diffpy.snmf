import numpy as np
import scipy.optimize
def lsqnonneg(C,d):
    """

    Parameters
    ----------
    C
    d

    Returns
    -------

    """
    C = np.asarray(C)
    d = np.asarray(d)
    return scipy.optimize.nnls(C,d)[0]