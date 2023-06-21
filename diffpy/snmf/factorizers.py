import numpy as np
import scipy.optimize
def lsqnonneg(c, d):
    """

    Parameters
    ----------
    c
    d

    Returns
    -------

    """
    c = np.asarray(c)
    d = np.asarray(d)
    return scipy.optimize.nnls(c, d)[0]