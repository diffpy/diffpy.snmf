import numpy as np
import scipy.optimize


def lsqnonneg(stretched_component_matrix, target_signal):
    """Finds the weights of stretched component signals under one-sided constraint.

    Solves ``argmin_x || Ax - b ||_2`` for ``x>=0`` where A is the stretched_component_matrix and b is the
    target_signal vector. Finds the weights of component signals given undecomposed signal data and stretched
    components under a one-sided constraint on the weights.

    Parameters
    ----------
    stretched_component_matrix: 2d array like
      The component matrix where each column contains a stretched component signal. Has dimensions R x C where R is
      the length of the signal and C is the number of components. Does not need to be nonnegative. Corresponds with
      'A' from the objective function.

    target_signal: 1d array like
      The signal that is used as reference against which weight factors will be determined. Any column from the
      matrix of the entire, unfactorized input data could be used. Has length R. Does not need to be nonnegative.
      Corresponds with 'b' from the objective function.

    Returns
    -------
    1d array like
      The vector containing component signal weights at a moment. Has length C.
    """
    stretched_component_matrix = np.asarray(stretched_component_matrix)
    target_signal = np.asarray(target_signal)
    return scipy.optimize.nnls(stretched_component_matrix, target_signal)[0]
