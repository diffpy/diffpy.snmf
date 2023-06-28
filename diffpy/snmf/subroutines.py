import numpy as np
from scipy.sparse import csc


def get_constants(data_input, component_amount, data_type, sparsity=1, stretching_factor_smoothing=1e18):
    """Determines the constants and initial values used in the SNMF algorithm.

    Parameters
    ----------
    data_input: 2d array like
      The observed or simulated PDF or XRD data provided by the user. Has dimensions R x N where R is the signal length
      and N is the number of PDF/XRD signals.

    component_amount: int
      The number of component signals the user would like to decompose 'data_input' into.

    data_type: str
      The type of data the user has passed into the program. Can assume the value of 'PDF' or 'XRD.'

    sparsity: int, optional

    stretching_factor_smoothing: int, optional

    Returns
    -------
    dictionary
      The collection of the names and values of the constants used in the algorithm. Contains the number of observed PDF
      /XRD patterns, the length of each pattern, the type of the data, the number of components the user would like to
      decompose the data into, an initial guess for the component matrix, and initial guess for the weight factor matrix
      ,an initial guess for the stretching factor matrix, and ... [not finised]

    """
    signal_length = data_input.shape[0]
    moment_amount = data_input.shape[1]

    component_matrix_guess = np.random.rand(signal_length, component_amount)
    weight_matrix_guess = np.random.rand(component_amount, moment_amount)
    stretching_matrix_guess = np.random.rand(component_amount, moment_amount)

    return {
        "signal_length": signal_length,
        "moment_amount": moment_amount,
        "component_matrix_guess": component_matrix_guess,
        "weight_matrix_guess": weight_matrix_guess,
        "stretching_matrix_guess": stretching_matrix_guess,
        "component_amount": component_amount,
        "data_type": data_type
        "sparsity": sparsity
        "stretching_factor_smoothing": stretching_factor_smoothing

    }
