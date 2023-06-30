import numpy as np
import scipy.sparse


def get_variables(data_input, component_amount, data_type, sparsity=1, smoothness=1e18):
    """Determines the variables and initial values used in the SNMF algorithm.

    Parameters
    ----------
    data_input: 2d array like
      The observed or simulated PDF or XRD data provided by the user. Has dimensions R x N where R is the signal length
      and N is the number of PDF/XRD signals.

    component_amount: int
      The number of component signals the user would like to decompose 'data_input' into.

    data_type: str
      The type of data the user has passed into the program. Can assume the value of 'PDF' or 'XRD.'

    sparsity: float, optional
      The regularization parameter that behaves as the coefficient of a "sparseness" regularization term that enhances
      the ability to decompose signals in the case of sparse data e.g. X-ray Diffraction data. A non-zero value
      indicates sparsity in the data; greater magnitudes indicate greater amounts of sparsity.

    smoothness: float, optional
      The regularization parameter that behaves as the coefficient of a "smoothness" term that ensures that component
      signal weightings change smoothly with time. Assumes a default value of 1e18.

    Returns
    -------
    dictionary
      The collection of the names and values of the constants used in the algorithm. Contains the number of observed PDF
      /XRD patterns, the length of each pattern, the type of the data, the number of components the user would like to
      decompose the data into, an initial guess for the component matrix, and initial guess for the weight factor matrix
      ,an initial guess for the stretching factor matrix, a parameter controlling smoothness of the solution, a
      parameter controlling sparseness of the solution,

    """
    signal_length = data_input.shape[0]
    moment_amount = data_input.shape[1]

    component_matrix_guess = np.random.rand(signal_length, component_amount)
    weight_matrix_guess = np.random.rand(component_amount, moment_amount)
    stretching_matrix_guess = np.ones(component_amount, moment_amount) + np.random.randn(component_amount,
                                                                                         moment_amount) * 1e-3

    diagonals = [np.ones(moment_amount-2),-2*np.ones(moment_amount-2),np.ones(moment_amount-2)]
    sparsity_term = .25 * scipy.sparse.diags(diagonals,[0,1,2], shape=(moment_amount-2,moment_amount))

    return {
        "signal_length": signal_length,
        "moment_amount": moment_amount,
        "component_matrix_guess": component_matrix_guess,
        "weight_matrix_guess": weight_matrix_guess,
        "stretching_matrix_guess": stretching_matrix_guess,
        "component_amount": component_amount,
        "data_type": data_type,
        "smoothness": smoothness,
        "sparsity": sparsity,
        "sparsity_term": sparsity_term

    }
