import numpy as np


def lift_data(data_input, lift=1):
    """Lifts values of data_input

    Adds 'lift' * the minimum value in data_input to data_input element-wise.

    Parameters
    ----------
    data_input: 2d array like
      The matrix containing a series of signals to be decomposed. Has dimensions N x M where N is the length of each
      signal and M is the number of signals.

    lift: float
      The factor representing how much to lift 'data_input'.

    Returns
    -------
    2d array like
      The matrix that contains data_input - (min(data_input) * lift).

    """
    data_input = np.asarray(data_input)
    return data_input + np.abs(np.min(data_input) * lift)


def objective_function(residual_matrix, stretching_factor_matrix, smoothness, smoothness_term, component_matrix,
                       sparsity):
    """Defines the objective function of the algorithm and returns its value.

    Calculates the value of '(||residual_matrix||_F) ** 2 + smoothness * (||smoothness_term *
    stretching_factor_matrix.T||)**2 + sparsity * sum(component_matrix ** .5)' and returns its value.

    Parameters
    ----------
    residual_matrix: 2d array like
      The matrix where each column is the difference between an experimental PDF/XRD pattern and a calculated PDF/XRD
      pattern at each grid point. Has dimensions R x M where R is the length of each pattern and M is the amount of
      patterns.

    stretching_factor_matrix: 2d array like
      The matrix containing the stretching factors of the calculated component signal. Has dimensions K x M where K is
      the amount of components and M is the number of experimental PDF/XRD patterns.

    smoothness: float
      The coefficient of the smoothness term which determines the intensity of the smoothness term and its behavior.
      It is not very sensitive and is usually adjusted by multiplying it by ten.

    smoothness_term: 2d array like
      The regularization term that ensures that smooth changes in the component stretching signals are favored.
      Has dimensions (M-2) x M where M is the amount of experimentally obtained PDF/XRD patterns, the moment amount.

    component_matrix: 2d array like
      The matrix containing the calculated component signals of the experimental PDF/XRD patterns. Has dimensions R x K
      where R is the signal length and K is the number of component signals.

    sparsity: float
      The parameter determining the intensity of the sparsity regularization term which enables the algorithm to
      exploit the sparse nature of XRD data. It is usually adjusted by doubling.

    Returns
    -------
    float
      The value of the objective function.

    """
    residual_matrix = np.asarray(residual_matrix)
    stretching_factor_matrix = np.asarray(stretching_factor_matrix)
    component_matrix = np.asarray(component_matrix)
    return .5 * np.linalg.norm(residual_matrix, 'fro') ** 2 + .5 * smoothness * np.linalg.norm(
        smoothness_term @ stretching_factor_matrix.T, 'fro') ** 2 + sparsity * np.sum(np.sqrt(component_matrix))


def construct_stretching_matrix(components, number_of_components, number_of_signals):
    """Constructs the stretching factor matrix

    Constructs a K x M matrix where K is the number of components and M is the number of signals. Each element is the
    stretching factor for a specific component signal for a specific signal from the data input.
    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signal objects.
    number_of_components: int
      The number specifying the number of components signals.
    number_of_signals: int
      The number of signals in the data input.

    Returns
    -------
    2d array
      The stretching factor matrix. Has dimensions (`number_of_components`) x (`number_of_signals`)
    """
    stretching_factor_matrix = np.zeros((number_of_components, number_of_signals))
    for c in components:
        stretching_factor_matrix[c.id, :] = c.stretching_factors
    return stretching_factor_matrix


def construct_component_matrix(components, number_of_components, signal_length):
    """Constructs the component matrix

    Constructs an N x K matrix where N is the length of the signals and K is the number of components. Each column
    is a component signal.
    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signal objects.
    number_of_components: int
      The number specifying the number of components signals.
    signal_length: int
      THe number specifying the length of the signals' grid.
    Returns
    -------
    2d array
      The array containing the component signal intensity/g(r) values.
    """
    component_matrix = np.zeros((signal_length, number_of_components))
    for c in components:
        component_matrix[:, c.id] = c.iq
    return component_matrix


def construct_weight_matrix(components, number_of_components, number_of_signals):
    """Constructs the weights matrix

    Constructs a K x M matrix where K is the number of components and M is the number of signals. Each element is the
    stretching factor for a specific weight for a specific signal from the data input.
    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signal objects.
    number_of_components: int
      The number specifying the number of components signals.
    number_of_signals: int
      The number of signals in the data input.
    Returns
    -------
    2d array like
      The 2d array containing the weights of the component signals.
    """
    weights_matrix = np.zeros((number_of_components, number_of_signals))
    for c in components:
        weights_matrix[c.id, :] = c.weights
    return weights_matrix
