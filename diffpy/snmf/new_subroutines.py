import numpy as np
import scipy.optimize
from scipy.sparse import spdiags
from diffpy.snmf.optimizers import get_weights
from diffpy.snmf.datapoint import ComponentSignal


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

    Returns
    -------
    2d array
      The stretching factor matrix

    """
    stretching_factor_matrix = np.zeros((number_of_components, number_of_signals))
    for c in components:
        stretching_factor_matrix[c.id, :] = c.stretching_factors
    return stretching_factor_matrix


def construct_component_matrix(components, number_of_components, signal_length, ):
    """Constructs the component matrix

    Constructs an N x K matrix where N is the length of the signals and K is the number of components. Each column
    is a component signal.

    Parameters
    ----------
    components: tuple of ComponentSignal objects

    Returns
    -------
    2d array

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

    Returns
    -------
    2d array like

    """
    weights_matrix = np.zeros((number_of_components, number_of_signals))
    for c in components:
        weights_matrix[c.id, :] = c.weights
    return weights_matrix


def update_weights(components, method, data_input, signal_length, number_of_signals, number_of_components):
    """Updates the 'weights' attribute for all ComponentSignal objects.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
    method: str
    data_input: 2d array

    Returns
    -------

    """
    for m in range(number_of_signals):
        stretched_components = np.zeros((signal_length, number_of_components))

        for c in components:
            stretched_components[:, c.id] = c.apply_stretch(m)[0]

        if method == 'align':
            weights = np.apply_along_axis(lambda x: scipy.optimize.nnls(stretched_components, x)[0], axis=0,
                                          arr=data_input)
        else:
            weights = np.apply_along_axis(lambda x: get_weights(
                stretched_components[0:signal_length, :].T @ stretched_components[0:signal_length, :],
                -1 * stretched_components[0:signal_length, :].T @ x, 0, 1), axis=0, arr=data_input)
    return 1


# def get_residual(components,data_input):
#     residual_matrix = -1 * data_input
#     components.apply_stretch().apply_weights()
#     for column in residual_matrix.T:


def reconstruct_signal(components, moment, signal_length):
    """Reconstruction a signal from its weighted and stretched components

    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the ComponentSignal objects
    moment: The index of the specific signal to be reconstructed
    signal_length: int

    Returns
    -------
    1d array like

    """
    reconstruction = np.zeros(signal_length)
    for c in components:
        stretched = c.apply_stretch(moment)[0]
        stretched_and_weighted = c.apply_weights(moment, stretched)
        reconstruction += stretched_and_weighted
    return reconstruction


def reconstruct_data(components, data_input, signal_length, number_of_signals):
    data_reconstruction = np.zeros((signal_length, number_of_signals))
    for m in range(number_of_signals):
        data_reconstruction[:, m] = reconstruct_signal(components, m, signal_length)  # To be completed
    return data_reconstruction


def stretching_operation_gra(components, residual, number_of_signals):
    stretching_operation_gra = []

    for c in components:
        for m in range(number_of_signals):
            stretched_gra = c.apply_stretch(m)[1]
            stretched_and_weighted_gra = c.apply_weights(m, stretched_gra)
            stretching_operation_gra.append(stretched_and_weighted_gra)

    return np.column_stack(stretching_operation_gra)


def stretching_operation_hess(components, residual, number_of_signals):
    stretching_operation_hess = []
    for c in components:
        for m in range(number_of_signals):
            stretched_hess = c.apply_stretch(m)[2]
            stretched_and_weighted_hess = c.apply_weights(m, stretched_hess)
            stretching_operation_hess.append(stretched_and_weighted_hess)

    return np.column_stack(stretching_operation_hess)


def update_stretching_factors(components, data_input, stretching_factor_matrix, smoothness, smoothness_term,
                              number_of_components, number_of_signals, signal_length, component_matrix, sparsity,
                              hessian_helper):
    data_input = np.asarray(data_input)
    stretching_factor_matrix = np.asarray(stretching_factor_matrix)
    component_matrix = np.asarray(component_matrix)

    def fun_value(stretching_factor_matrix):
        stretching_factor_matrix = stretching_factor_matrix.reshape(number_of_components, number_of_signals)
        residual = reconstruct_data(components, data_input, signal_length, number_of_signals) - data_input
        return objective_function(residual, stretching_factor_matrix, smoothness,
                                  smoothness_term, component_matrix, sparsity)

    def fun_gra(stretching_factor_matrix):
        stretching_factor_matrix = stretching_factor_matrix.reshape(number_of_components, number_of_signals)
        residual = reconstruct_data(components, data_input, signal_length, number_of_signals) - data_input
        tx = stretching_operation_gra(components, residual, number_of_signals)
        fun_gra = np.sum(tx * np.repeat(residual, number_of_components, axis=1), axis=0).reshape(number_of_components,
                                                                                                 number_of_signals)
        fun_gra += smoothness * stretching_factor_matrix @ smoothness_term.T @ smoothness_term
        return fun_gra.flatten()

    def fun_hess(stretching_factor_matrix):
        stretching_factor_matrix = stretching_factor_matrix.reshape(number_of_components, number_of_signals)
        residual = reconstruct_data(components, data_input, signal_length, number_of_signals) - data_input
        tx = stretching_operation_gra(components, residual, number_of_signals)
        hx = stretching_operation_hess(components, residual, number_of_signals)
        blocks = []
        for signal in range(number_of_signals):
            block = tx[:, signal*number_of_signals::number_of_signals]
            blocks.append(block.T @ block)

        fun_hess = scipy.linalg.block_diag(*blocks)

        residual_repeated = np.tile(residual,(1,number_of_components))
        product = hx * residual_repeated
        sum_result = np.sum(product,axis=0)
        reshaped_result = np.reshape(sum_result, (number_of_signals, number_of_components))
        reshaped_result = reshaped_result.T
        reshaped_result = reshaped_result.reshape(number_of_signals * number_of_components, 1)
        sparse_diag_matrix = spdiags(reshaped_result, [0] * reshaped_result.shape[0], (number_of_signals * number_of_components, number_of_signals * number_of_components))
        return fun_hess + sparse_diag_matrix + smoothness

    bounds = scipy.optimize.Bounds([.1] * number_of_components * number_of_signals,
                                   [np.inf] * number_of_components * number_of_signals)
    return scipy.optimize.minimize(fun_value, x0=stretching_factor_matrix.flatten(), jac=fun_gra, hess=fun_hess,
                                   method='trust-ncg',
                                   bounds=bounds)


def update_components(stretc):
    pass
