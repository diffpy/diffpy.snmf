import numpy as np
from diffpy.snmf.optimizers import get_weights
from diffpy.snmf.factorizers import lsqnonneg
from diffpy.snmf.containers import ComponentSignal
import numdifftools


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


def create_components(number_of_components, grid_vector, number_of_signals, signal_length):
    """Creates the ComponentSignal objects

    Parameters
    ----------
    number_of_components: int
      The number specifying the number of components signals.
    grid_vector: 1d array
      The 1d array containing the grid of the signals.
    number_of_signals: int
      The number of signals in the data input.
    signal_length: int
      The number specifying the length of the signals' grid.

    Returns
    -------
    tuple of ComponentSignal objects

    """
    component_list = []
    for c in range(number_of_components):
        iq_guess = np.random.rand(signal_length)
        weights_guess = np.random.rand(number_of_signals)
        stretching_factors_guess = np.ones(number_of_signals) + np.random.randn(number_of_signals) * 1e-3
        comp = ComponentSignal(grid_vector, iq_guess, weights_guess, stretching_factors_guess, c)
        component_list.append(comp)
    return tuple(component_list)


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
      The number specifying the length of the signals' grid.
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

# def update_weights_matrix(component_amount, signal_length, stretching_factor_matrix, component_matrix, data_input,
#                           moment_amount, weights_matrix, method):
#     """Update the weight factors matrix.
#
#     Parameters
#     ----------
#     component_amount: int
#       The number of component signals the user would like to determine from the experimental data.
#
#     signal_length: int
#       The length of the experimental signal patterns
#
#     stretching_factor_matrix: 2d array like
#       The matrx containing the stretching factors of the calculated component signals. Has dimensions K x M where K is
#       the number of component signals and M is the number of XRD/PDF patterns.
#
#     component_matrix: 2d array like
#       The matrix containing the unstretched calculated component signals. Has dimensions N x K where N is the length of
#       the signals and K is the number of component signals.
#
#     data_input: 2d array like
#       The experimental series of PDF/XRD patterns. Has dimensions N x M where N is the length of the PDF/XRD signals and
#       M is the number of PDF/XRD patterns.
#
#     moment_amount: int
#       The number of PDF/XRD patterns from the experimental data.
#
#     weights_matrix: 2d array like
#       The matrix containing the weights of the stretched component signals. Has dimensions K x M where K is the number
#       of component signals and M is the number of XRD/PDF patterns.
#
#     method: str
#       The string specifying the method for obtaining individual weights.
#
#     Returns
#     -------
#     2d array like
#       The matrix containing the new weight factors of the stretched component signals.
#
#     """
#     stretching_factor_matrix = np.asarray(stretching_factor_matrix)
#     component_matrix = np.asarray(component_matrix)
#     data_input = np.asarray(data_input)
#     weights_matrix = np.asarray(weights_matrix)
#     weight = np.zeros(component_amount)
#     for i in range(moment_amount):
#         stretched_components = np.zeros((signal_length, component_amount))
#         for n in range(component_amount):
#             stretched_components[:, n] = get_stretched_component(stretching_factor_matrix[n, i], component_matrix[:, n],
#                                                                  signal_length)[0]
#         if method == 'align':
#             weight = lsqnonneg(stretched_components[0:signal_length, :], data_input[0:signal_length, i])
#         else:
#             weight = get_weights(
#                 stretched_components[0:signal_length, :].T @ stretched_components[0:signal_length, :],
#                 -1 * stretched_components[0:signal_length, :].T @ data_input[0:signal_length, i],
#                 0, 1)
#         weights_matrix[:, i] = weight
#     return weights_matrix


# def get_residual_matrix(component_matrix, weights_matrix, stretching_matrix, data_input, moment_amount,
#                         component_amount, signal_length):
#     """Obtains the residual matrix between the experimental data and calculated data
#
#     Calculates the difference between the experimental data and the reconstructed experimental data created from the
#     calculated components, weights, and stretching factors. For each experimental pattern, the stretched and weighted
#     components making up that pattern are subtracted.
#
#     Parameters
#     ----------
#     component_matrix: 2d array like
#       The matrix containing the calculated component signals. Has dimensions N x K where N is the length of the signal
#       and K is the number of calculated component signals.
#
#     weights_matrix: 2d array like
#       The matrix containing the calculated weights of the stretched component signals. Has dimensions K x M where K is
#       the number of components and M is the number of moments or experimental PDF/XRD patterns.
#
#     stretching_matrix: 2d array like
#       The matrix containing the calculated stretching factors of the calculated component signals. Has dimensions K x M
#       where K is the number of components and M is the number of moments or experimental PDF/XRD patterns.
#
#     data_input: 2d array like
#       The matrix containing the experimental PDF/XRD data. Has dimensions N x M where N is the length of the signals and
#       M is the number of signal patterns.
#
#     moment_amount: int
#       The number of patterns in the experimental data. Represents the number of moments in time in the data series
#
#     component_amount: int
#       The number of component signals the user would like to obtain from the experimental data.
#
#     signal_length: int
#       The length of the signals in the experimental data.
#
#
#     Returns
#     -------
#     2d array like
#       The matrix containing the residual between the experimental data and reconstructed data from calculated values.
#       Has dimensions N x M where N is the signal length and M is the number of moments. Each column contains the
#       difference between an experimental signal and a reconstruction of that signal from the calculated weights,
#       components, and stretching factors.
#
#     """
#     component_matrix = np.asarray(component_matrix)
#     weights_matrix = np.asarray(weights_matrix)
#     stretching_matrix = np.asarray(stretching_matrix)
#     data_input = np.asarray(data_input)
#     residual_matrx = -1 * data_input
#     for m in range(moment_amount):
#         residual = residual_matrx[:, m]
#         for k in range(component_amount):
#             residual = residual + weights_matrix[k, m] * get_stretched_component(stretching_matrix[k, m],
#                                                                                  component_matrix[:, k], signal_length)[
#                 0]
#         residual_matrx[:, m] = residual
#     return residual_matrx


# def reconstruct_data(stretching_factor_matrix, component_matrix, weight_matrix, component_amount,
#                      moment_amount, signal_length):
#     """Reconstructs the experimental data from the component signals, stretching factors, and weights.
#
#     Calculates the stretched and weighted components at each moment.
#
#     Parameters
#     ----------
#     stretching_factor_matrix: 2d array like
#       The matrix containing the stretching factors of the component signals. Has dimensions K x M where K is the number
#       of components and M is the number of moments.
#
#     component_matrix: 2d array like
#       The matrix containing the unstretched component signals. Has dimensions N x K where N is the length of the signals
#       and K is the number of components.
#
#     weight_matrix: 2d array like
#       The matrix containing the weights of the stretched component signals at each moment in time. Has dimensions
#       K x M where K is the number of components and M is the number of moments.
#
#     component_amount: int
#       The number of component signals the user would like to obtain from the experimental data.
#
#     moment_amount: int
#       The number of patterns in the experimental data. Represents the number of moments in time in the data series.
#
#     signal_length: int
#       The length of the signals in the experimental data.
#
#     Returns
#     -------
#     tuple of 2d array of floats
#       The stretched and weighted component signals at each moment. Has dimensions N x (M * K) where N is the length of
#       the signals, M is the number of moments, and K is the number of components. The resulting matrix has M blocks
#       stacked horizontally where each block is the weighted and stretched components at each moment. Also contains
#       the gradient and hessian matrices of the reconstructed data.
#
#     """
#     stretching_factor_matrix = np.asarray(stretching_factor_matrix)
#     component_matrix = np.asarray(component_matrix)
#     weight_matrix = np.asarray(weight_matrix)
#     stretched_component_series = []
#     stretched_component_series_gra = []
#     stretched_component_series_hess = []
#     for moment in range(moment_amount):
#         for component in range(component_amount):
#             stretched_component = get_stretched_component(stretching_factor_matrix[component, moment],
#                                                           component_matrix[:, component], signal_length)
#             stretched_component_series.append(stretched_component[0])
#             stretched_component_series_gra.append(stretched_component[1])
#             stretched_component_series_hess.append(stretched_component[2])
#     stretched_component_series = np.column_stack(stretched_component_series)
#     stretched_component_series_gra = np.column_stack(stretched_component_series_gra)
#     stretched_component_series_hess = np.column_stack(stretched_component_series_hess)
#
#     reconstructed_data = []
#     reconstructed_data_gra = []
#     reconstructed_data_hess = []
#     moment = 0
#     for s_component in range(0, moment_amount * component_amount, component_amount):
#         block = stretched_component_series[:, s_component:s_component + component_amount]
#         block_gra = stretched_component_series_gra[:, s_component:s_component + component_amount]
#         block_hess = stretched_component_series_hess[:, s_component:s_component + component_amount]
#         for component in range(component_amount):
#             block[:, component] = block[:, component] * weight_matrix[component, moment]
#             block_gra[:, component] = block_gra[:, component] * weight_matrix[component, moment]
#             block_hess[:, component] = block_hess[:, component] * weight_matrix[component, moment]
#         reconstructed_data.append(block)
#         reconstructed_data_gra.append(block_gra)
#         reconstructed_data_hess.append(block_hess)
#         moment += 1
#     return np.column_stack(reconstructed_data), np.column_stack(reconstructed_data_gra), np.column_stack(
#         reconstructed_data_hess)
