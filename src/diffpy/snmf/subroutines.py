import numdifftools
import numpy as np

from diffpy.snmf.containers import ComponentSignal
from diffpy.snmf.factorizers import lsqnonneg
from diffpy.snmf.optimizers import get_weights


def initialize_components(number_of_components, number_of_signals, grid_vector):
    """Initializes ComponentSignals for each of the components in the decomposition.

    Parameters
    ----------
    number_of_components: int
      The number of component signals in the NMF decomposition
    number_of_signals: int
    grid_vector: 1d array
      The grid of the user provided signals.

    Returns
    -------
    tuple of ComponentSignal objects
      The tuple containing `number_of_components` of initialized ComponentSignal objects.
    """
    if number_of_components <= 0:
        raise ValueError(f"Number of components = {number_of_components}. Number_of_components must be >= 1.")
    components = list()
    for component in range(number_of_components):
        component = ComponentSignal(grid_vector, number_of_signals, component)
        components.append(component)
    return tuple(components)


def lift_data(data_input, lift=1):
    """Lifts values of data_input.

    Adds 'lift' * the minimum value in data_input to data_input element-wise.

    Parameters
    ----------
    data_input: 2d array like
      The matrix containing a series of signals to be decomposed. Has dimensions N x M where N is the length
      of each signal and M is the number of signals.

    lift: float
      The factor representing how much to lift 'data_input'.

    Returns
    -------
    2d array like
      The matrix that contains data_input - (min(data_input) * lift).
    """
    data_input = np.asarray(data_input)
    return data_input + np.abs(np.min(data_input) * lift)


def construct_stretching_matrix(components, number_of_components, number_of_signals):
    """Constructs the stretching factor matrix.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
       The tuple containing the component signals in ComponentSignal objects.
    number_of_signals: int
      The number of signals in the data provided by the user.

    Returns
    -------
    2d array
      The matrix containing the stretching factors for the component signals for each of the signals in the
      raw data. Has dimensions `component_signal` x `number_of_signals`
    """
    if (len(components)) == 0:
        raise ValueError(f"Number of components = {number_of_components}. Number_of_components must be >= 1.")
    number_of_components = len(components)

    if number_of_signals <= 0:
        raise ValueError(f"Number of signals = {number_of_signals}. Number_of_signals must be >= 1.")

    stretching_factor_matrix = np.zeros((number_of_components, number_of_signals))
    for i, component in enumerate(components):
        stretching_factor_matrix[i] = component.stretching_factors
    return stretching_factor_matrix


def construct_component_matrix(components):
    """Constructs the component matrix.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signals in ComponentSignal objects.

    Returns
    -------
    2d array
      The matrix containing the component signal values. Has dimensions `signal_length` x `number_of_components`.
    """
    signal_length = len(components[0].iq)
    number_of_components = len(components)
    if signal_length == 0:
        raise ValueError(f"Signal length = {signal_length}. Signal length must be >= 1")
    if number_of_components == 0:
        raise ValueError(f"Number of components = {number_of_components}. Number_of_components must be >= 1")

    component_matrix = np.zeros((number_of_components, signal_length))
    for i, component in enumerate(components):
        component_matrix[i] = component.iq
    return component_matrix


def construct_weight_matrix(components):
    """Constructs the weights matrix.

    Constructs a Ķ x M matrix where K is the number of components and M is the
    number of signals. Each element is the stretching factor for a specific
    weights for a specific signal from the data input.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signals.

    Returns
    -------
    2d array like
      The 2d array containing the weightings for each component for each signal.
    """
    number_of_components = len(components)
    number_of_signals = len(components[0].weights)
    if number_of_components == 0:
        raise ValueError(f"Number of components = {number_of_components}. Number of components must be >= 1")
    if number_of_signals == 0:
        raise ValueError(f"Number of signals = {number_of_signals}. Number_of_signals must be >= 1.")
    weights_matrix = np.zeros((number_of_components, number_of_signals))
    for i, component in enumerate(components):
        weights_matrix[i] = component.weights
    return weights_matrix


def update_weights(components, data_input, method=None):
    """Updates the weights matrix.

    Updates the weights matrix and the weights vector for each ComponentSignal object.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signals.
    method: str
      The string specifying which method should be used to find a new weight matrix: non-negative least squares
      or a quadratic program.
    data_input: 2d array
      The 2d array containing the user-provided signals.

    Returns
    -------
    2d array
      The 2d array containing the weight factors for each component for each signal from `data_input`.
      Has dimensions K x M where K is the number of components and M is the number of signals in `data_input.`
    """
    data_input = np.asarray(data_input)
    weight_matrix = construct_weight_matrix(components)
    number_of_signals = len(components[0].weights)
    number_of_components = len(components)
    signal_length = len(components[0].grid)
    for signal in range(number_of_signals):
        stretched_components = np.zeros((signal_length, number_of_components))
        for i, component in enumerate(components):
            stretched_components[:, i] = component.apply_stretch(signal)[0]
        if method == "align":
            weights = lsqnonneg(stretched_components, data_input[:, signal])
        else:
            weights = get_weights(
                stretched_components.T @ stretched_components,
                -stretched_components.T @ data_input[:, signal],
                0,
                1,
            )
            weight_matrix[:, signal] = weights
    return weight_matrix


def reconstruct_signal(components, signal_idx):
    """Reconstructs a specific signal from its weighted and stretched components.

    Calculates the linear combination of stretched components where each term is the stretched component multiplied
    by its weight factor.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the ComponentSignal objects
    signal_idx: int
     The index of the specific signal in the input data to be reconstructed

    Returns
    -------
    1d array like
      The reconstruction of a signal from calculated weights, stretching factors, and iq values.
    """
    signal_length = len(components[0].grid)
    reconstruction = np.zeros(signal_length)
    for component in components:
        stretched = component.apply_stretch(signal_idx)[0]
        stretched_and_weighted = component.apply_weight(signal_idx, stretched)
        reconstruction += stretched_and_weighted
    return reconstruction


def initialize_arrays(number_of_components, number_of_moments, signal_length):
    """Generates the initial guesses for the weight, stretching, and component matrices.

    Calculates the initial guesses for the component matrix, stretching factor matrix, and weight matrix. The
    initial guess for the component matrix is a random (signal_length) x (number_of_components) matrix where
    each element is between 0 and 1. The initial stretching factor matrix is a random
    (number_of_components) x (number_of_moments) matrix where each element is number slightly perturbed from 1.
    The initial weight matrix guess is a random (number_of_components) x (number_of_moments) matrix where
    each element is between 0 and 1.

    Parameters
    ----------
    number_of_components: int
      The number of component signals to obtain from the stretched nmf decomposition.

    number_of_moments: int
      The number of signals in the user provided dataset where each signal is at a different moment.

    signal_length: int
      The length of each signal in the user provided dataset.

    Returns
    -------
    tuple of 2d arrays of floats
      The tuple containing three elements: the initial component matrix guess, the initial stretching factor matrix
      guess, and the initial weight factor matrix guess in that order.
    """
    component_matrix_guess = np.random.rand(signal_length, number_of_components)
    weight_matrix_guess = np.random.rand(number_of_components, number_of_moments)
    stretching_matrix_guess = (
        np.ones(number_of_components, number_of_moments)
        + np.random.randn(number_of_components, number_of_moments) * 1e-3
    )
    return component_matrix_guess, weight_matrix_guess, stretching_matrix_guess


def objective_function(
    residual_matrix, stretching_factor_matrix, smoothness, smoothness_term, component_matrix, sparsity
):
    """Defines the objective function of the algorithm and returns its value.

    Calculates the value of '(||residual_matrix||_F) ** 2 + smoothness * (||smoothness_term *
    stretching_factor_matrix.T||)**2 + sparsity * sum(component_matrix ** .5)' and returns its value.

    Parameters
    ----------
    residual_matrix: 2d array like
      The matrix where each column is the difference between an experimental PDF/XRD pattern and a calculated
      PDF/XRD pattern at each grid point. Has dimensions R x M where R is the length of each pattern and M is
      the amount of patterns.

    stretching_factor_matrix: 2d array like
      The matrix containing the stretching factors of the calculated component signal. Has dimensions K x M where
      K is the amount of components and M is the number of experimental PDF/XRD patterns.

    smoothness: float
      The coefficient of the smoothness term which determines the intensity of the smoothness term and its
      behavior. It is not very sensitive and is usually adjusted by multiplying it by ten.

    smoothness_term: 2d array like
      The regularization term that ensures that smooth changes in the component stretching signals are favored.
      Has dimensions (M-2) x M where M is the amount of experimentally obtained PDF/XRD patterns, the moment
      amount.

    component_matrix: 2d array like
      The matrix containing the calculated component signals of the experimental PDF/XRD patterns. Has dimensions
      R x K where R is the signal length and K is the number of component signals.

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
    return (
        0.5 * np.linalg.norm(residual_matrix, "fro") ** 2
        + 0.5 * smoothness * np.linalg.norm(smoothness_term @ stretching_factor_matrix.T, "fro") ** 2
        + sparsity * np.sum(np.sqrt(component_matrix))
    )


def get_stretched_component(stretching_factor, component, signal_length):
    """Applies a stretching factor to a component signal.

    Computes a stretched signal and reinterpolates it onto the original grid of points. Uses a normalized grid
    of evenly spaced integers counting from 0 to signal_length (exclusive) to approximate values in between grid
    nodes. Once this grid is stretched, values at grid nodes past the unstretched signal's domain are set to zero.
    Returns the approximate values of x(r/a) from x(r) where x is a component signal.

    Parameters
    ----------
    stretching_factor: float
      The stretching factor of a component signal at a particular moment.
    component: 1d array like
      The calculated component signal without stretching or weighting. Has length N, the length of the signal.
    signal_length: int
      The length of the component signal.

    Returns
    -------
    tuple of 1d array of floats
      The calculated component signal with stretching factors applied. Has length N, the length of the unstretched
      component signal. Also returns the gradient and hessian of the stretching transformation.
    """
    component = np.asarray(component)
    normalized_grid = np.arange(signal_length)

    def stretched_component_func(stretching_factor):
        return np.interp(normalized_grid / stretching_factor, normalized_grid, component, left=0, right=0)

    derivative_func = numdifftools.Derivative(stretched_component_func)
    second_derivative_func = numdifftools.Derivative(derivative_func)

    stretched_component = stretched_component_func(stretching_factor)
    stretched_component_gra = derivative_func(stretching_factor)
    stretched_component_hess = second_derivative_func(stretching_factor)

    return (
        np.asarray(stretched_component),
        np.asarray(stretched_component_gra),
        np.asarray(stretched_component_hess),
    )


def update_weights_matrix(
    component_amount,
    signal_length,
    stretching_factor_matrix,
    component_matrix,
    data_input,
    moment_amount,
    weights_matrix,
    method,
):
    """Update the weight factors matrix.

    Parameters
    ----------
    component_amount: int
      The number of component signals the user would like to determine from the experimental data.

    signal_length: int
      The length of the experimental signal patterns

    stretching_factor_matrix: 2d array like
      The matrx containing the stretching factors of the calculated component signals. Has dimensions K x M
      where K is the number of component signals and M is the number of XRD/PDF patterns.

    component_matrix: 2d array lik
      The matrix containing the unstretched calculated component signals. Has dimensions N x K where N is the
      length of the signals and K is the number of component signals.

    data_input: 2d array like
      The experimental series of PDF/XRD patterns. Has dimensions N x M where N is the length of the PDF/XRD
      signals and M is the number of PDF/XRD patterns.

    moment_amount: int
      The number of PDF/XRD patterns from the experimental data.

    weights_matrix: 2d array like
      The matrix containing the weights of the stretched component signals. Has dimensions K x M where K is
      the number of component signals and M is the number of XRD/PDF patterns.

    method: str
      The string specifying the method for obtaining individual weights.

    Returns
    -------
    2d array like
      The matrix containing the new weight factors of the stretched component signals.
    """
    stretching_factor_matrix = np.asarray(stretching_factor_matrix)
    component_matrix = np.asarray(component_matrix)
    data_input = np.asarray(data_input)
    weights_matrix = np.asarray(weights_matrix)
    weight = np.zeros(component_amount)
    for i in range(moment_amount):
        stretched_components = np.zeros((signal_length, component_amount))
        for n in range(component_amount):
            stretched_components[:, n] = get_stretched_component(
                stretching_factor_matrix[n, i], component_matrix[:, n], signal_length
            )[0]
        if method == "align":
            weight = lsqnonneg(stretched_components[0:signal_length, :], data_input[0:signal_length, i])
        else:
            weight = get_weights(
                stretched_components[0:signal_length, :].T @ stretched_components[0:signal_length, :],
                -1 * stretched_components[0:signal_length, :].T @ data_input[0:signal_length, i],
                0,
                1,
            )
        weights_matrix[:, i] = weight
    return weights_matrix


def get_residual_matrix(
    component_matrix, weights_matrix, stretching_matrix, data_input, moment_amount, component_amount, signal_length
):
    """Obtains the residual matrix between the experimental data and calculated data.

    Calculates the difference between the experimental data and the reconstructed experimental data created from
    the calculated components, weights, and stretching factors. For each experimental pattern, the stretched and
    weighted components making up that pattern are subtracted.

    Parameters
    ----------
    component_matrix: 2d array like
      The matrix containing the calculated component signals. Has dimensions N x K where N is the length of the
      signal and K is the number of calculated component signals.

    weights_matrix: 2d array like
      The matrix containing the calculated weights of the stretched component signals. Has dimensions K x M where
      K is the number of components and M is the number of moments or experimental PDF/XRD patterns.

    stretching_matrix: 2d array like
      The matrix containing the calculated stretching factors of the calculated component signals. Has dimensions
      K x M where K is the number of components and M is the number of moments or experimental PDF/XRD patterns.

    data_input: 2d array like
      The matrix containing the experimental PDF/XRD data. Has dimensions N x M where N is the length of the
      signals and M is the number of signal patterns.

    moment_amount: int
      The number of patterns in the experimental data. Represents the number of moments in time in the data series

    component_amount: int
      The number of component signals the user would like to obtain from the experimental data.

    signal_length: int
      The length of the signals in the experimental data.


    Returns
    -------
    2d array like
      The matrix containing the residual between the experimental data and reconstructed data from calculated
      values. Has dimensions N x M where N is the signal length and M is the number of moments. Each column
      contains the difference between an experimental signal and a reconstruction of that signal from the
      calculated weights, components, and stretching factors.
    """
    component_matrix = np.asarray(component_matrix)
    weights_matrix = np.asarray(weights_matrix)
    stretching_matrix = np.asarray(stretching_matrix)
    data_input = np.asarray(data_input)
    residual_matrx = -1 * data_input
    for m in range(moment_amount):
        residual = residual_matrx[:, m]
        for k in range(component_amount):
            residual = (
                residual
                + weights_matrix[k, m]
                * get_stretched_component(stretching_matrix[k, m], component_matrix[:, k], signal_length)[0]
            )
        residual_matrx[:, m] = residual
    return residual_matrx


def reconstruct_data(components):
    """Reconstructs the `input_data` matrix.

    Reconstructs the `input_data` matrix from calculated component signals, weights, and stretching factors.

    Parameters
    ----------
    components: tuple of ComponentSignal objects
      The tuple containing the component signals.

    Returns
    -------
    2d array
      The 2d array containing the reconstruction of input_data.
    """
    signal_length = len(components[0].iq)
    number_of_signals = len(components[0].weights)
    data_reconstruction = np.zeros((signal_length, number_of_signals))
    for signal in range(number_of_signals):
        data_reconstruction[:, signal] = reconstruct_signal(components, signal)
    return data_reconstruction
