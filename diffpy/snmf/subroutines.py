import numpy as np
from diffpy.snmf.optimizers import get_weights
from diffpy.snmf.factorizers import lsqnonneg


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


def get_stretched_component(stretching_factor, component, signal_length):
    """Applies a stretching factor to a component signal.

    Computes a stretched signal and reinterpolates it onto the original grid of points. Uses a normalized grid of evenly
    spaced integers counting from 0 to signal_length (exclusive) to approximate values in between grid nodes. Once this
    grid is stretched, values at grid nodes past the unstretched signal's domain are set to zero. Returns the
    approximate values of x(r/a) from x(r) where x is a component signal.

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
    1d array of floats
      The calculated component signal with stretching factors applied. Has length N, the length of the unstretched
      component signal.

    """
    component = np.asarray(component)
    normalized_grid = np.arange(signal_length)
    return np.interp(normalized_grid / stretching_factor, normalized_grid, component, left=0, right=0)


def update_weights_matrix(component_amount, signal_length, stretching_factor_matrix, component_matrix, data_input,
                          moment_amount, weights_matrix, method):
    """Update the weight factors matrix.

    Parameters
    ----------
    component_amount: int
      The number of component signals the user would like to determine from the experimental data.

    signal_length: int
      The length of the experimental signal patterns

    stretching_factor_matrix: 2d array like
      The matrx containing the stretching factors of the calculated component signals. Has dimensions K x M where K is
      the number of component signals and M is the number of XRD/PDF patterns.

    component_matrix: 2d array like
      The matrix containing the unstretched calculated component signals. Has dimensions N x K where N is the length of
      the signals and K is the number of component signals.

    data_input: 2d array like
      The experimental series of PDF/XRD patterns. Has dimensions N x M where N is the length of the PDF/XRD signals and
      M is the number of PDF/XRD patterns.

    moment_amount: int
      The number of PDF/XRD patterns from the experimental data.

    weights_matrix: 2d array like
      The matrix containing the weights of the stretched component signals. Has dimensions K x M where K is the number
      of component signals and M is the number of XRD/PDF patterns.

    method: str
      The string specifying the method for obtaining individual weights.

    Returns
    -------
    2d array like
      The matrix containing the new weight factors of the stretched component signals.

    """
    weight = np.zeros(component_amount)
    for i in range(moment_amount):
        stretched_components = np.zeros(signal_length, component_amount)
        for n in range(component_amount):
            stretched_components[:, n] = get_stretched_component(stretching_factor_matrix[n, i], component_matrix[:, n])
            if method == 'align':
                weight = lsqnonneg(stretched_components[0:signal_length, :], data_input[0:signal_length, i])
            else:
                weight = get_weights(
                    stretched_components[0:signal_length, :].T @ stretched_components[0:signal_length, :],
                    -1 * stretched_components[0:signal_length, :].T @ data_input[0:signal_length, i],
                    np.zeros(component_amount), np.ones(component_amount))
        weights_matrix[:, i] = weight
    return weights_matrix


def get_residual_matrix(component_matrix, weights_matrix, stretching_matrix, data_input, moment_amount,
                        component_amount):
    """Obtains the residual matrix between the experimental data and calculated data

    Calculates the difference between the experimental data and the reconstructed experimental data created from the
    calculated components, weights, and stretching factors.

    Parameters
    ----------
    component_matrix: 2d array like
      The matrix containing the calculated component signals. Has dimensions N x K where N is the length of the signal
      and K is the number of calculated component signals.

    weights_matrix: 2d array like
      The matrix containing the calculated weights of the stretched component signals. Has dimensions K x M where K is
      the number of components and M is the number of moments or experimental PDF/XRD patterns.

    stretching_matrix: 2d array like
      The matrix containing the calculated stretching factors of the calculated component signals. Has dimensions K x M
      where K is the number of components and M is the number of moments or experimental PDF/XRD patterns.

    data_input: 2d array like
      The matrix containing the experimental PDF/XRD data. Has dimensions N x M where N is the length of the signals and
      M is the number of signal patterns.

    moment_amount: int
      The number of PDF/XRD patterns.

    component_amount: int
      The number of component signals that user would like to experimental data.


    Returns
    -------
    2d array like

    """
    residual_matrx = -1 * data_input
    for m in range(moment_amount):
        residual = residual_matrx[:, m]
        for k in range(component_amount):
            residual = residual + weights_matrix[k, m] * get_stretched_component(stretching_matrix[k, m],
                                                                                 component_matrix[:, k])
        residual_matrx[:, m] = residual
    return residual_matrx


def reconstruct_data():
    pass
