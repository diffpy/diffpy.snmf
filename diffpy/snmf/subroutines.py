import numpy as np
import scipy


# import scipy.interpolate


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

    Approximates the values of a component signal at points in between its grid nodes using quadratic spline
    interpolation. Uses a normalized grid of evenly spaced integers counting from 0 to signal_length (exclusive) to
    approximate values in between grid nodes. Once this grid is stretched, values at grid nodes past the unstretched
    signal's domain are set to zero. Returns the approximate values of x(r/a) from x(r) where x is a component signal.

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
    normalized_grid = np.arange(0, signal_length)
    spline = scipy.interpolate.UnivariateSpline(normalized_grid, component, k=2, ext=1)
    stretched_grid = normalized_grid / stretching_factor
    stretched_component = spline.__call__(stretched_grid, ext=1)
    return stretched_component
