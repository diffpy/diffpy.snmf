import numdifftools
import numpy as np


class ComponentSignal:
    """
    Attributes
    ----------
    grid: 1d array of floats
      The vector containing the grid points of the component.
    iq: 1d array of floats
      The intensity/g(r) values of the component.
    weights: 1d array of floats
      The vector containing the weight of the component signal for each signal.
    stretching_factors: 1d array of floats
      The vector containing the stretching factor for the component signal for each signal.
    id: int
      The number identifying the component.
    """

    def __init__(self, grid, number_of_signals, id_number, perturbation=1e-3):
        self.grid = np.asarray(grid)
        self.iq = np.random.rand(len(grid))
        self.weights = np.random.rand(number_of_signals)
        self.stretching_factors = np.ones(number_of_signals) + np.random.randn(number_of_signals) * perturbation
        self.id = int(id_number)

    def apply_stretch(self, m):
        """Applies a stretching factor to a component

        Parameters
        ----------
        m: int
          The index specifying which stretching factor to apply

        Returns
        -------
        tuple of 1d arrays
          The tuple of vectors where one vector is the stretched component, one vector is the 1st derivative of the
          stretching operation, and one vector is the second derivative of the stretching operation.
        """
        normalized_grid = np.arange(len(self.grid))
        interpolate_intensity = lambda stretching_factor: np.interp(  # noqa: E731
            normalized_grid / stretching_factor, normalized_grid, self.iq, left=0, right=0
        )
        derivative_func = numdifftools.Derivative(interpolate_intensity)
        second_derivative_func = numdifftools.Derivative(derivative_func)

        stretched_component = interpolate_intensity(self.stretching_factors[m])
        stretched_component_gra = derivative_func(self.stretching_factors[m])
        stretched_component_hess = second_derivative_func(self.stretching_factors[m])

        return (
            np.asarray(stretched_component),
            np.asarray(stretched_component_gra),
            np.asarray(stretched_component_hess),
        )

    def apply_weight(self, m, stretched_component=None):
        """Applies as weight factor to a component signal.

        Parameters
        ----------
        m: int
          The index specifying with weight to apply
        stretched_component: 1d array
          The 1d array containing a stretched component.

        Returns
        -------
        1d array
          The vector containing a component signal or stretched component signal with a weight factor applied.
        """
        if stretched_component is None:
            return self.iq * self.weights[m]
        else:
            return stretched_component * self.weights[m]
