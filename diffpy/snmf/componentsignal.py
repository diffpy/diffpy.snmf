import numpy as np
import numdifftools


class ComponentSignal:
    """
    Attributes
    ----------
    iq: 1d array of floats
      The intensity/g(r) values of the component
    grid: 1d array of floats
      The vector containing the grid points of the component
    weights: 1d array of floats
      The vector containing the weight of the component signal for each signal
    stretching_factors: 1d array of floats
      The vector containing the stretching factor for the component signal for each signal
    id: int
      The component number.
    """
    def __init__(self, grid, iq, weights, stretching_factors, id_number):
        self.iq = np.asarray(iq)
        self.grid = np.asarray(grid)
        self.weights = np.asarray(weights)
        self.stretching_factors = np.asarray(stretching_factors)
        self.id = int(id_number)
        self.stretched_iqs = ()

    def apply_stretch(self, m):
        """Applies a stretching factor to a component

        Parameters
        ----------
        m: int
          The index specifying which stretching factor to apply

        Returns
        -------
        tuple of 1d arrays
          The tuple of vectors where one vector is the stretched component, one vector is the gradient of the stretching
          operation, and one vector is the second derivative of the stretching operation.

        """
        normalized_grid = np.arange(len(self.grid))
        func = lambda sf: np.interp(normalized_grid / sf, normalized_grid, self.iq, left=0, right=0)
        derivative_func = numdifftools.Derivative(func)
        second_derivative_func = numdifftools.Derivative(derivative_func)

        stretched_component = func(self.stretching_factors[m])
        stretched_component_gra = derivative_func(self.stretching_factors[m])
        stretched_component_hess = second_derivative_func(self.stretching_factors[m])

        return np.asarray(stretched_component), np.asarray(stretched_component_gra), np.asarray(
            stretched_component_hess)

    def apply_weights(self, m, stretched_component=None):
        """Applies a weight factor to a component

        Parameters
        ----------
        m: int
          The index specifying which weight to apply
        stretched_component: 1d array
          The 1d array containing a component with stretching applied

        Returns
        -------
        1d array

        """
        if stretched_component is None:
            return self.iq * self.weights[m]
        else:
            return stretched_component * self.weights[m]
