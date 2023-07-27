import numpy as np
import numdifftools


class ComponentSignal:
    def __init__(self, grid, iq, weights, stretching_factors, id_number):
        self.iq = np.asarray(iq)
        self.grid = np.asarray(grid)
        self.weights = np.asarray(weights)
        self.stretching_factors = np.asarray(stretching_factors)
        self.id = int(id_number)
        self.stretched_iqs = ()

    def apply_stretch(self, m):
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
        if stretched_component is None:
            return self.iq * self.weights[m]
        else:
            return stretched_component * self.weights[m]
