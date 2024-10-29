import numpy as np
import pytest

from diffpy.snmf.polynomials import compute_root


@pytest.mark.parametrize(
    "inputs, expected",  # Corrected here to match the variables used in the function
    [
        ([0, 0], 0),
        ([-99.99, 12.50], 9.936397678254531),
        ([-4, 0], 2),
        ([10, 0], 0),
        ([-7, -7], 3.04891734),
        ([100, 72], 0),
        ([1, 3], 0),
        ([0, -7], 0),
        ([-9, 0], 3),
        ([-9, 3], 2.8169),
        ([[2, 2], 2], [0, 0]),
        ([[[2, 2], [2, 2]], 2], [[0, 0], [0, 0]]),
        ([[[[3, 2], [-2, -2], [100, 0]]], 2], [[[0, 0], [0, 0], [0, 0]]]),
    ],
)
def test_rooth(inputs, expected):
    linear_coefficient, constants = inputs
    actual = compute_root(linear_coefficient, constants)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
