import numpy as np
import pytest

from diffpy.snmf.polynomials import rooth

tr = [
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
]


@pytest.mark.parametrize("tr", tr)
def test_rooth(tr):
    actual = rooth(tr[0][0], tr[0][1])
    expected = tr[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
