import numpy as np
import scipy
import pytest
from diffpy.snmf.factorizers import lsqnonneg

tl = [
    ([np.array([[1, 0], [1, 0], [0, 1]]), np.array([2, 1, 1])], np.array([1.5, 1.])),
    ([np.array([[2, 3], [1, 2], [0, 0]]), np.array([7, 7, 2])], np.array([0, 2.6923])),
    ([np.array([[3, 2], [5, 3]]), np.array([2, 3])])
]


@pytest.mark.parametrize('tl', tl)
def test_lsqnonneg(tl):
    actual = lsqnonneg(tl[0][0], tl[0][1])
    expected = tl[1]
    np.testing.assert_array_almost_equal(actual, expected, decimal=4)
