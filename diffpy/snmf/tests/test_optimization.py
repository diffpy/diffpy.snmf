import pytest
from diffpy.snmf.optimizers import get_weights

tm = [
    ([[[1, 0], [0, 1]], [1, 1], 0, 1], [0, 0]),
    ([[[1, 0], [0, 1]], [1, 1], -1, 1], [-1, -1]),
    ([[[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [1, 1, 1], -10, 12], [-1.5, -2, -1.5]),
    ([[[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [1, -1, -1], -10, 12], [0, 1, 1]),
    ([[[4, 0, 0, 0], [0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], [-2, -3, -4, -1], 0, 1000], [.5, 1, 2, 1]),

]


@pytest.mark.parametrize('tm', tm)
def test_get_weights(tm):
    expected = tm[1]
    actual = get_weights(tm[0][0], tm[0][1], tm[0][2], tm[0][3])
    assert (actual == pytest.approx(expected))
