import pytest

from diffpy.snmf.optimizers import get_weights

test_matrix = [
    # ([stretched_component_gram_matrix, linear_coefficient, lower_bound, upper_bound], expected)
    ([[[1, 0], [0, 1]], [1, 1], 0, 0], [0, 0]),
    ([[[1, 0], [0, 1]], [1, 1], -1, 1], [-1, -1]),
    ([[[1.75, 0], [0, 1.5]], [1, 1.2], -1, 1], [-0.571428571428571, -0.8]),
    ([[[0.75, 0.2], [0.2, 0.75]], [-0.1, -0.2], -1, 1], [0.066985645933014, 0.248803827751196]),
    ([[[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [1, 1, 1], -10, 12], [-1.5, -2, -1.5]),
    ([[[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [1, -1, -1], -10, 12], [0, 1, 1]),
    ([[[4, 0, 0, 0], [0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], [-2, -3, -4, -1], 0, 1000], [0.5, 1, 2, 1]),
]


@pytest.mark.parametrize("tm", test_matrix)
def test_get_weights(tm):
    stretched_component_gram_matrix = tm[0][0]
    linear_coefficient = tm[0][1]
    lower_bound = tm[0][2]
    upper_bound = tm[0][3]
    expected = tm[1]
    actual = get_weights(stretched_component_gram_matrix, linear_coefficient, lower_bound, upper_bound)
    assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6)
