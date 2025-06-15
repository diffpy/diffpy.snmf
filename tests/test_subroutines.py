import numpy as np
import pytest

from diffpy.snmf.containers import ComponentSignal
from diffpy.snmf.subroutines import (
    construct_component_matrix,
    construct_stretching_matrix,
    construct_weight_matrix,
    get_residual_matrix,
    get_stretched_component,
    initialize_components,
    lift_data,
    objective_function,
    reconstruct_data,
    reconstruct_signal,
    update_weights,
    update_weights_matrix,
)


@pytest.mark.parametrize(
    "residual_matrix, stretching_factor_matrix, smoothness, smoothness_term, component_matrix, sparsity, expected",
    [
        (
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            1e11,
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            1,
            2.574e14,
        ),
        (
            [[11, 2], [31, 4]],
            [[5, 63], [7, 18]],
            0.001,
            [[21, 2], [3, 4]],
            [[11, 22], [3, 40]],
            1,
            650.4576,
        ),
        (
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            1e11,
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            0,
            2.574e14,
        ),
    ],
)
def test_objective_function(
    residual_matrix,
    stretching_factor_matrix,
    smoothness,
    smoothness_term,
    component_matrix,
    sparsity,
    expected,
):
    actual = objective_function(
        residual_matrix,
        stretching_factor_matrix,
        smoothness,
        smoothness_term,
        component_matrix,
        sparsity,
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "stretching_factor, component, signal_length, expected",
    [
        (
            0.25,
            [6.55, 0.357, 8.49, 9.33, 6.78, 7.57, 7.43, 3.92, 6.55, 1.71],
            10,
            (
                [6.55, 6.78, 6.55, 0, 0, 0, 0, 0, 0, 0],
                [0, 14.07893122, 35.36478086, 0, 0, 0, 0, 0, 0, 0],
                [0, -19.92049156, 11.6931482, 0, 0, 0, 0, 0, 0, 0],
            ),
        ),
        (
            1.25,
            [-11.47, -10.688, -8.095, -29.44, 14.38],
            5,
            (
                [-11.47, -10.8444, -9.1322, -16.633, -20.6760],
                [0, -0.50048, -3.31904, 40.9824, -112.1792],
                [0, 0.800768, 5.310464, -65.57184, 179.48672],
            ),
        ),
        (
            0.88,
            [-11.47, -10.688, -8.095, -29.44, 14.38],
            5,
            (
                [-11.47, -10.3344, -13.9164, -11.5136, 0],
                [0, -3.3484, 55.1265, -169.7572, 0],
                [0, 7.609997, -125.2876, 385.81189, 0],
            ),
        ),
        (
            0.88,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            10,
            (
                [
                    1,
                    2.1364,
                    3.2727,
                    4.4091,
                    5.5455,
                    6.6818,
                    7.8182,
                    8.9545,
                    0,
                    0,
                ],
                [0, -1.29, -2.58, -3.87, -5.165, -6.45, -7.74, -9.039, 0, 0],
                [0, 2.93, 5.869, 8.084, 11.739, 14.674, 17.608, 20.5437, 0, 0],
            ),
        ),
        (
            0.55,
            [
                -2.9384,
                -1.4623,
                -2.0913,
                4.6304,
                -1.2127,
                1.4737,
                -0.3791,
                1.7506,
                -1.5068,
                -2.7625,
                0.9617,
                -0.3494,
                -0.3862,
                2.7960,
            ],
            14,
            (
                [
                    -2.9384,
                    -1.9769,
                    0.9121,
                    0.6314,
                    0.8622,
                    -2.4239,
                    -0.2302,
                    1.9281,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    2.07933,
                    38.632,
                    18.3748,
                    43.07305,
                    -61.557,
                    26.005,
                    -73.637,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    -7.56,
                    -140.480,
                    -66.81,
                    -156.6293,
                    223.84,
                    -94.564,
                    267.7734,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ),
        (
            0.987,
            [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
            11,
            (
                [
                    0,
                    0.2533,
                    0.5066,
                    0.7599,
                    1.0132,
                    1.2665,
                    1.5198,
                    1.7730,
                    2.0263,
                    2.2796,
                    0,
                ],
                [
                    0,
                    -0.2566,
                    -0.5132,
                    -0.7699,
                    -1.0265,
                    -1.2831,
                    -1.5398,
                    -1.7964,
                    -2.0530,
                    -2.3097,
                    0,
                ],
                [
                    0,
                    0.5200,
                    1.0400,
                    1.56005,
                    2.08007,
                    2.6000,
                    3.1201,
                    3.6401,
                    4.1601,
                    4.6801,
                    0,
                ],
            ),
        ),
        (
            -0.4,
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            9,
            (
                [-1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ),
    ],
)
def test_get_stretched_component(stretching_factor, component, signal_length, expected):
    actual = get_stretched_component(stretching_factor, component, signal_length)
    np.testing.assert_allclose(actual, expected, rtol=1e-01)


@pytest.mark.parametrize(
    "component_amount, signal_length, "
    "stretching_factor_matrix, component_matrix, data, moment, weights_matrix, method, expected",
    [
        (
            2,
            2,
            [[0.5, 0.6], [0.7, 0.8]],
            [[1, 2], [4, 8]],
            [[1.6, 2.8], [5, 8.8]],
            2,
            [[0.78, 0.12], [0.5, 0.5]],
            None,
            [[0.533333, 0.933333], [0.533333, 0.933333]],
        ),
        (
            2,
            3,
            [[0.5], [0.5]],
            [[1, 2.5], [1.5, 3], [2, 3.5]],
            [[1, 2], [3, 4], [5, 6]],
            1,
            [[0.5], [0.5]],
            None,
            [[1], [0.1892]],
        ),
        (
            2,
            3,
            [[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]],
            [[1, 2.5], [1.5, 3], [2, 3.5]],
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            3,
            [[0.5, 0.45, 0.4], [0.5, 0.45, 0.4]],
            None,
            [[1, 1, 1], [0.1892, 0.5600, 0.938]],
        ),
        (
            3,
            3,
            [[0.7, 0.8, 0.9], [0.71, 0.72, 0.73], [0.8, 0.85, 0.9]],
            [[-1, -2.7, -3], [-11, -6, -5.1], [0, -1, -0.5]],
            [[-2, -3, -4], [-9, -5, -5], [0, -2, -1]],
            3,
            [[0.9, 0.4, 0.5], [1, 0, 0.4], [0, 0, 0.98]],
            None,
            [
                [1.0, 0.0900485, 0.0],
                [0.585632, 0.497497, 0.179719],
                [0.0, 0.52223655, 1.0],
            ],
        ),
        (
            2,
            2,
            [[0.5], [0.5]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            1,
            [[0.6], [0.4]],
            "align",
            [[0], [0]],
        ),
        (
            1,
            3,
            [[0.5, 0.3]],
            [[1], [1.1], [1.3]],
            [[1, 2], [2, 3], [3, 2]],
            2,
            [[0.6, 0.4]],
            None,
            [[1, 1]],
        ),
        (
            2,
            2,
            [[0.5, 0.6], [0.7, 0.8]],
            [[1, 2], [4, 8]],
            [[1.6, 2.8], [5, 8.8]],
            2,
            [[0.78, 0.12], [0.5, 0.5]],
            "align",
            [[0, 0], [0.8, 1.4]],
        ),
        (
            2,
            3,
            [[0.5], [0.5]],
            [[1, 2.5], [1.5, 3], [2, 3.5]],
            [[1, 2], [3, 4], [5, 6]],
            1,
            [[0.5], [0.5]],
            "align",
            [[1.4], [0]],
        ),
        (
            3,
            3,
            [[0.7, 0.8, 0.9], [0.71, 0.72, 0.73], [0.8, 0.85, 0.9]],
            [[-1, -2.7, -3], [-11, -6, -5.1], [0, -1, -0.5]],
            [[-2, -3, -4], [-9, -5, -5], [0, -2, -1]],
            3,
            [[0.9, 0.4, 0.5], [1, 0, 0.4], [0, 0, 0.98]],
            "align",
            [
                [1.281265, 0.104355, 0],
                [0.0, 0.0, 0.0],
                [0.239578, 0.965215, 1.162571],
            ],
        ),
        (
            2,
            2,
            [[0.5], [0.5]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            1,
            [[0.6], [0.4]],
            "align",
            [[0], [0]],
        ),
        (
            1,
            3,
            [[0.5, 0.3]],
            [[1], [1.1], [1.3]],
            [[1, 2], [2, 3], [3, 2]],
            2,
            [[0.6, 0.4]],
            "align",
            [[1.3383, 2]],
        ),
    ],
)
def test_update_weights_matrix(
    component_amount,
    signal_length,
    stretching_factor_matrix,
    component_matrix,
    data,
    moment,
    weights_matrix,
    method,
    expected,
):
    actual = update_weights_matrix(
        component_amount,
        signal_length,
        stretching_factor_matrix,
        component_matrix,
        data,
        moment,
        weights_matrix,
        method,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize(
    "component_matrix, weights_matrix, stretching_matrix, data_input, moment_amount, "
    "component_amount, signal_length, expected",
    [
        (
            [[1, 2], [3, 4]],
            [[0.25], [0.75]],
            [[0.9], [0.7]],
            [[11, 22], [33, 44]],
            1,
            2,
            2,
            [[-9, -22], [-33, -44]],
        ),
        (
            [[1, 2], [3, 4]],
            [[1], [1]],
            [[1], [1]],
            [[11, 22], [33, 44]],
            1,
            2,
            2,
            [[-8, -22], [-26, -44]],
        ),
        (
            [[1.1, 4.4], [1.2, 4.5], [14, 7.8]],
            [[0.4, 0.6], [0.75, 0.25]],
            [[0.9, 0.89], [0.98, 0.88]],
            [[10, 20], [-10.5, -20.6], [0.6, 0.9]],
            2,
            2,
            3,
            [[-6.26, -18.24], [14.9744, 23.5067], [-0.6, -0.9]],
        ),
        # positive float
        (
            [[-1.1, -4.4], [-1.2, -4.5], [-14, -7.8]],
            [[0.4, 0.6], [0.75, 0.25]],
            [[0.9, 0.89], [0.98, 0.88]],
            [[10, 20], [-10.5, -20.6], [0.6, 0.9]],
            2,
            2,
            3,
            [[-13.74, -21.76], [6.0256, 17.6933], [-0.6, -0.9]],
        ),
        # negative floats
        (
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[0.4], [0.2], [0.3], [0.3]],
            [[0.9], [0.9], [0.9], [0.9]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            1,
            4,
            2,
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ),
    ],
)
def test_get_residual_matrix(
    component_matrix,
    weights_matrix,
    stretching_matrix,
    data_input,
    moment_amount,
    component_amount,
    signal_length,
    expected,
):
    actual = get_residual_matrix(
        component_matrix,
        weights_matrix,
        stretching_matrix,
        data_input,
        moment_amount,
        component_amount,
        signal_length,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-04)


@pytest.mark.parametrize(
    "components",
    [
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ]
        ),
        ([ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0)]),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 3),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 4),
            ]
        ),
        # ([]) # Exception expected
    ],
)
def test_reconstruct_data(components):
    actual = reconstruct_data(components)
    assert actual.shape == (len(components[0].iq), len(components[0].weights))


@pytest.mark.parametrize(
    "data_input, lift, expected",
    [
        # Correct structure: Each test case should be a tuple with three elements.pt
        (
            [[1, -1, 1], [0, 0, 0], [2, 10, -3]],
            1,
            [[4, 2, 4], [3, 3, 3], [5, 13, 0]],
        ),
        (
            [[1, -1, 1], [0, 0, 0], [2, 10, -3]],
            0,
            [[1, -1, 1], [0, 0, 0], [2, 10, -3]],
        ),
        (
            [[1, -1, 1], [0, 0, 0], [2, 10, -3]],
            0.5,
            [[2.5, 0.5, 2.5], [1.5, 1.5, 1.5], [3.5, 11.5, -1.5]],
        ),
        (
            [[1, -1, 1], [0, 0, 0], [2, 10, -3]],
            -1,
            [[4, 2, 4], [3, 3, 3], [5, 13, 0]],
        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            100,
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
        ([[1.5, 2], [10.5, 1], [0.5, 2]], 1, [[2, 2.5], [11, 1.5], [1, 2.5]]),
        (
            [[-10, -10.5], [-12.2, -12.2], [0, 0]],
            1,
            [[2.2, 1.7], [0, 0], [12.2, 12.2]],
        ),
    ],
)
def test_lift_data(data_input, lift, expected):
    actual = lift_data(data_input, lift)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "number_of_components, number_of_signals, grid_vector",
    [
        (2, 3, [0, 0.5, 1, 1.5]),  # Regular usage
        # (0, 3,[0, .5, 1, 1.5]), # Zero components raise an exception. Not tested
    ],
)
def test_initialize_componentstest_initialize_components(number_of_components, number_of_signals, grid_vector):
    actual = initialize_components(number_of_components, number_of_signals, grid_vector)
    assert len(actual) == number_of_components
    assert len(actual[0].weights) == number_of_signals
    assert (actual[0].grid == np.array(grid_vector)).all()


@pytest.mark.parametrize(
    "components, number_of_components, number_of_signals",
    [
        ([ComponentSignal([0, 0.5, 1, 1.5], 20, 0)], 1, 20),
        ([ComponentSignal([0, 0.5, 1, 1.5], 20, 0)], 4, 20),
        # ([ComponentSignal([0,.5,1,1.5],20,0)],0,20), # Raises an exception
        # ([ComponentSignal([0,.5,1,1.5],20,0)],-2,20), # Raises an exception
        # ([ComponentSignal([0,.5,1,1.5],20,0)],1,0), # Raises an Exception
        # ([ComponentSignal([0,.5,1,1.5],20,0)],1,-3), # Raises an exception
        (
            [
                ComponentSignal([0, 0.5, 1, 1.5], 20, 0),
                ComponentSignal([0, 0.5, 1, 1.5], 20, 1),
            ],
            2,
            20,
        ),
        (
            [
                ComponentSignal([0, 0.5, 1, 1.5], 20, 0),
                ComponentSignal([0, 0.5, 1, 21.5], 20, 1),
            ],
            2,
            20,
        ),
        (
            [
                ComponentSignal([0, 1, 1.5], 20, 0),
                ComponentSignal([0, 0.5, 1, 21.5], 20, 1),
            ],
            2,
            20,
        ),
        # ([ComponentSignal([0,.5,1,1.5],20,0),ComponentSignal([0,.5,1,1.5],20,1)],1,-3),
        # Negative signal length. Raises an exception
        # ([],1,20), # Empty components. Raises an Exception
        # ([],-1,20), # Empty components with negative number of components. Raises an exception
        # ([],0,20), # Empty components with zero number of components. Raises an exception
        # ([],1,0), # Empty components with zero signal length. Raises an exception.
        # ([],-1,-2), # Empty components with negative number of components and signal length Raises an exception.
    ],
)
def test_construct_stretching_matrix(components, number_of_components, number_of_signals):
    actual = construct_stretching_matrix(components, number_of_components, number_of_signals)
    for component in components:
        np.testing.assert_allclose(actual[component.id, :], component.stretching_factors)
        # assert actual[component.id, :] == component.stretching_factors


@pytest.mark.parametrize(
    "components",
    [
        ([ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0)]),
        ([ComponentSignal([0, 0.25, 0.5, 0.75, 1], 0, 0)]),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 2.75, 1], 20, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0.25], 20, 0),
                ComponentSignal([0.25], 20, 1),
                ComponentSignal([0.25], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 1),
            ]
        ),
        # ([ComponentSignal([[0, .25, .5, .75, 1],[0, .25, .5, .75, 1]], 20, 0),
        # ComponentSignal([[0, .25, .5, .75, 1],[0, .25, .5, .75, 1]], 20, 1)]),
        # iq is multidimensional. Expected to fail
        # (ComponentSignal([], 20, 0)), # Expected to fail
        # ([]), #Expected to fail
    ],
)
def test_construct_component_matrix(components):
    actual = construct_component_matrix(components)
    for component in components:
        np.testing.assert_allclose(actual[component.id], component.iq)


@pytest.mark.parametrize(
    "components",
    [
        ([ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0)]),
        # ([ComponentSignal([0,.25,.5,.75,1],0,0)]), # 0 signal length. Failure expected
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 2.75, 1], 20, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0.25], 20, 0),
                ComponentSignal([0.25], 20, 1),
                ComponentSignal([0.25], 20, 2),
            ]
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 20, 1),
            ]
        ),
        # (ComponentSignal([], 20, 0)), # Expected to fail
        # ([]), #Expected to fail
    ],
)
def test_construct_weight_matrix(components):
    actual = construct_weight_matrix(components)
    for component in components:
        np.testing.assert_allclose(actual[component.id], component.weights)


@pytest.mark.parametrize(
    "components, data_input, method",
    [
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            [[1, 1], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5], [2, 2.1]],
            None,
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            [[1, 1], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5], [2, 2.1]],
            "align",
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            None,
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            "align",
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            [[-0.5, 1], [1.2, -1.3], [1.1, -1], [0, -1.5], [0, 0.1]],
            None,
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            [[-0.5, 1], [1.2, -1.3], [1.1, -1], [0, -1.5], [0, 0.1]],
            "align",
        ),
        # ([ComponentSignal([0, .25, .5, .75, 1], 0, 0), ComponentSignal([0, .25, .5, .75, 1], 0, 1),
        # ComponentSignal([0, .25, .5, .75, 1], 0, 2)], [[1, 1], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5], [2, 2.1]],
        # None),
        # ([ComponentSignal([0, .25, .5, .75, 1], 0, 0), ComponentSignal([0, .25, .5, .75, 1], 0, 1),
        # ComponentSignal([0, .25, .5, .75, 1], 0, 2)], [], None),
        # ([ComponentSignal([0, .25, .5, .75, 1], 2, 0), ComponentSignal([0, .25, .5, .75, 1], 2, 1),
        # ComponentSignal([0, .25, .5, .75, 1], 2, 2)], [], 170),
    ],
)
def test_update_weights(components, data_input, method):
    actual = update_weights(components, data_input, method)
    assert np.shape(actual) == (len(components), len(components[0].weights))


@pytest.mark.parametrize(
    "components, expected",
    [
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            1,
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 2, 2),
            ],
            0,
        ),
        (
            [
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 3, 0),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 3, 1),
                ComponentSignal([0, 0.25, 0.5, 0.75, 1], 3, 2),
            ],
            2,
        ),
        # ([ComponentSignal([0, .25, .5, .75, 1], 2, 0), ComponentSignal([0, .25, .5, .75, 1], 2, 1),
        # ComponentSignal([0, .25, .5, .75, 1], 2, 2)], -1),
    ],
)
def test_reconstruct_signal(components, expected):
    actual = reconstruct_signal(components, expected)
    assert len(actual) == len(components[0].grid)
