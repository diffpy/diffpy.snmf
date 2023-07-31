import pytest
import numpy as np
from diffpy.snmf.datapoint import ComponentSignal
from diffpy.snmf.subroutines import objective_function
from diffpy.snmf.new_subroutines import construct_stretching_matrix, construct_component_matrix, \
    construct_weight_matrix, reconstruct_signal, update_stretching_factors

to = [
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], 1e11, [[1, 2], [3, 4]], [[1, 2], [3, 4]], 1], 2.574e14),
    ([[[11, 2], [31, 4]], [[5, 63], [7, 18]], .001, [[21, 2], [3, 4]], [[11, 22], [3, 40]], 1], 650.4576),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], 1e11, [[1, 2], [3, 4]], [[1, 2], [3, 4]], 0], 2.574e14),
]
@pytest.mark.parametrize("to", to)
def test_objective_function(to):
    actual = objective_function(to[0][0], to[0][1], to[0][2], to[0][3], to[0][4], to[0][5])
    expected = to[1]
    assert actual == pytest.approx(expected)


tcso = [([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0)], 1, 3], [[.9, .8, .7]]),
        ([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0),
           ComponentSignal([0, 1, 2], [10, 2, 32], [.5, 4, .3], [.3, .8, .75], 1)], 2, 3],
         [[.9, .8, .7], [.3, .8, .75]]),
        ]
@pytest.mark.parametrize('tcso', tcso)
def test_construct_stretching_matrix(tcso):
    actual = construct_stretching_matrix(tcso[0][0], tcso[0][1], tcso[0][2])
    expected = tcso[1]
    np.testing.assert_allclose(actual, expected)


tcco = [([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0)], 1, 3], [[1], [2], [3]]),
        ([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0),
           ComponentSignal([0, 1, 2], [10, 2, 32], [.5, 4, .3], [.3, .8, .75], 1)], 2, 3],
         [[1, 10], [2, 2], [3, 32]])
        ]
@pytest.mark.parametrize('tcco', tcco)
def test_construct_component_matrix(tcco):
    actual = construct_component_matrix(tcco[0][0], tcco[0][1], tcco[0][2])
    expected = tcco[1]
    np.testing.assert_allclose(actual, expected)


tcwo = [([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0)], 1, 3], [[.5, 4, .3]]),
        ([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0),
           ComponentSignal([0, 1, 2], [10, 2, 32], [.5, 4, .3], [.3, .8, .75], 1)], 2, 3],
         [[.5, 4, .3], [.5, 4, .3]])
        ]
@pytest.mark.parametrize('tcwo', tcwo)
def test_construct_weight_matrix(tcwo):
    actual = construct_weight_matrix(tcwo[0][0], tcwo[0][1], tcwo[0][2])
    expected = tcwo[1]
    np.testing.assert_allclose(actual, expected)


trs = [([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0),
          ComponentSignal([0, 1, 2], [10, 2, 32], [.5, 4, .3], [.3, .8, .75], 1)], 1, 3],
        [44, 47, 0])
       ]
@pytest.mark.parametrize('trs', trs)
def test_reconstruct_signal(trs):
    actual = reconstruct_signal(trs[0][0], trs[0][1], trs[0][2])
    expected = trs[1]
    np.testing.assert_allclose(actual, expected)


tusf = [([[ComponentSignal([0, 1, 2], [1, 2, 3], [.5, 4, .3], [.9, .8, .7], 0),
           ComponentSignal([0, 1, 2], [10, 2, 32], [.5, 4, .3], [.9, .8, .75], 1)], [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          [[.9, .9, .9], [.9, .9, .9]], 1e11, np.zeros((2,3)), 2, 3, 3, [[1, 10], [2, 2], [3, 32]], 0,np.zeros((6,6))],
         [44, 47, 0])
        ]
@pytest.mark.parametrize('tusf', tusf)
def test_update_stretching_factors(tusf):
    actual = update_stretching_factors(tusf[0][0], tusf[0][1], tusf[0][2], tusf[0][3], tusf[0][4], tusf[0][5],
                                       tusf[0][6], tusf[0][7], tusf[0][8], tusf[0][9],tusf[0][10])
    print(actual)
    expected = tusf[1]
    np.testing.assert_allclose(actual, expected)
