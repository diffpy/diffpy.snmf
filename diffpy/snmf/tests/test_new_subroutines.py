import pytest
import numpy as np
from diffpy.snmf.componentsignal import ComponentSignal
from diffpy.snmf.subroutines import objective_function
from diffpy.snmf.new_subroutines import construct_stretching_matrix, construct_component_matrix, \
    construct_weight_matrix, reconstruct_signal, update_stretching_factors, create_components

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


tusf = [([[ComponentSignal([0, 1], [.5, .3], [.5, .4], [.9, .9], 0),
           ComponentSignal([0, 1], [1, 1.5], [.5, .6], [.8, .98], 1)], [[1, 2], [3, 4]],
          [[.9, .9], [.8, .98]], 1, np.ones((2, 2)), 2, 2, 2, [[.5, 1], [.3, 1.5]], 0,
          np.zeros((4, 4))],
         [44, 47, 0]),
        ([[ComponentSignal([0, 1, 2, 3, 4], [.6, .7, .8, .9, .95], [.6, .4, .5], [.8, .9, .77], 0),
           ComponentSignal([0, 1, 2, 3, 4], [.8, .7, .6, .5, .4], [.5, .6, .25], [.8, .88, .9], 1),
           ComponentSignal([0, 1, 2, 3, 4], [1, 1.2, 1, .8, .75], [.5, .25, .75], [.9, 95, .5], 2)],
          [[10, 11, 12], [9, 8, 7], [8, 7, 7], [7, 6, 6.5], [6.5, 5, 4]],
          [[.8, .9, .77], [.8, .88, .9], [.9, .95, .5]], 1e11, np.ones((1, 3)), 3, 3, 5,
          [[.6, .8, 1], [.7, .7, 1.2], [.8, .6, 1], [.9, .5, .8], [.95, .4, .75]], 0,
          np.zeros((15, 15))],
         [44, 47, 0])

        ]


@pytest.mark.parametrize('tusf', tusf)
def test_update_stretching_factors(tusf):
    actual = update_stretching_factors(tusf[0][0], tusf[0][1], tusf[0][2], tusf[0][3], tusf[0][4], tusf[0][5],
                                       tusf[0][6], tusf[0][7], tusf[0][8], tusf[0][9], tusf[0][10])
    print(actual)
    expected = tusf[1]
    np.testing.assert_allclose(actual, expected)


tcc = [(2, [0, .5, 1, 1.5], 3, 3),
       (3, [0, 10, 20, 30], 10, 15),
       (0, [0], 11, 30),
       (5, [1, 1, 1, 1, 1, 1], 10000, 40000),
       (3, np.arange(stop=125, step=.05), 20, 2500),
       ]


@pytest.mark.parametrize('tcc', tcc)
def test_create_components(tcc):
    actual = create_components(tcc[0], tcc[1], tcc[2], tcc[3])
    print(actual)
    assert len(actual) == tcc[0]
    for c in actual:
        assert len(c.iq) == tcc[3]
        assert len(c.weights) == tcc[2]
        assert len(c.stretching_factors) == tcc[2]
        assert (c.grid == tcc[1]).all()
