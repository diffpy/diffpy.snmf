import pytest
from diffpy.snmf.optimization import mkr_box

tm = [
    ([[[1,0],[0,1]],[1,1],0,1],[0,0]),
    ([[[1,0],[0,1]],[1,1],-1,1],[-1,-1]),
    ([[[2,-1,0],[-1,2,-1],[0,-1,2]],[1,1,1],-10,12],[-1.5,-2,-1.5]),
    ([[[2,-1,0],[-1,2,-1],[0,-1,2]],[1,-1,-1],-10,12],[0,1,1])
]


@pytest.mark.parametrize('tm',tm)
def test_mkr_box(tm):
    expected = tm[1]
    actual = mkr_box(tm[0][0],tm[0][1],tm[0][2],tm[0][3])
    assert (actual == pytest.approx(expected))
