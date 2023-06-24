import pytest
from diffpy.snmf.optimization import mkr_box

tm = [
    ([[[1,0],[0,1]],[1,1],0,1],[0,0])


]


@pytest.mark.parametrize('tm',tm)
def test_mkr_box(tm):
    expected = tm[1]
    actual = mkr_box(tm[0][0],tm[0][1],tm[0][2],tm[0][3])
    assert (actual == pytest.approx(expected))
