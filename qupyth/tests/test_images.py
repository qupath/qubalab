import pytest

from ..images import *


def test_region2D_defaults():
    region = Region2D()
    assert region.x == 0
    assert region.y == 0
    assert region.width == -1
    assert region.height == -1
    assert region.z == 0
    assert region.t == 0


def test_region2D_downsample_same():
    region = Region2D(downsample=1, x=1, y=2, width=3, height=4)
    region2 = region.downsample_region()
    assert region == region2


def test_region2D_downsample_different():
    region = Region2D(downsample=2, x=1, y=2, width=3, height=4)
    region2 = region.downsample_region()
    assert region != region2
    assert region2.x == 0
    assert region2.y == 1
    assert region2.width == 2
    assert region2.height == 2
    assert region2.z == 0
    assert region2.t == 0
    

def test_region_downsample_zero():
    with pytest.raises(ValueError):
        region = Region2D(downsample=0, x=1, y=2, width=3, height=4)
        region2 = region.downsample_region()
