import pytest
from qubalab.images.region_2d import Region2D


def test_default_values():
    expected_region = Region2D(downsample = None, x = 0, y = 0, width = -1, height = -1, z = 0, t = 0)

    region = Region2D()

    assert expected_region == region


def test_downsampled_region_with_same_factor():
    region = Region2D(downsample=1, x=1, y=2, width=3, height=4)

    downsampled_region = region.downsample_region()
    
    assert region == downsampled_region


def test_downsampled_region_with_different_factor():
    region = Region2D(downsample=2, x=1, y=2, width=3, height=4)
    expected_downsampled_region = Region2D(x=0, y=1, width=2, height=2, z=0, t=0)

    downsampled_region = region.downsample_region()
    
    assert downsampled_region == expected_downsampled_region


def test_scaled_region_with_different_factor():
    region = Region2D(x=1, y=2, width=3, height=4)
    expected_scaled_region = Region2D(x=2, y=4, width=6, height=8, z=0, t=0)

    scaled_region = region.scale_region(2)
    
    assert scaled_region == expected_scaled_region


def test_downsampled_region_with_factor_of_zero():
    region = Region2D(downsample=0, x=1, y=2, width=3, height=4)

    with pytest.raises(ValueError):
        region.downsample_region()