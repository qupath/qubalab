import pytest

from qupyth.images.servers import ImageShape

from ..images import *

import numpy as np

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


def test_pixel_length():
    len1 = PixelLength()
    assert len1 == PixelLength()
    assert len1.is_default
    assert len1.length == 1.0 and len1.unit == 'pixels'
    len2 = PixelLength.create_microns(0.25)
    assert len2 != len1
    assert len2.length == 0.25
    assert len2.unit == 'micrometer'

def test_pixel_calibration():
    cal = PixelCalibration()
    assert not cal.is_calibrated
    cal2 = PixelCalibration(length_x=PixelLength())
    assert cal == cal2
    cal3 = PixelCalibration(length_x=PixelLength.create_microns(0.25))
    assert cal != cal3
    assert cal3.is_calibrated


def test_image_shape():
    x = 1
    y = 2
    z = 3
    t = 4
    c = 5
    shape = ImageShape(x=x, y=y, t=t, z=z, c=c)
    assert shape.as_tuple('yx') == (y, x)
    assert shape.as_tuple() == (t, c, z, y, x)
    shape2 = ImageShape.from_tczyx(t, c, z, y, x)
    assert shape == shape2
    assert shape2.as_tuple() == (t, c, z, y, x)
    assert shape2.as_tuple() != (c, t, z, y, x)
    

def test_resize():
    from ..images.servers import _resize
    rng = np.random.default_rng(100)
    size_orig = (50, 65)
    for dt in [np.uint8, np.int8, np.uint16, np.float32, np.float64]:
        for n_channels in range(1, 5):
            im = rng.normal(loc=100, scale=10, size=size_orig + (n_channels,))
            im = im.astype(dt)
            for target_size in ((40, 40), (50, 25), (60, 65), (100, 100)):
                im2 = _resize(im, target_size=target_size)
                assert im2.shape[0] == target_size[1]
                assert im2.shape[1] == target_size[0]
                assert im.dtype == im2.dtype
