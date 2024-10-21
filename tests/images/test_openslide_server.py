import numpy as np
from qubalab.images.openslide_server import OpenSlideServer
from qubalab.images.region_2d import Region2D
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from ..res import multi_resolution_uint8_3channels


def test_image_name():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())

    name = openslide_server.metadata.name

    assert name == multi_resolution_uint8_3channels.get_name()
    
    openslide_server.close()


def test_image_shapes():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())

    shapes = openslide_server.metadata.shapes

    # only the full resolution can be detected
    assert shapes[0] == multi_resolution_uint8_3channels.get_shapes()[0]
    
    openslide_server.close()



def test_is_rgb():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())

    is_rgb = openslide_server.metadata.is_rgb

    assert is_rgb
    
    openslide_server.close()


def test_image_dtype():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())

    dtype = openslide_server.metadata.dtype

    assert dtype == multi_resolution_uint8_3channels.get_dtype()
    
    openslide_server.close()


def test_number_of_channels_when_alpha_stripped():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())

    n_channels = openslide_server.metadata.n_channels

    assert n_channels == 3
    
    openslide_server.close()


def test_number_of_channels_when_alpha_not_stripped():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path(), strip_alpha=False)

    n_channels = openslide_server.metadata.n_channels

    assert n_channels == 4
    
    openslide_server.close()


def test_number_of_channels_when_single_channel():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path(), single_channel=True)

    n_channels = openslide_server.metadata.n_channels

    assert n_channels == 1
    
    openslide_server.close()


def test_downsamples():
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())

    downsamples = openslide_server.metadata.downsamples

    # only the full resolution can be detected
    assert downsamples[0] == multi_resolution_uint8_3channels.get_downsamples()[0]
    
    openslide_server.close()


def test_read_full_resolution_image():
    level = 0
    full_resolution = multi_resolution_uint8_3channels.get_shapes()[level]
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    openslide_server = OpenSlideServer(multi_resolution_uint8_3channels.get_path())
    expected_pixels = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )

    image = openslide_server.read_region(
        downsample,
        Region2D(width=openslide_server.metadata.width, height=openslide_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)
    
    openslide_server.close()
