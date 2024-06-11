import numpy as np
from qubalab.images.openslide_server import OpenSlideServer
from qubalab.images.metadata.region_2d import Region2D
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from .res import multi_resolution_rgb


def test_image_name():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    name = openslide_server.metadata.name

    assert name == multi_resolution_rgb.get_name()


def test_image_shapes():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    shapes = openslide_server.metadata.shapes

    assert shapes == multi_resolution_rgb.get_shapes()


def test_image_pixel_calibration():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    pixel_calibration = openslide_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength.create_microns(multi_resolution_rgb.get_pixel_size_x_y_in_micrometers()),
        PixelLength.create_microns(multi_resolution_rgb.get_pixel_size_x_y_in_micrometers())
    )


def test_is_rgb():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    is_rgb = openslide_server.metadata.is_rgb

    assert is_rgb


def test_image_dtype():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    dtype = openslide_server.metadata.dtype

    assert dtype == np.uint8


def test_image_shape():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    shape = openslide_server.metadata.shape

    assert shape == multi_resolution_rgb.get_shapes()[0]


def test_image_width():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    width = openslide_server.metadata.width

    assert width == multi_resolution_rgb.get_shapes()[0].x


def test_image_height():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    height = openslide_server.metadata.height

    assert height == multi_resolution_rgb.get_shapes()[0].y


def test_number_of_channels_when_alpha_stripped():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    n_channels = openslide_server.metadata.n_channels

    assert n_channels == 3


def test_number_of_channels_when_alpha_not_stripped():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path(), strip_alpha=False)

    n_channels = openslide_server.metadata.n_channels

    assert n_channels == 4


def test_number_of_channels_when_single_channel():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path(), single_channel=True)

    n_channels = openslide_server.metadata.n_channels

    assert n_channels == 1


def test_number_of_timepoints():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    n_timepoints = openslide_server.metadata.n_timepoints

    assert n_timepoints == 1


def test_number_of_z_slices():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    n_z_slices = openslide_server.metadata.n_z_slices

    assert n_z_slices == 1


def test_number_of_resolutions():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    n_resolutions = openslide_server.metadata.n_resolutions

    assert n_resolutions == len(multi_resolution_rgb.get_shapes())


def test_downsamples():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())

    downsamples = openslide_server.metadata.downsamples

    assert downsamples == multi_resolution_rgb.get_downsamples()


def test_read_image():
    openslide_server = OpenSlideServer(multi_resolution_rgb.get_path())
    expected_pixels = np.array([[[
        multi_resolution_rgb.get_pixel_value(x, y, 0),
        multi_resolution_rgb.get_pixel_value(x, y, 1),
        multi_resolution_rgb.get_pixel_value(x, y, 2)
    ] for x in range(multi_resolution_rgb.get_shapes()[0].x)] for y in range(multi_resolution_rgb.get_shapes()[0].y)], np.uint8)

    image = openslide_server.read_region(
        1,
        Region2D(width=openslide_server.metadata.width, height=openslide_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)

