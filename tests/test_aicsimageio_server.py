import numpy as np
from qubalab.images.aicsimageio_server import AICSImageIoServer
from qubalab.images.metadata.region_2d import Region2D
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from .res import multi_resolution_rgb


def test_image_name():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())

    name = aicsimageio_server.metadata.name

    assert name == multi_resolution_rgb.get_name()


def test_image_shapes():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())

    shapes = aicsimageio_server.metadata.shapes

    assert shapes == (multi_resolution_rgb.get_shapes()[0], )      # The AICSImage library does not properly support pyramids


def test_image_pixel_calibration():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())

    pixel_calibration = aicsimageio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(multi_resolution_rgb.get_pixel_size_x_y_in_micrometers()),      # The AICSImage library does not currently handle unit attachment
        PixelLength(multi_resolution_rgb.get_pixel_size_x_y_in_micrometers())
    )


def test_is_rgb():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())

    is_rgb = aicsimageio_server.metadata.is_rgb

    assert is_rgb


def test_image_dtype():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())

    dtype = aicsimageio_server.metadata.dtype

    assert dtype == np.uint8


def test_downsamples():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())

    downsamples = aicsimageio_server.metadata.downsamples

    assert downsamples == (multi_resolution_rgb.get_downsamples()[0], )      # The AICSImage library does not properly support pyramids


def test_read_image():
    aicsimageio_server = AICSImageIoServer(multi_resolution_rgb.get_path())
    expected_pixels = np.array([[[
        multi_resolution_rgb.get_pixel_value(x, y, 0),
        multi_resolution_rgb.get_pixel_value(x, y, 1),
        multi_resolution_rgb.get_pixel_value(x, y, 2)
    ] for x in range(multi_resolution_rgb.get_shapes()[0].x)] for y in range(multi_resolution_rgb.get_shapes()[0].y)], np.uint8)

    image = aicsimageio_server.read_region(
        1,
        Region2D(width=aicsimageio_server.metadata.width, height=aicsimageio_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)
    

#TODO: add test for 5D image, investigate S dimension