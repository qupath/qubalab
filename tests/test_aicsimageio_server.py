import numpy as np
from qubalab.images.aicsimageio_server import AICSImageIoServer
from qubalab.images.metadata.region_2d import Region2D
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from .res import multi_resolution_uint8_3channels, single_resolution_float_5d, single_resolution_rgb_image


def test_uint8_3channels_image_name():
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())

    name = aicsimageio_server.metadata.name

    assert name == multi_resolution_uint8_3channels.get_name()


def test_uint8_3channels_image_shapes():
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())

    shapes = aicsimageio_server.metadata.shapes

    assert shapes == (multi_resolution_uint8_3channels.get_shapes()[0], )      # The AICSImage library does not properly support pyramids


def test_uint8_3channels_image_pixel_calibration():
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())

    pixel_calibration = aicsimageio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(multi_resolution_uint8_3channels.get_pixel_size_x_y_in_micrometers()),      # The AICSImage library does not currently handle unit attachment
        PixelLength(multi_resolution_uint8_3channels.get_pixel_size_x_y_in_micrometers())
    )


def test_uint8_3channels_image_is_rgb():
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())

    is_rgb = aicsimageio_server.metadata.is_rgb

    assert is_rgb


def test_uint8_3channels_image_dtype():
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())

    dtype = aicsimageio_server.metadata.dtype

    assert dtype == multi_resolution_uint8_3channels.get_dtype()


def test_uint8_3channels_image_downsamples():
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())

    downsamples = aicsimageio_server.metadata.downsamples

    assert downsamples == (multi_resolution_uint8_3channels.get_downsamples()[0], )      # The AICSImage library does not properly support pyramids


def test_read_uint8_3channels_image():
    level = 0
    full_resolution = multi_resolution_uint8_3channels.get_shapes()[level]
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())
    expected_pixels = np.array([[[
        multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, 0),
        multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, 1),
        multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, 2)
    ] for x in range(full_resolution.x)] for y in range(full_resolution.y)], multi_resolution_uint8_3channels.get_dtype())

    image = aicsimageio_server.read_region(
        downsample,
        Region2D(width=aicsimageio_server.metadata.width, height=aicsimageio_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)


def test_read_uint8_3channels_image_with_dask():
    level = 0
    full_resolution = multi_resolution_uint8_3channels.get_shapes()[level]
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    aicsimageio_server = AICSImageIoServer(multi_resolution_uint8_3channels.get_path())
    expected_pixels = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )

    image = aicsimageio_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_pixels)


def test_float_5d_image_name():
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())

    name = aicsimageio_server.metadata.name

    assert name == single_resolution_float_5d.get_name()


def test_float_5d_image_shapes():
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())

    shapes = aicsimageio_server.metadata.shapes

    assert shapes == single_resolution_float_5d.get_shapes()


def test_float_5d_image_pixel_calibration():
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())

    pixel_calibration = aicsimageio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(single_resolution_float_5d.get_pixel_size_x_y_in_micrometers()),      # The AICSImage library does not currently handle unit attachment
        PixelLength(single_resolution_float_5d.get_pixel_size_x_y_in_micrometers())
    )


def test_float_5d_image_is_not_rgb():
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())

    is_rgb = aicsimageio_server.metadata.is_rgb

    assert not(is_rgb)


def test_float_5d_image_dtype():
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())

    dtype = aicsimageio_server.metadata.dtype

    assert dtype == single_resolution_float_5d.get_dtype()


def test_float_5d_image_downsamples():
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())

    downsamples = aicsimageio_server.metadata.downsamples

    assert downsamples == single_resolution_float_5d.get_downsamples()


def test_read_float_5d_image():
    full_resolution = single_resolution_float_5d.get_shapes()[0]
    z = int(full_resolution.z / 2)
    t = int(full_resolution.t / 2)
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())
    expected_pixels = np.array(
        [[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for c in range(full_resolution.c)]
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)],
        single_resolution_float_5d.get_dtype()
    )

    image = aicsimageio_server.read_region(
        1,
        Region2D(width=aicsimageio_server.metadata.width, height=aicsimageio_server.metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_pixels)


def test_read_float_5d_image_with_dask():
    full_resolution = single_resolution_float_5d.get_shapes()[0]
    aicsimageio_server = AICSImageIoServer(single_resolution_float_5d.get_path())
    expected_pixels = np.array(
        [[[[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for z in range(full_resolution.z)]
            for c in range(full_resolution.c)]
            for t in range(full_resolution.t)],
        single_resolution_float_5d.get_dtype()
    )

    image = aicsimageio_server.level_to_dask(0).compute()

    np.testing.assert_array_equal(image, expected_pixels)


def test_rgb_image_name():
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())

    name = aicsimageio_server.metadata.name

    assert name == single_resolution_rgb_image.get_name()


def test_rgb_image_shapes():
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())

    shapes = aicsimageio_server.metadata.shapes

    assert shapes == single_resolution_rgb_image.get_shapes()


def test_rgb_image_pixel_calibration():
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())

    pixel_calibration = aicsimageio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(single_resolution_rgb_image.get_pixel_size_x_y_in_micrometers()),      # The AICSImage library does not currently handle unit attachment
        PixelLength(single_resolution_rgb_image.get_pixel_size_x_y_in_micrometers())
    )


def test_rgb_image_is_rgb():
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())

    is_rgb = aicsimageio_server.metadata.is_rgb

    assert is_rgb


def test_rgb_image_dtype():
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())

    dtype = aicsimageio_server.metadata.dtype

    assert dtype == single_resolution_rgb_image.get_dtype()


def test_rgb_image_downsamples():
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())

    downsamples = aicsimageio_server.metadata.downsamples

    assert downsamples == single_resolution_rgb_image.get_downsamples()


def test_read_rgb_image():
    full_resolution = single_resolution_rgb_image.get_shapes()[0]
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())
    expected_pixels = np.array(
        [[[single_resolution_rgb_image.get_pixel_value(x, y, c)
            for c in range(full_resolution.c)]
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)],
        single_resolution_rgb_image.get_dtype()
    )

    image = aicsimageio_server.read_region(
        1,
        Region2D(width=aicsimageio_server.metadata.width, height=aicsimageio_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)


def test_read_rgb_image_with_dask():
    full_resolution = single_resolution_rgb_image.get_shapes()[0]
    aicsimageio_server = AICSImageIoServer(single_resolution_rgb_image.get_path())
    expected_pixels = np.array(
        [[[single_resolution_rgb_image.get_pixel_value(x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        single_resolution_rgb_image.get_dtype()
    )

    image = aicsimageio_server.level_to_dask(0).compute()

    np.testing.assert_array_equal(image, expected_pixels)
