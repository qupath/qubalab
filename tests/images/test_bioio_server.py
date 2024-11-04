import numpy as np
from qubalab.images.bioio_server import BioIOServer
from qubalab.images.region_2d import Region2D
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from ..res import multi_resolution_uint8_3channels, single_resolution_float_5d, single_resolution_float_5d_zarr, single_resolution_rgb_image


def test_uint8_3channels_image_name():
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())

    name = bioio_server.metadata.name

    assert name == multi_resolution_uint8_3channels.get_name()

    bioio_server.close()


def test_uint8_3channels_image_shapes():
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())

    shapes = bioio_server.metadata.shapes

    assert shapes == (multi_resolution_uint8_3channels.get_shapes()[0], )      # The BioIO library does not properly support pyramids

    bioio_server.close()


def test_uint8_3channels_image_pixel_calibration():
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())

    pixel_calibration = bioio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(multi_resolution_uint8_3channels.get_pixel_size_x_y_in_micrometers()),      # The BioIO library does not currently handle unit attachment
        PixelLength(multi_resolution_uint8_3channels.get_pixel_size_x_y_in_micrometers())
    )

    bioio_server.close()


def test_uint8_3channels_image_is_rgb():
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())

    is_rgb = bioio_server.metadata.is_rgb

    assert is_rgb

    bioio_server.close()


def test_uint8_3channels_image_dtype():
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())

    dtype = bioio_server.metadata.dtype

    assert dtype == multi_resolution_uint8_3channels.get_dtype()

    bioio_server.close()


def test_uint8_3channels_image_downsamples():
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())

    downsamples = bioio_server.metadata.downsamples

    assert downsamples == (multi_resolution_uint8_3channels.get_downsamples()[0], )      # The BioIO library does not properly support pyramids

    bioio_server.close()


def test_read_uint8_3channels_image():
    level = 0
    full_resolution = multi_resolution_uint8_3channels.get_shapes()[level]
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())
    expected_pixels = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )

    image = bioio_server.read_region(
        downsample,
        Region2D(width=bioio_server.metadata.width, height=bioio_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()


def test_read_uint8_3channels_image_with_dask():
    level = 0
    full_resolution = multi_resolution_uint8_3channels.get_shapes()[level]
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    bioio_server = BioIOServer(multi_resolution_uint8_3channels.get_path())
    expected_pixels = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )

    image = bioio_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()


def test_float_5d_image_name():
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())

    name = bioio_server.metadata.name

    assert name == single_resolution_float_5d.get_name()

    bioio_server.close()


def test_float_5d_image_shapes():
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())

    shapes = bioio_server.metadata.shapes

    assert shapes == single_resolution_float_5d.get_shapes()

    bioio_server.close()


def test_float_5d_image_pixel_calibration():
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())

    pixel_calibration = bioio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(single_resolution_float_5d.get_pixel_size_x_y_in_micrometers()),      # The BioIO library does not currently handle unit attachment
        PixelLength(single_resolution_float_5d.get_pixel_size_x_y_in_micrometers())
    )

    bioio_server.close()


def test_float_5d_image_is_not_rgb():
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())

    is_rgb = bioio_server.metadata.is_rgb

    assert not(is_rgb)

    bioio_server.close()


def test_float_5d_image_dtype():
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())

    dtype = bioio_server.metadata.dtype

    assert dtype == single_resolution_float_5d.get_dtype()

    bioio_server.close()


def test_float_5d_image_downsamples():
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())

    downsamples = bioio_server.metadata.downsamples

    assert downsamples == single_resolution_float_5d.get_downsamples()

    bioio_server.close()


def test_read_float_5d_image():
    full_resolution = single_resolution_float_5d.get_shapes()[0]
    z = int(full_resolution.z / 2)
    t = int(full_resolution.t / 2)
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())
    expected_pixels = np.array(
        [[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        single_resolution_float_5d.get_dtype()
    )

    image = bioio_server.read_region(
        1,
        Region2D(width=bioio_server.metadata.width, height=bioio_server.metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()


def test_read_float_5d_image_with_dask():
    full_resolution = single_resolution_float_5d.get_shapes()[0]
    bioio_server = BioIOServer(single_resolution_float_5d.get_path())
    expected_pixels = np.array(
        [[[[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for z in range(full_resolution.z)]
            for c in range(full_resolution.c)]
            for t in range(full_resolution.t)],
        single_resolution_float_5d.get_dtype()
    )

    image = bioio_server.level_to_dask(0).compute()

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()



def test_float_5d_zarr_image_name():
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())

    name = bioio_server.metadata.name

    assert name == single_resolution_float_5d_zarr.get_name()

    bioio_server.close()


def test_float_5d_zarr_image_shapes():
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())

    shapes = bioio_server.metadata.shapes

    assert shapes == single_resolution_float_5d_zarr.get_shapes()

    bioio_server.close()


def test_float_5d_zarr_image_pixel_calibration():
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())

    pixel_calibration = bioio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(single_resolution_float_5d_zarr.get_pixel_size_x_y_in_micrometers()),      # The BioIO library does not currently handle unit attachment
        PixelLength(single_resolution_float_5d_zarr.get_pixel_size_x_y_in_micrometers()),
        PixelLength(single_resolution_float_5d_zarr.get_pixel_size_x_y_in_micrometers())
    )

    bioio_server.close()


def test_float_5d_zarr_image_is_not_rgb():
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())

    is_rgb = bioio_server.metadata.is_rgb

    assert not(is_rgb)

    bioio_server.close()


def test_float_5d_zarr_image_dtype():
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())

    dtype = bioio_server.metadata.dtype

    assert dtype == single_resolution_float_5d_zarr.get_dtype()

    bioio_server.close()


def test_float_5d_zarr_image_downsamples():
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())

    downsamples = bioio_server.metadata.downsamples

    assert downsamples == single_resolution_float_5d_zarr.get_downsamples()

    bioio_server.close()


def test_read_float_5d_zarr_image():
    full_resolution = single_resolution_float_5d_zarr.get_shapes()[0]
    z = int(full_resolution.z / 2)
    t = int(full_resolution.t / 2)
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())
    expected_pixels = np.array(
        [[[single_resolution_float_5d_zarr.get_pixel_value(x, y, c, z, t)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        single_resolution_float_5d_zarr.get_dtype()
    )

    image = bioio_server.read_region(
        1,
        Region2D(width=bioio_server.metadata.width, height=bioio_server.metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()


def test_read_float_5d_zarr_image_with_dask():
    full_resolution = single_resolution_float_5d_zarr.get_shapes()[0]
    bioio_server = BioIOServer(single_resolution_float_5d_zarr.get_path())
    expected_pixels = np.array(
        [[[[[single_resolution_float_5d_zarr.get_pixel_value(x, y, c, z, t)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for z in range(full_resolution.z)]
            for c in range(full_resolution.c)]
            for t in range(full_resolution.t)],
        single_resolution_float_5d_zarr.get_dtype()
    )

    image = bioio_server.level_to_dask(0).compute()

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()


def test_rgb_image_name():
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())

    name = bioio_server.metadata.name

    assert name == single_resolution_rgb_image.get_name()

    bioio_server.close()


def test_rgb_image_shapes():
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())

    shapes = bioio_server.metadata.shapes

    assert shapes == single_resolution_rgb_image.get_shapes()

    bioio_server.close()


def test_rgb_image_pixel_calibration():
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())

    pixel_calibration = bioio_server.metadata.pixel_calibration

    assert pixel_calibration == PixelCalibration(
        PixelLength(single_resolution_rgb_image.get_pixel_size_x_y_in_micrometers()),      # The BioIO library does not currently handle unit attachment
        PixelLength(single_resolution_rgb_image.get_pixel_size_x_y_in_micrometers())
    )

    bioio_server.close()


def test_rgb_image_is_rgb():
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())

    is_rgb = bioio_server.metadata.is_rgb

    assert is_rgb

    bioio_server.close()


def test_rgb_image_dtype():
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())

    dtype = bioio_server.metadata.dtype

    assert dtype == single_resolution_rgb_image.get_dtype()

    bioio_server.close()


def test_rgb_image_downsamples():
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())

    downsamples = bioio_server.metadata.downsamples

    assert downsamples == single_resolution_rgb_image.get_downsamples()

    bioio_server.close()


def test_read_rgb_image():
    full_resolution = single_resolution_rgb_image.get_shapes()[0]
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())
    expected_pixels = np.array(
        [[[single_resolution_rgb_image.get_pixel_value(x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        single_resolution_rgb_image.get_dtype()
    )

    image = bioio_server.read_region(
        1,
        Region2D(width=bioio_server.metadata.width, height=bioio_server.metadata.height)
    )

    np.testing.assert_array_equal(image, expected_pixels)

    bioio_server.close()


def test_read_rgb_image_with_dask():
    full_resolution = single_resolution_rgb_image.get_shapes()[0]
    bioio_server = BioIOServer(single_resolution_rgb_image.get_path())
    expected_pixels = np.array(
        [[[single_resolution_rgb_image.get_pixel_value(x, y, c)
            for x in range(full_resolution.x)]
            for y in range(full_resolution.y)]
            for c in range(full_resolution.c)],
        single_resolution_rgb_image.get_dtype()
    )

    image = bioio_server.level_to_dask(0).compute()

    np.testing.assert_array_equal(image, expected_pixels)
    
    bioio_server.close()
