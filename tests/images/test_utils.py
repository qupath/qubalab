import numpy as np
import imageio.v3 as iio
import base64
from qubalab.images.utils import bytes_to_image, base64_to_image
from ..res import multi_resolution_uint8_3channels


def test_uri_to_image():
    level = 0
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    path = multi_resolution_uint8_3channels.get_path()
    image_shape = multi_resolution_uint8_3channels.get_shapes()[level]
    expected_image = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(image_shape.x)]
            for y in range(image_shape.y)]
            for c in range(image_shape.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )
    
    image = bytes_to_image(path, False, image_shape)

    np.testing.assert_array_equal(image, expected_image)


def test_bytes_to_image():
    level = 0
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    image_shape = multi_resolution_uint8_3channels.get_shapes()[level]
    expected_image = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(image_shape.x)]
            for y in range(image_shape.y)]
            for c in range(image_shape.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )
    bytes = iio.imwrite("<bytes>", expected_image, extension=".tiff", photometric='rgb')
    
    image = bytes_to_image(bytes, False, image_shape)

    np.testing.assert_array_equal(image, expected_image)


def test_base64_to_image():
    level = 0
    downsample = multi_resolution_uint8_3channels.get_downsamples()[level]
    image_shape = multi_resolution_uint8_3channels.get_shapes()[level]
    expected_image = np.array(
        [[[multi_resolution_uint8_3channels.get_pixel_value(downsample, x, y, c)
            for x in range(image_shape.x)]
            for y in range(image_shape.y)]
            for c in range(image_shape.c)],
        multi_resolution_uint8_3channels.get_dtype()
    )
    bytes = iio.imwrite("<bytes>", expected_image, extension=".tiff", photometric='rgb')
    base64_image = base64.b64encode(bytes)
    
    image = base64_to_image(base64_image, False, image_shape)

    np.testing.assert_array_equal(image, expected_image)
