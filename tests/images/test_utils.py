import numpy as np
import imageio.v3 as iio
import base64
from qubalab.images.utils import bytes_to_image, base64_to_image
from ..res import single_resolution_float_5d


def test_uri_to_image():
    z = 0
    t = 0
    path = single_resolution_float_5d.get_path()
    image_shape = single_resolution_float_5d.get_shapes()[0]
    expected_image = np.array(
        [[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for c in range(image_shape.c)]
            for x in range(image_shape.x)]
            for y in range(image_shape.y)],
        single_resolution_float_5d.get_dtype()
    )
    
    image = bytes_to_image(path, False)

    np.testing.assert_array_equal(image, expected_image)


def test_bytes_to_image():
    z = 0
    t = 0
    image_shape = single_resolution_float_5d.get_shapes()[0]
    expected_image = np.array(
        [[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for c in range(image_shape.c)]
            for x in range(image_shape.x)]
            for y in range(image_shape.y)],
        single_resolution_float_5d.get_dtype()
    )
    bytes = iio.imwrite("<bytes>", expected_image, extension=".tiff", photometric='rgb')
    
    image = bytes_to_image(bytes, False)

    np.testing.assert_array_equal(image, expected_image)


def test_base64_to_image():
    z = 0
    t = 0
    image_shape = single_resolution_float_5d.get_shapes()[0]
    expected_image = np.array(
        [[[single_resolution_float_5d.get_pixel_value(x, y, c, z, t)
            for c in range(image_shape.c)]
            for x in range(image_shape.x)]
            for y in range(image_shape.y)],
        single_resolution_float_5d.get_dtype()
    )
    bytes = iio.imwrite("<bytes>", expected_image, extension=".tiff", photometric='rgb')
    base64_image = base64.b64encode(bytes)
    
    image = base64_to_image(base64_image, False)

    np.testing.assert_array_equal(image, expected_image)
