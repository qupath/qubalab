import numpy as np
import imageio
import base64
from .metadata.image_shape import ImageShape


def bytes_to_image(uri: {str, bytes}, is_rgb: bool, shape: ImageShape = None) -> np.ndarray:
    """
    Read the provided bytes (or URI pointing to a binary file) and convert them to a numpy array
    representing an image.

    :param uri: an URI pointing to a binary file, or an array of bytes. This must
                represent a 2-dimensionnal or 3-dimensionnal (with a channel axis) image
    :param is_rgb: whether the expected image should have the RGB format
    :param shape: the shape of the image to open. This will be used to reorder axes.
                  Can be None to not reorder axes
    :returns: a numpy array representing the input image. If shape is defined, this function
              will try to set its dimensions to (c, y, x), but this may not always be the case if two
              dimensions have the same number of elements
    """
    if is_rgb:
        image = imageio.v3.imread(uri)
    else:
        image = imageio.volread(uri, format="tiff")

    if shape is None:
        return image
    else:
        return _reorder_axes(image, shape)


def base64_to_image(data: str, is_rgb: bool, shape: ImageShape = None) -> np.ndarray:
    """
    Read the provided string with the Base64 format and convert it to a numpy array
    representing an image.

    :param uri: a text with the Base64 format. This must represent a 2-dimensionnal
                or 3-dimensionnal (with a channel axis) image
    :param is_rgb: whether the expected image should have the RGB format
    :param shape: the shape of the image to open. This will be used to reorder axes.
                  Can be None to not reorder axes
    :returns: a numpy array representing the input image. If shape is defined, this function
              will try to set its dimensions to (c, y, x), but this may not always be the case if two
              dimensions have the same number of elements
    """
    return bytes_to_image(base64.b64decode(data), is_rgb, shape)


def _reorder_axes(image: np.ndarray, shape: ImageShape):
    """
    Reorder the axes of the provided image to (c, y, x).

    :param image: the image to reorder
    :param shape: the desired shape of the image
    :param shape: the shape of the image to open. This will be used to reorder axes
    :returns: a numpy array representing the input image. This function will try to set
              its dimensions to (c, y, x), but this may not be the case if two
              dimensions have the same number of elements
    """
    x_axis = -1
    y_axis = -1
    c_axis = -1
    for axis, number_of_elements_on_axis in enumerate(image.shape):
        if c_axis == -1 and number_of_elements_on_axis == shape.c:
            c_axis = axis
        elif y_axis == -1 and number_of_elements_on_axis == shape.y:
            y_axis = axis
        elif x_axis == -1 and number_of_elements_on_axis == shape.x:
            x_axis = axis

    if x_axis != -1 and y_axis != -1 and c_axis != -1:
        return np.moveaxis(image, [c_axis, y_axis, x_axis], [0, 1, 2]).copy()
    else:
        return image
