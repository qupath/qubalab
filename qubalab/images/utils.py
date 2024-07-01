import numpy as np
import imageio
import base64


def bytes_to_image(uri: {str, bytes}, is_rgb: bool) -> np.ndarray:
    """
    Read the provided bytes (or URI pointing to a binary file) and convert them to a numpy array
    representing an image.

    :param uri: an URI pointing to a binary file, or an array of bytes
    :param is_rgb: whether the expected image should have the RGB format
    :returns: a numpy array with dimensions (y, x, c) representing the image
    """
    return imageio.v3.imread(uri) if is_rgb else imageio.volread(uri)


def base64_to_image(data: str, is_rgb: bool) -> np.ndarray:
    """
    Read the provided string with the Base64 format and convert them to a numpy array
    representing an image.

    :param uri: a text with the Base64 format
    :param is_rgb: whether the expected image should have the RGB format
    :returns: a numpy array with dimensions (y, x, c) representing the image
    """
    return bytes_to_image(base64.b64decode(data), is_rgb)