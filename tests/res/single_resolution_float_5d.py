import tifffile
import os
import numpy as np
from qubalab.images.metadata.image_shape import ImageShape


def get_name() -> str:
    return "single_resolution_float_5d.ome.tif"


def get_path() -> str:
    return os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, get_name()))


def get_shapes() -> tuple[ImageShape, ...]:
    return (ImageShape(32, 16, 10, 5, 15),)


def get_pixel_size_x_y_in_micrometers() -> float:
    return 0.5


def get_dtype():
    return np.float64


def get_downsamples() -> tuple[float, ...]:
    return tuple([get_shapes()[0].x / shape.x for shape in get_shapes()])


def get_pixel_value(x: int, y: int, c: int, z: int, t: int) -> int:
    return pixels[t, z, c, y, x]


def _get_pixels() -> np.array:
    return np.random.rand(get_shapes()[0].t, get_shapes()[0].z, get_shapes()[0].c, get_shapes()[0].y, get_shapes()[0].x)


def _write_image(pixels: np.array):
    metadata = {
        'PhysicalSizeX': get_pixel_size_x_y_in_micrometers(),
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': get_pixel_size_x_y_in_micrometers(),
        'PhysicalSizeYUnit': 'µm'
    }

    with tifffile.TiffWriter(get_path()) as tif:
        number_of_pixels_per_cm = 1e4 / get_pixel_size_x_y_in_micrometers()

        tif.write(
            pixels,
            metadata=metadata,
            resolution=(number_of_pixels_per_cm, number_of_pixels_per_cm),
            resolutionunit=3    # indicate that the resolution above is in cm^-1
        )


pixels = _get_pixels()
_write_image(pixels)
