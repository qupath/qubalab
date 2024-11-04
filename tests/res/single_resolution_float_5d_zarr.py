import bioio.writers
import os
import shutil
import numpy as np
from bioio_base.types import PhysicalPixelSizes
from qubalab.images.metadata.image_shape import ImageShape


def get_name() -> str:
    return "single_resolution_float_5d.ome.zarr"


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
    return pixels[t, c, z, y, x]


def _get_pixels() -> np.array:
    return np.random.rand(get_shapes()[0].t, get_shapes()[0].c, get_shapes()[0].z, get_shapes()[0].y, get_shapes()[0].x)


def _write_image(pixels: np.array):
    ## zarr writer fails if dir exists
    if os.path.exists(get_path()) and os.path.isdir(get_path()):
        shutil.rmtree(get_path())

    zarr = bioio.writers.OmeZarrWriter(get_path())
    zarr.write_image(
        image_data = pixels,
        image_name = "single_resolution_float_5d",
        channel_names = ["Channel " + str(i) for i in range(get_shapes()[0].c)],
        channel_colors = [i for i in range(get_shapes()[0].c)],
        physical_pixel_sizes = PhysicalPixelSizes(
            X = get_pixel_size_x_y_in_micrometers(),
            Y = get_pixel_size_x_y_in_micrometers(),
            Z = get_pixel_size_x_y_in_micrometers())
    )


pixels = _get_pixels()
_write_image(pixels)
