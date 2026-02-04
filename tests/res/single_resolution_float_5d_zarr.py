import bioio.writers
import os
import shutil
import numpy as np
from bioio_base.types import PhysicalPixelSizes
from qubalab.images.metadata.image_shape import ImageShape
import bioio_ome_zarr.writers


def get_name() -> str:
    return "single_resolution_float_5d.ome.zarr"


def get_path() -> str:
    return os.path.realpath(
        os.path.join(os.path.realpath(__file__), os.pardir, get_name())
    )


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
    return np.random.rand(
        get_shapes()[0].t,
        get_shapes()[0].c,
        get_shapes()[0].z,
        get_shapes()[0].y,
        get_shapes()[0].x,
    )


def _write_image(pixels: np.array):
    ## zarr writer fails if dir exists
    if os.path.exists(get_path()) and os.path.isdir(get_path()):
        shutil.rmtree(get_path())

    zarr = bioio.writers.OMEZarrWriter(
        get_path(),
        dtype=get_dtype(),
        level_shapes=get_shapes()[0].as_tuple(),
        image_name="single_resolution_float_5d",
        physical_pixel_size=[
            1,  ## tczyx
            1,
            get_pixel_size_x_y_in_micrometers(),
            get_pixel_size_x_y_in_micrometers(),
            get_pixel_size_x_y_in_micrometers(),
        ],
        channels=[
            bioio_ome_zarr.writers.Channel(label="Channel " + str(i), color=i)
            for i in range(get_shapes()[0].c)
        ],
    )
    zarr.write_full_volume(input_data=pixels)


pixels = _get_pixels()
_write_image(pixels)
