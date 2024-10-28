import tifffile
import os
import numpy as np
from qubalab.images.metadata.image_shape import ImageShape


def get_name() -> str:
    return "multi_resolution_uint8_3channels.ome.tif"


def get_path() -> str:
    return os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, get_name()))


def get_shapes() -> tuple[ImageShape, ...]:
    return (
        ImageShape(512, 256, c=3),
        ImageShape(256, 128, c=3),
        ImageShape(128, 64, c=3),
        ImageShape(64, 32, c=3),
        ImageShape(32, 16, c=3),
        ImageShape(16, 8, c=3),
        ImageShape(8, 4, c=3),
    )


def get_pixel_size_x_y_in_micrometers() -> float:
    return 0.25


def get_dtype():
    return np.uint8


def get_downsamples() -> tuple[float, ...]:
    return tuple([get_shapes()[0].x / shape.x for shape in get_shapes()])


def get_pixel_value(downsample: float, x: int, y: int, c: int) -> int:
    return pixels[..., ::int(downsample), ::int(downsample)][c, y, x]


def _get_pixels() -> np.array:
    width = get_shapes()[0].x
    height = get_shapes()[0].y

    pixels = []
    for c in range(get_shapes()[0].c):
        channel = []
        for y in range(height):
            row = []
            for x in range(width):
                if c == 0:
                    row.append(int(255 * x / width))
                elif c == 1:
                    row.append(int(255 * y / height))
                else:
                    row.append(int(255 * x / width * y / height))
            channel.append(row)
        pixels.append(channel)
    return np.array(pixels, get_dtype())


def _write_image(pixels: np.array):
    metadata = {
        'PhysicalSizeX': get_pixel_size_x_y_in_micrometers(),
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': get_pixel_size_x_y_in_micrometers(),
        'PhysicalSizeYUnit': 'µm'
    }

    with tifffile.TiffWriter(get_path()) as tif:
        number_of_subresolutions = len(get_downsamples())-1
        number_of_pixels_per_cm = 1e4 / get_pixel_size_x_y_in_micrometers()

        tif.write(
            pixels,
            metadata=metadata,
            subifds=number_of_subresolutions,
            resolution=(number_of_pixels_per_cm, number_of_pixels_per_cm),
            resolutionunit=3,    # indicate that the resolution above is in cm^-1,
            photometric='rgb',
            tile=(128, 128)
        )

        # Write sub resolutions
        for downsample in get_downsamples():
            if downsample > 1:
                tif.write(
                    pixels[..., ::int(downsample), ::int(downsample)],
                    metadata=metadata,
                    subfiletype=1,      # indicate that the image is part of a multi-page image
                    resolution=(number_of_pixels_per_cm / downsample, number_of_pixels_per_cm / downsample),
                    resolutionunit=3,
                    photometric='rgb', 
                    tile=(128, 128)
                )


pixels = _get_pixels()
_write_image(pixels)
