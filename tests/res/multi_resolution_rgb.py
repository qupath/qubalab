import tifffile
import os
import numpy as np
from qubalab.images.metadata.image_shape import ImageShape


def get_name():
    return "multi_resolution_rgb.ome.tif"


def get_path():
    return os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, get_name()))


def get_shapes():
    return (
        ImageShape(512, 256, c=3),
        ImageShape(256, 128, c=3),
        ImageShape(128, 64, c=3),
        ImageShape(64, 32, c=3),
        ImageShape(32, 16, c=3),
        ImageShape(16, 8, c=3),
        ImageShape(8, 4, c=3)
    )


def get_pixel_size_x_y_in_micrometers():
    return 0.25


def get_downsamples():
    return tuple([get_shapes()[0].x / shape.x for shape in get_shapes()])


def get_pixel_value(x, y, c):
    return pixels[c, y, x]


def _get_pixels():
    width = get_shapes()[0].x
    height = get_shapes()[0].y

    pixels = []
    for c in range(3):
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
    return np.array(pixels, np.uint8)


def _write_image(pixels):
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
            resolutionunit=3    # indicate that the resolution above is in cm^-1
        )

        # Write sub resolutions
        for downsample in get_downsamples():
            if downsample > 1:
                tif.write(
                    pixels[..., ::int(downsample), ::int(downsample)],
                    metadata=metadata,
                    subfiletype=1,      # indicate that the image is part of a multi-page image
                    resolution=(number_of_pixels_per_cm / downsample, number_of_pixels_per_cm / downsample),
                    resolutionunit=3
                )


pixels = _get_pixels()
_write_image(pixels)
