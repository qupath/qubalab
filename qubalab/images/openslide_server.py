import numpy as np
from pathlib import Path
from typing import Tuple
from dataclasses import astuple
import warnings
try:
    import openslide
except ImportError as e:
    warnings.warn(f'Unable to import OpenSlide, will try TiffSlide instead')
    try:
        import tiffslide as openslide
    except ImportError as e:
        warnings.warn(f'Unable to import TiffSlide')
from .image_server import ImageServer
from .metadata.image_server_metadata import ImageServerMetadata
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .metadata.image_shape import ImageShape
from .metadata.region_2d import Region2D


class OpenSlideServer(ImageServer):
    """
    An image server that relies on OpenSlide (https://openslide.org/) to read images.

    It can only be used to read RGB images.
    """

    def __init__(self, path: str, strip_alpha=True, single_channel=False, limit_bounds=True, **kwargs):
        """
        Create the server.

        :param path: the local path to the image to open
        :param strip_alpha: whether to strip the alpha channel from the image
        :param single_channel: whether to keep only the first channel of the image
        :param limit_bounds: whether to only consider a rectangle bounding the non-empty region of the slide,
                             if such rectangle is defined in the properties of the image
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(**kwargs)
        self._openslide = openslide.OpenSlide(path)
        self._path = path
        self._strip_alpha = strip_alpha
        self._single_channel = single_channel
        self._limit_bounds = limit_bounds

    def _build_metadata(self) -> ImageServerMetadata:
        if self._single_channel:
            n_channels = 1
        elif self._strip_alpha:
            n_channels = 3
        else:
            n_channels = 4

        full_bounds = (0, 0) + self._openslide.dimensions
        bounds = (self._openslide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X),
                  self._openslide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y),
                  self._openslide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH),
                  self._openslide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT))
        if self._limit_bounds and not any(v is None for v in bounds):
            self._bounds = bounds
        else:
            self._bounds = full_bounds
        if full_bounds:
            shapes = tuple(ImageShape(x=d[0], y=d[1], c=n_channels) for d in self._openslide.level_dimensions)
        else:
            w = self._bounds[3]
            h = self._bounds[2]
            shapes = tuple(ImageShape(x=int(w / d), y=int(h / d), c=n_channels) for d in self._openslide.level_downsamples)

        pixel_width = self._openslide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        pixel_height = self._openslide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if pixel_width is not None and pixel_height is not None:
            calibration = PixelCalibration(
                length_x=PixelLength.create_microns(float(pixel_width)),
                length_y=PixelLength.create_microns(float(pixel_height))
            )
        else:
            calibration = PixelCalibration()

        return ImageServerMetadata(
            path=self._path,
            name=Path(self._path).name,
            shapes=shapes,
            pixel_calibration=calibration,
            is_rgb=True,
            dtype=np.uint8
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        x, y, width, height, z, t = astuple(region)
        assert z == 0
        assert t == 0

        # x and y are provided in the coordinate space of the level, while openslide
        # expect coordinates in the level 0 reference frame
        x = int(x * self.metadata.downsamples[level])
        y = int(y * self.metadata.downsamples[level])

        # Map coordinates to bounding rectangle
        x += self._bounds[0]
        y += self._bounds[1]

        image = self._openslide.read_region((x, y), level, (width, height))
        im = np.asarray(image)
        image.close()

        # Return image, stripping alpha/converting to single-channel if needed
        return im[:, :, :self.metadata.n_channels]

    def close(self):
        self._openslide.close()
