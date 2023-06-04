from . import ImageServer, ImageServerMetadata, PixelCalibration, PixelLength, ImageShape
from .servers import _validate_block

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

import numpy as np
from pathlib import Path


class OpenSlideServer(ImageServer):

    def __init__(self, path: str, strip_alpha=True, single_channel=False, limit_bounds=True, **kwargs):
        super().__init__(**kwargs)
        self._osr = openslide.OpenSlide(path)
        self._path = path
        self._strip_alpha = strip_alpha
        self._single_channel = single_channel
        self._limit_bounds = limit_bounds

    def _build_metadata(self) -> ImageServerMetadata:
        full_bounds = (0, 0) + self._osr.dimensions
        bounds = (self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_X),
                  self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y),
                  self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH),
                  self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT))
        if self._limit_bounds and not any(v is None for v in bounds):
            self._bounds = bounds
        else:
            self._bounds = full_bounds
        if self._single_channel:
            n_channels = 1
        elif self._strip_alpha:
            n_channels = 3
        else:
            n_channels = 4
        name = Path(self._path).name if self._path else 'Unnamed image'
        pixel_width = self._osr.properties.get('openslide.mpp-x')
        pixel_height = self._osr.properties.get('openslide.mpp-y')
        if pixel_width is not None and pixel_height is not None:
            cal = PixelCalibration(
                length_x=PixelLength.create_microns(float(pixel_width)),
                length_y=PixelLength.create_microns(float(pixel_height))
            )
        else:
            cal = PixelCalibration()

        # Determine shapes for all levels
        if full_bounds:
            shapes = tuple(ImageShape(x=d[0], y=d[1], c=n_channels) for d in self._osr.level_dimensions)
        else:
            w = self._bounds[3]
            h = self._bounds[2]
            shapes = tuple(ImageShape(x=int(w / d), y=int(h / d), c=n_channels) for d in self._osr.level_downsamples)

        return ImageServerMetadata(
            path=self._path,
            name=name,
            shapes=shapes,
            pixel_calibration=cal,
            is_rgb=True,
            dtype=np.uint8
        )

    def read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        _, x, y, width, height, z, t = astuple(_validate_block(block))
        assert z == 0
        assert t == 0
        # Request pixels
        level_downsample = self.downsamples[level]
        location = (int(x * level_downsample), int(y * level_downsample))
        image = self._osr.read_region(location, level, (width, height))
        im = np.asarray(image)
        image.close()
        # Return image, stripping alpha/converting to single-channel if needed
        return im[:, :, :self.n_channels]

    def close(self):
        self._osr.close()
