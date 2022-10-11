from . import ImageServer, ImageServerMetadata, PixelCalibration, Units
from .servers import _validate_block

from typing import Tuple
from dataclasses import astuple

import openslide
import numpy as np
from pathlib import Path

class OpenSlideServer(ImageServer):

    def __init__(self, path: str, strip_alpha=True, single_channel=False, limit_bounds=True):
        super().__init__()
        self._osr = openslide.open_slide(path)
        self._path = path
        self._strip_alpha = strip_alpha
        self._single_channel = single_channel
        self._limit_bounds = limit_bounds

    def _build_metadata(self) -> ImageServerMetadata:
        bounds = (self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_X),
                  self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y),
                  self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH),
                  self._osr.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT))
        if self._limit_bounds and not any(v is None for v in bounds):
            self._bounds = bounds
        else:
            self._bounds = (0, 0) + self._osr.dimensions
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
            cal = PixelCalibration(pixel_width=pixel_width, pixel_height=pixel_height, units=Units.MICRONS)
        else:
            cal = PixelCalibration()
        return ImageServerMetadata(
            path=self._path,
            name=name,
            downsamples=tuple(self._osr.level_downsamples),
            pixel_calibration=cal,
            shape=(self._bounds[3], self._bounds[2], n_channels),
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
