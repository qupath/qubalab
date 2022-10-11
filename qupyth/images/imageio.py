from . import ImageServer, ImageServerMetadata, PixelCalibration, _validate_block, Region2D

from typing import Tuple, Union
from pathlib import Path
from imageio.v3 import imread

import numpy as np

class ImageIoServer(ImageServer):
    """
    An ImageServer using ImageIO.
    This is used for most 'regular' images (e.g. JPEG, PNG, small TIFF).
    TODO: Support volread
    """

    def __init__(self, path: str, strip_alpha=True, cache_image=True, im: np.ndarray = None):
        super().__init__()
        self._path = path
        self._strip_alpha = strip_alpha
        self._cache_image = cache_image
        self._im = im

    def _load_image(self) -> np.ndarray:
        if self._im is not None:
            return self._im
        im = imread(self._path)
        if self._strip_alpha and im.ndim > 2 and im.shape[2] == 4:
            im = im[:, :, :3]
        if self._cache_image:
            self._im = im
        return im

    def _build_metadata(self) -> ImageServerMetadata:
        # TODO: Read metadata only
        im = self._load_image()
        name = Path(self._path).name if self._path else 'Unnamed image'
        return ImageServerMetadata(
            path=self._path,
            name=name,
            downsamples=(1.0,),
            pixel_calibration=PixelCalibration(),
            shape=im.shape,
            dtype=im.dtype
        )

    def read_block(self, level: int, block: Union[Region2D, Tuple[int, ...]]) -> np.ndarray:
        assert level == 0
        from dataclasses import astuple
        _, x, y, width, height, z, t = astuple(_validate_block(block))
        im = self._load_image()
        if im.ndim == 2 or im.ndim == 2:
            assert z == 0
            assert t == 0
            return im[y:y+height, x:x+width]
        if im.ndim == 4:
            assert t == 0
            return im[y:y+height, x:x+width, :, z]
        return im[y:y+height, x:x+width, :, z, t]

