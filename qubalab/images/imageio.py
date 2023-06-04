from . import ImageServer, ImageServerMetadata, Region2D, ImageShape
from .servers import _validate_block

from typing import Tuple, Union
from os import PathLike
from pathlib import Path

import numpy as np


class BasicServer(ImageServer):
    """
    An ImageServer that reads an image with ImageIO, and/or wraps an existing numpy array.
    This is used for most 'regular' images (e.g. JPEG, PNG, small TIFF).

    Note: This is quite experimental and limited. It doesn't support physical pixel sizes, z-stacks or time series.
    """

    def __init__(self, path: Union[str, PathLike] = None, strip_alpha=True, cache_image=True, im: np.ndarray = None, **kwargs):
        super().__init__(**kwargs)
        self._path = path
        self._strip_alpha = strip_alpha
        self._cache_image = cache_image
        self._im = im

    def _load_image(self) -> np.ndarray:
        if self._im is not None:
            return self._im
        from imageio.v3 import imread
        im = imread(self._path)
        if im.ndim < 2:
            raise ValueError('Image must have at least two dimensions')
        elif im.ndim == 3:
            if im.shape[0] < np.minimum(im.shape[1], im.shape[2]):
                print('First dimension smaller than the others - assuming it is channels')
                im = np.moveaxis(im, 0, -1)
            if self._strip_alpha and im.shape[-1] == 4:
                im = im[..., :3]
        elif im.ndim > 3:
            raise ValueError('Image must be 2D, or 2D + channels')
        if self._cache_image:
            self._im = im
        return im

    def _build_metadata(self) -> ImageServerMetadata:
        im = self._load_image()
        name = Path(self._path).name if self._path else 'Unnamed image'
        return ImageServerMetadata(
            path=self._path,
            name=name,
            shapes=ImageShape(x=im.shape[1], y=im.shape[0], c=1 if im.ndim == 2 else im.shape[2]),
            dtype=im.dtype,
            is_rgb=im.shape[-1] == 3 and im.dtype == np.uint8  # TODO: Better determine if RGB
        )

    def read_block(self, level: int, block: Union[Region2D, Tuple[int, ...]]) -> np.ndarray:
        assert level == 0
        from dataclasses import astuple
        _, x, y, width, height, z, t = astuple(_validate_block(block))
        im = self._load_image()
        if im.ndim == 2 or im.ndim == 2:
            assert z == 0
            assert t == 0
            return im[y:y + height, x:x + width]
        if im.ndim == 4:
            assert t == 0
            return im[y:y + height, x:x + width, :, z]
        return im[y:y + height, x:x + width, :, z, t]
