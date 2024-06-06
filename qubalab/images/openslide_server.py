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
from image_server import ImageServer
from .metadata.image_server_metadata import ImageServerMetadata
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .metadata.image_shape import ImageShape


class OpenSlideServer(ImageServer):
    """
    An image server that relies on OpenSlide (https://openslide.org/) to read images.

    It can only be used to read RGB images.
    """

    def __init__(self, path: str, strip_alpha=True, single_channel=False, **kwargs):
        """
        Create the server.

        :param path: the local path to the image to open
        :param strip_alpha: whether to strip the alpha channel from the image
        :param single_channel: whether to keep only the first channel of the image
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(**kwargs)
        self._openslide = openslide.OpenSlide(path)
        self._path = path
        self._strip_alpha = strip_alpha
        self._single_channel = single_channel

    def _build_metadata(self) -> ImageServerMetadata:
        if self._single_channel:
            n_channels = 1
        elif self._strip_alpha:
            n_channels = 3
        else:
            n_channels = 4

        pixel_width = self._openslide.properties.get('openslide.mpp-x')
        pixel_height = self._openslide.properties.get('openslide.mpp-y')
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
            shapes=tuple(ImageShape(x=d[0], y=d[1], c=n_channels) for d in self._openslide.level_dimensions),
            pixel_calibration=calibration,
            is_rgb=True,
            dtype=np.uint8
        )

    def _read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        _, x, y, width, height, z, t = astuple(_validate_block(block))
        assert z == 0
        assert t == 0
        # Request pixels
        level_downsample = self.downsamples[level]
        location = (int(x * level_downsample), int(y * level_downsample))
        image = self._openslide.read_region(location, level, (width, height))
        im = np.asarray(image)
        image.close()
        # Return image, stripping alpha/converting to single-channel if needed
        return im[:, :, :self.n_channels]

    def close(self):
        self._openslide.close()
