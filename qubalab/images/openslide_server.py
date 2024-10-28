import numpy as np
import dask.array as da
from pathlib import Path
from dataclasses import astuple
import warnings
try:
    import openslide
except ImportError as e:
    warnings.warn(f'Unable to import OpenSlide, will try TiffSlide instead')
    import tiffslide as openslide
from .image_server import ImageServer
from .metadata.image_metadata import ImageMetadata
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .metadata.image_shape import ImageShape
from .region_2d import Region2D


class OpenSlideServer(ImageServer):
    """
    An image server that relies on OpenSlide (https://openslide.org/) to read RGB images.

    This server may only be able to detect the full resolution of a pyramidal image.

    OpenSlide provides some properties to define a rectangle bounding the non-empty region of the slide
    (see https://openslide.org/api/python/#standard-properties). If such properties are found, only this
    rectangle will be read (but note that this behaviour was not properly tested).
    """

    def __init__(self, path: str, strip_alpha=True, single_channel=False, limit_bounds=True, **kwargs):
        """
        :param path: the local path to the image to open
        :param strip_alpha: whether to strip the alpha channel from the image
        :param single_channel: whether to keep only the first channel of the image
        :param limit_bounds: whether to only consider a rectangle bounding the non-empty region of the slide,
                             if such rectangle is defined in the properties of the image
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(**kwargs)
        self._reader = openslide.OpenSlide(path)
        self._path = path
        self._strip_alpha = strip_alpha
        self._single_channel = single_channel
        self._limit_bounds = limit_bounds

    def close(self):
        self._reader.close()

    def _build_metadata(self) -> ImageMetadata:
        n_channels = OpenSlideServer._get_n_channels(self._single_channel, self._strip_alpha)

        full_bounds = (0, 0) + self._reader.dimensions
        bounds = (self._reader.properties.get(openslide.PROPERTY_NAME_BOUNDS_X),
                  self._reader.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y),
                  self._reader.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH),
                  self._reader.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT))
        if self._limit_bounds and not any(v is None for v in bounds):
            self._bounds = bounds
            w = self._bounds[3]
            h = self._bounds[2]
            shapes = tuple(ImageShape(x=int(w / d), y=int(h / d), c=n_channels) for d in self._reader.level_downsamples)
        else:
            self._bounds = full_bounds
            shapes = tuple(ImageShape(x=d[0], y=d[1], c=n_channels) for d in self._reader.level_dimensions)

        return ImageMetadata(
            self._path,
            Path(self._path).name,
            shapes,
            OpenSlideServer._get_pixel_calibration(
                self._reader.properties.get(openslide.PROPERTY_NAME_MPP_X),
                self._reader.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            ),
            True,
            np.uint8
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        x, y, width, height, z, t = astuple(region)
        assert z == 0, "OpenSlide can't read 3d images"
        assert t == 0, "OpenSlide can't read time-varying images"

        # x and y are provided in the coordinate space of the level, while openslide
        # expect coordinates in the level 0 reference frame
        x = int(x * self.metadata.downsamples[level])
        y = int(y * self.metadata.downsamples[level])

        # Map coordinates to bounding rectangle
        x += self._bounds[0]
        y += self._bounds[1]

        image = self._reader.read_region((x, y), level, (width, height))
        im = np.moveaxis(np.asarray(image), 2, 0)
        image.close()

        # Return image, stripping alpha/converting to single-channel if needed
        return im[:self.metadata.n_channels, :, :]

    @staticmethod
    def _get_n_channels(single_channel: bool, strip_alpha: bool):
        if single_channel:
            return 1
        elif strip_alpha:
            return 3
        else:
            return 4
        
    @staticmethod
    def _get_pixel_calibration(pixel_width: str, pixel_height: str):
        if pixel_width is not None and pixel_height is not None:
            return PixelCalibration(
                length_x=PixelLength.create_microns(float(pixel_width)),
                length_y=PixelLength.create_microns(float(pixel_height))
            )
        else:
            return PixelCalibration()
