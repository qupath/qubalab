import logging
import math
import numpy as np
from .image_channel import ImageChannel
from .image_shape import ImageShape
from .pixel_calibration import PixelCalibration


class ImageMetadata:
    """
    Simple class to store core metadata for a pyramidal image.
    """

    def __init__(
        self,
        path: str,
        name: str,
        shapes: tuple[ImageShape, ...],
        pixel_calibration: PixelCalibration,
        is_rgb: bool,
        dtype: np.dtype,
        channels: tuple[ImageChannel, ...] = None,
        downsamples = None
    ):
        """
        :param path: the local path to the image
        :param name: the image name
        :param shapes: the image shape, for each resolution of the image
        :param pixel_calibration: the pixel calibration information of the image
        :param is_rgb: whether pixels of the image are stored with the RGB format
        :param dtype: the type of the pixel values
        :param _channels: the channels of the image (optional)
        :param _downsamples: the downsamples of the image (optional)
        """
        self.path = path
        self.name = name
        self.shapes = shapes
        self.pixel_calibration = pixel_calibration
        self.is_rgb = is_rgb
        self.dtype = dtype
        self._channels = channels
        self._downsamples = downsamples

    _DEFAULT_CHANNEL_SINGLE = (
        ImageChannel(name='Single channel', color=(1, 1, 1)),
    )
    _DEFAULT_CHANNEL_RGB = (
        ImageChannel(name='Red', color=(1, 0, 0)),
        ImageChannel(name='Green', color=(0, 1, 0)),
        ImageChannel(name='Green', color=(0, 0, 1)),
    )
    _DEFAULT_CHANNEL_TWO = (
        ImageChannel(name='Channel 1', color=(1, 0, 1)),
        ImageChannel(name='Channel 2', color=(0, 1, 0))
    )
    _DEFAULT_CHANNEL_COLORS = (
        (0, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    )

    @property
    def shape(self) -> ImageShape:
        """
        The dimensions of the full-resolution image.
        """
        return self.shapes[0]
    
    @property
    def width(self) -> int:
        """
        The width of the full-resolution image.
        """
        return self.shape.x

    @property
    def height(self) -> int:
        """
        The height of the full-resolution image.
        """
        return self.shape.y

    @property
    def n_channels(self) -> int:
        """
        The number of channels of the image.
        """
        return self.shape.c

    @property
    def n_timepoints(self) -> int:
        """
        The number of time points of the image.
        """
        return self.shape.t

    @property
    def n_z_slices(self) -> int:
        """
        The number of z-slices of the image.
        """
        return self.shape.z

    @property
    def n_resolutions(self) -> int:
        """
        The number of resolutions of the image.
        """
        return len(self.shapes)

    @property
    def downsamples(self) -> tuple[float, ...]:
        """
        The downsamples of the image.
        """
        if self._downsamples is None:
            self._downsamples = tuple(self._estimate_downsample(self.shape, s) for s in self.shapes)
        return self._downsamples

    @property
    def channels(self) -> tuple[ImageChannel, ...]:
        """
        The channels of the image.
        """
        if self._channels is None:
            if self.is_rgb:
                self._channels = self._DEFAULT_CHANNEL_RGB
            else:
                if self.n_channels == 1:
                    self._channels = self._DEFAULT_CHANNEL_SINGLE
                elif self.n_channels == 2:
                    self._channels = self._DEFAULT_CHANNEL_TWO
                else:
                    self._channels = [
                        ImageChannel(
                            f'Channel {ii + 1}',
                            self._DEFAULT_CHANNEL_COLORS[ii % len(self._DEFAULT_CHANNEL_COLORS)]
                        ) for ii in range(self.n_channels)
                    ]
        return self._channels
    
    def __eq__(self, other):
        if isinstance(other, ImageMetadata):
            return self.path == other.path and \
                self.path == other.path and self.name == other.name and self.shapes == other.shapes \
                and self.pixel_calibration == other.pixel_calibration and self.is_rgb == other.is_rgb \
                and self.dtype == other.dtype and self.channels == other.channels and self.downsamples == other.downsamples
        else:
            return False
    
    def _estimate_downsample(self, higher_resolution_shape: ImageShape, lower_resolution_shape: ImageShape) -> float:
        """
        Estimate the downsample factor between a higher resolution and a lower resolution ImageShape.
        
        This is used to prefer values like 4 rather than 4.000345, which arise due to resolutions having to have
        integer pixel dimensions.
        The downsample is computed between the width and the heights of the shapes.
        :param higher_resolution_shape: a higher resolution shape
        :param lower_resolution_shape: a lower resolution shape
        :return: the downsample between the higher and the lower resolution shape
        """

        dx = higher_resolution_shape.x / lower_resolution_shape.x
        dy = higher_resolution_shape.y / lower_resolution_shape.y
        downsample = (dx + dy) / 2.0
        downsample_round = round(downsample)

        if (
            self._possible_downsample(higher_resolution_shape.x, lower_resolution_shape.x, downsample_round) and
            self._possible_downsample(higher_resolution_shape.y, lower_resolution_shape.y, downsample_round)
        ):
            if downsample != downsample_round:
                logging.debug(f'Returning rounded downsample value {downsample_round} instead of {downsample}')
            return downsample_round
        else:
            return downsample

    def _possible_downsample(self, higher_resolution_value: int, lower_resolution_value: int, downsample: float) -> bool:
        """
        Determine if an image dimension is what you'd expect after downsampling and applying floor or ceil.

        :param higher_resolution_value: a higher resolution value to downsample
        :param lower_resolution_value: a lower resolution value to compare to the downsampled higher resolution value
        :param downsample: the downsample to apply to the higher resolution value
        :return: whether the downsampled higher resolution value corresponds to the lower resolution value
        """
        return (math.floor(higher_resolution_value / downsample) == lower_resolution_value or
                math.ceil(higher_resolution_value / downsample) == lower_resolution_value)
