from dataclasses import dataclass
from typing import Union
import numpy as np
import warnings
from abc import ABC, abstractmethod
from PIL import Image
from .image_channel import ImageChannel
from .image_shape import ImageShape
from .pixel_calibration import PixelCalibration
from .region_2d import Region2D


@dataclass
class ImageServerMetadata:
    """
    Simple data class to store core metadata for a pyramidal image.

    :param path: the local path to the image
    :param name: the image name
    :param shapes: the image shape, for each resolution of the image
    :param pixel_calibration: the pixel calibration information of the image
    :param is_rgb: whether pixels of the image are stored with the RGB format
    :param dtype: the type of the pixel values
    :param channels: the channels of the image
    """
    path: str
    name: str
    shapes: tuple[ImageShape, ...]
    pixel_calibration: PixelCalibration
    is_rgb: bool
    dtype: np.dtype
    channels: tuple[ImageChannel, ...] = None


class ImageServer(ABC):
    """
    An abtract class to read pixels and metadata of an image.
    """

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

    def __init__(self, resize_method: Image.Resampling = Image.Resampling.BICUBIC):
        """
        Create the ImageServer.

        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__()
        self._metadata = None
        self._downsamples = None
        self._channels = None
        self._resize_method = resize_method

    @abstractmethod
    def read_block(self, level: int, block: tuple[int, ...]) -> np.ndarray:
        """
        Read a block of pixels from a specific level.

        Coordinates are provided in the coordinate space of the level, NOT the full-resolution image.
        This means that the returned image should have the width and height specified.

        Note that this is a lower-level method than :func:`read_region`; usually you should use that method instead.

        TODO: Consider if this should actually return a dask array or xarray
        TODO: Consider if this should return using the dimension ordering of AICSImageIO
        TODO: block is tuple, not region2d?

        :param level: the pyramidal level to read from
        :param block: a tuple of integers (x, y, width, height, z, t) specifying the block to read
        :return: a numpy array containing the requested pixels from the 2D region, in the order [y, x, c]
        """
        pass

    @abstractmethod
    def _build_metadata(self) -> ImageServerMetadata:
        """
        Create metadata for the current image.

        :return: the metadata of the image
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close this image server.
        
        This should be called whenever this server is not used anymore.
        """
        pass

    @property
    def resize_method(self) -> Image.Resampling:
        """
        Resampling method to use when resizing the image for downsampling.

        Subclasses can override this, e.g. to enforce nearest neighbor resampling for label images.
        """
        return self._resize_method

    @property
    def channels(self) -> tuple[ImageChannel, ...]:
        """
        The channels of the image.
        """
        # Get from metadata as most recent channels
        channels = self.metadata.channels
        if channels:
            return channels
        # Cache default channels if we don't have them already
        if not self._channels:
            if self.n_channels == 1:
                channels = self._DEFAULT_CHANNEL_SINGLE
            elif self.n_channels == 2:
                channels = self._DEFAULT_CHANNEL_TWO
            elif self.is_rgb:
                channels = self._DEFAULT_CHANNEL_RGB
            else:
                channels = [
                    ImageChannel(
                        f'Channel {ii + 1}',
                        self._DEFAULT_CHANNEL_COLORS[ii % len(self._DEFAULT_CHANNEL_COLORS)]
                    ) for ii in range(self.n_channels)
                ]
            self._channels = channels
        return self._channels

    @property
    def is_rgb(self) -> bool:
        """
        Whether pixels are stored with the RGB format.
        """
        return self.metadata.is_rgb

    @property
    def metadata(self) -> ImageServerMetadata:
        """
        The image metadata.
        """
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    @property
    def shape(self) -> ImageShape:
        """
        The dimensions of the full-resolution image.
        """
        return self.metadata.shapes[0]

    @property
    def dtype(self) -> np.dtype:
        """
        The type of the pixel values.
        """
        return self.metadata.dtype

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
    def n_resolutions(self) -> int:
        """
        The number of resolutions of the image.
        """
        return len(self.metadata.shapes)

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
    def downsamples(self) -> tuple[float, ...]:
        """
        The downsamples of the image.
        """
        if not self._downsamples:
            main_shape = self.metadata.shapes[0]
            self._downsamples = tuple(self._estimate_downsample(main_shape, s) for s in self.metadata.shapes)
        return self._downsamples

    @property
    def path(self) -> str:
        """
        The local path to the image.
        """
        return self.metadata.path

    @property
    def name(self) -> str:
        """
        The name of the image.
        """
        return self.metadata.name

    def read_region(self,
                    region: Union[Region2D, tuple[int, ...]] = None,
                    downsample: float = None,
                    x: int = 0,
                    y: int = 0,
                    width: int = -1,
                    height: int = -1,
                    z: int = 0,
                    t: int = 0
                    ) -> np.ndarray:
        """
        Read pixels from any arbitrary image region, at any resolution determined by the downsample.

        This method can be called in one of two ways: passing a region (as a Region2D object or a tuple of integers),
        or passing x, y, width, height, z and t parameters separately. The latter can be more convenient and readable
        when calling interactively, without the need to create a region object.
        If a region is passed, the other parameters are ignored.

        Important: coordinates and width/height values are given in the coordinate space of the full-resolution image,
        and the downsample is applied before reading the region.

        This means that, except when the downsample == 1.0, the width and height of the returned image will usually
        be different from the width and height passed as parameters.
        This may result in off-by-one issues due to user-expectation and rounding; these can be avoided by using
        :func:`read_block` if the downsample corresponds exactly to an existing level.

        TODO: Consider if this should actually return a dask array or xarray
        TODO: Consider if this should return using the dimension ordering of AICSImageIO

        :param region: a Region2D object or a tuple of integers (x, y, width, height, z, t)
        :param downsample: the downsample to use (ignored if a Region2D is provided)
        :param x: the x coordinate of the region to read
        :param y: the y coordinate of the region to read
        :param width: the width of the region to read
        :param height: the height of the region to read
        :param z: the z index of the region to read
        :param t: the t index of the region to read
        :return: a numpy array containing the requested pixels from the 2D region, in the order [y, x, c]
        :raises ValueError: when the region to read is not specified
        """

        if region is None:
            region = Region2D(downsample=downsample, x=x, y=y, width=width, height=height, z=z, t=t)
        elif isinstance(region, tuple):
            # If we have a tuple, use it along with the downsample if available
            if downsample is None:
                region = Region2D(*region)
            else:
                region = Region2D((downsample,) + region)
        else:
            raise ValueError('No valid region provided to read_region method')

        # Fix negative values for width or height
        if region.width < 0 or region.height < 0:
            w = region.width if region.width >= 0 else self.width - region.x
            h = region.height if region.height >= 0 else self.height - region.y
            region = Region2D(downsample=region.downsample, x=region.x, y=region.y, width=w, height=h, z=region.z, t=region.t)

        level = self._get_level(region.downsample)
        level_downsample = self.downsamples[level]

        block = region.downsample_region(downsample=level_downsample)
        image = self.read_block(level=level, block=block)
        
        if downsample == level_downsample:
            return image
        else:
            target_size = (round(region.width / region.downsample), round(region.height / region.downsample))
            return self._resize(image, target_size, self.resize_method)

    def rebuild_metadata(self):
        """
        Request that the metadata is rebuilt.

        This shouldn't normally be required, but may be useful in some cases where the 
        metadata set at initialization has been updated (i.e. this ImageServer wraps 
        around some mutable instance).
        """
        self._metadata = self._build_metadata()
    

    def _get_level(self, downsample: float, abs_tol=1e-3) -> int:
        """
        Get the level (index) from the image downsamples that is best for fulfilling an image region request.

        This is the index of the entry in self.downsamples that either (almost) matches the requested downsample,
        or relates to the next highest resolution image (so that any required scaling is to reduce resolution).

        :param downsample: the requested downsample value
        :param abs_tol: absolute tolerance when comparing downsample values; this allows for a stored downsample
                        value to be slightly off due to rounding
                        (e.g. requesting 4.0 would match a level 4 +/- abs_tol)
        :return: the level that is best for fulfilling an image region request at the specified downsample
        """
        if len(self.downsamples) == 1 or downsample <= self.downsamples[0]:
            return 0
        elif downsample >= self.downsamples[-1]:
            return len(self.downsamples) - 1
        else:
            # Allow a little bit of a tolerance because downsamples are often calculated
            # by rounding the ratio of image dimensions... and can end up a little bit off
            for level, d in reversed(list(enumerate(self.downsamples))):
                if downsample >= d - abs_tol:
                    return level
            return 0
        
    def _resize(self, image: Union[np.ndarray, Image.Image], target_size: tuple[int, int], resample: int = Image.Resampling.BICUBIC):
        """
        Resize an image to a target size.

        This uses the implementation from PIL.

        :param image: input image; assumed channels-last with 2 or 3 dimensions
        :param target_size: target size in (width, height) format
        :param resample: resampling mode to use, by default bicubic
        :return: the resized image
        """
        if self._get_size(image) == target_size:
            return image

        # If we have a PIL image, just resize normally
        if isinstance(image, Image.Image):
            return image.resize(size=target_size, resample=resample)
        # If we have NumPy, do one channel at a time
        else:
            if image.ndim == 2:
                if image.dtype in [np.uint8, np.float32]:
                    image = Image.fromarray(image)
                elif np.issubdtype(image.dtype, np.integer):
                    image = Image.fromarray(image.astype(np.int32), mode='I')
                else:
                    image = Image.fromarray(image.astype(np.float32), mode='F')
                image = self._resize(image, target_size=target_size, resample=resample)
                return np.asarray(image).astype(image.dtype)
            else:
                image_channels = [
                    self._resize(image[:, :, c, ...], target_size=target_size, resample=resample)
                    for c in range(image.shape[2])
                ]
                if len(image_channels) == 1:
                    return np.atleast_3d(image_channels[0])
                else:
                    return np.stack(image_channels, axis=-1)
    
    def _estimate_downsample(self, main_shape: ImageShape, secondary_shape: ImageShape) -> float:
        """
        Estimate the downsample factor between two ImageShapes.
        
        This is used to prefer values like 4 rather than 4.000345, which arise due to resolutions having to have
        integer pixel dimensions.
        """
        dx = main_shape.x / secondary_shape.x
        dy = main_shape.y / secondary_shape.y
        downsample = (dx + dy) / 2.0
        downsample_round = round(downsample)

        if (
            self._possible_downsample(main_shape.x, secondary_shape.x, downsample_round) and
            self._possible_downsample(main_shape.y, secondary_shape.y, downsample_round)
        ):
            if downsample != downsample_round:
                warnings.warn(f'Returning rounded downsample value {downsample_round} instead of {downsample}')
            return downsample_round
        return downsample
        
    def _get_size(self, image: Union[np.ndarray, Image.Image]):
        """
        Get the size of an image as a two-element tuple (width, height).
        """
        return image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]

    def _possible_downsample(self, x1: int, x2: int, downsample: float) -> bool:
        """
        Determine if an image dimension is what you'd expect after downsampling, and then flooring/rounding to decide
        the new dimension.
        """
        return int(int(x1 / downsample) * downsample) == x2 or int(round(int(round(x1 / downsample)) * downsample)) == x2