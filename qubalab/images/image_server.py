from dataclasses import dataclass
from typing import Union
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from image_channel import ImageChannel
from image_shape import ImageShape
from pixel_calibration import PixelCalibration
from region_2d import Region2D


@dataclass
class ImageServerMetadata:
    """
    Simple data class to store core metadata for a pyramidal image.

    :param path: the local path to the image
    :param name: the image name
    :param shapes: the image shape, for each resolution of the image
    :param pixel_calibration: the pixel calibration information of the image
    :param is_rgb: whether the image is RGB
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

    """
    Create the ImageServer.

    :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
    """
    def __init__(self, resize_method: Image.Resampling = Image.Resampling.BICUBIC):
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

    @property
    def resize_method(self) -> Image.Resampling:
        """
        Resampling method to use when resizing the image for downsampling.
        Subclasses can override this, e.g. to enforce nearest neighbor resampling for label images.
        """
        return self._resize_method

    @abstractmethod
    def _build_metadata(self) -> ImageServerMetadata:
        ...

    def rebuild_metadata(self):
        """
        Request that the metadata is rebuilt.

        This shouldn't normally be required, but may be useful in some cases where the 
        metadata set at initialization has been updated (i.e. this ImageServer wraps 
        around some mutable instance).
        """
        self._metadata = self._build_metadata()

    @property
    def channels(self) -> Tuple[ImageChannel, ...]:
        # Get from metadata as most recent channels
        channels = self.metadata.channels
        if channels:
            return channels
        # Cache default channels if we don't have them already
        if not self._channels:
            if self.n_channels == 1:
                channels = _DEFAULT_CHANNEL_SINGLE
            elif self.n_channels == 2:
                channels = _DEFAULT_CHANNEL_TWO
            elif self.is_rgb:
                channels = _DEFAULT_CHANNEL_RGB
            else:
                channels = [ImageChannel(f'Channel {ii + 1}',
                                         _DEFAULT_CHANNEL_COLORS[ii % len(_DEFAULT_CHANNEL_COLORS)]) for ii in
                            range(self.n_channels)]
            self._channels = channels
        return self._channels

    @property
    def is_rgb(self) -> bool:
        return self.metadata.is_rgb

    def _dim_length(self, dim: int):
        shape = self.metadata.shape
        return shape[dim] if len(shape) > dim else 1

    @property
    def metadata(self) -> ImageServerMetadata:
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    @property
    def shape(self) -> ImageShape:
        return self.metadata.shapes[0]

    @property
    def dtype(self) -> np.dtype:
        return self.metadata.dtype

    @property
    def width(self) -> int:
        return self.shape.x

    @property
    def height(self) -> int:
        return self.shape.y

    @property
    def n_channels(self) -> int:
        return self.shape.c

    @property
    def n_resolutions(self) -> int:
        return len(self.metadata.shapes)

    @property
    def n_timepoints(self) -> int:
        return self.shape.t

    @property
    def n_z_slices(self) -> int:
        return self.shape.z

    @property
    def downsamples(self) -> Tuple[float, ...]:
        if not self._downsamples:
            main_shape = self.metadata.shapes[0]
            self._downsamples = tuple(_estimate_downsample(main_shape, s) for s in self.metadata.shapes)
        return self._downsamples

    @property
    def path(self) -> str:
        return self.metadata.path

    @property
    def name(self) -> str:
        return self.metadata.name

    def close(self):
        pass

    def level_to_dask(self, level: int = 0):
        """
        Convert a single pyramid level to a dask array.

        :param level: the pyramid level (0 is full resolution)
        """
        pass

    def to_dask(self, downsample: Union[float, Iterable[float]] = None):
        """
        Convert this image to one or more dask arrays, at any arbitary downsample factor.

        TODO: Consider API that creates downsamples from requested pixel sizes, since these are more intuitive.

        :param: downsample the downsample factor to use, or a list of downsample factors to use.
                If None, all available resolutions will be used.
        :return: a dask array or tuple of dask arrays, depending upon whether one or more downsample factors are required
        """

        if downsample is None:
            if self.n_resolutions == 1:
                return self.level_to_dask(level=0)
            else:
                return tuple([self.level_to_dask(level=level) for level in range(self.n_resolutions)])

        if isinstance(downsample, Iterable):
            return tuple([self.to_dask(downsample=float(d)) for d in downsample])

        level = _get_level(self.downsamples, downsample)
        array = self.level_to_dask(level=level)

        # Check if we need to resize the array
        # Don't rely on rescale to be exactly equal to 1.0, and instead check if we need a different output size
        rescale = downsample / self.downsamples[level]
        input_width = array.shape[3]
        input_height = array.shape[2]
        output_width = int(round(input_width / rescale))
        output_height = int(round(input_height / rescale))
        if input_width == output_width and input_height == output_height:
            return array

        # Couldn't find an easy resizing method for dask arrays... so we try this instead

        # TODO: Urgently need something better! Performance is terrible for large images - all pixels requested
        #       upon first compute (even for a small region), and then resized. This is not scalable.

        if array.size > 10000:
            print('Warning - calling affine_transform on a large dask array can be *very* slow')

        from dask_image.ndinterp import affine_transform
        from dask import array as da
        array = da.asarray(array)
        transform = np.eye(array.ndim)
        transform[2, 2] = rescale
        transform[3, 3] = rescale
        # Not sure why rechunking is necessary here, but it is
        array = da.rechunk(array)
        array = affine_transform(array,
                                 transform,
                                 order=1,
                                 output_chunks=array.chunks)
        return array[:, :, :output_height, :output_width, ...]
    

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
        
    def _get_size(self, image: Union[np.ndarray, Image.Image]):
        """
        Get the size of an image as a two-element tuple (width, height).
        """
        return image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]