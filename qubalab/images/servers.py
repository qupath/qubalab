from abc import ABC, abstractmethod
from typing import Union, Tuple, Iterable
from dataclasses import dataclass
from PIL import Image, ImageCms

import io
import warnings
import numpy as np
from geojson import Polygon


@dataclass(frozen=True)
class Region2D:
    """
    Simple data class to represent the bounding box for a region in 2D.
    This optionally includes a downsample value to indicate that the 
    region may be required at a lower resolution.
    """
    downsample: float = None
    x: int = 0
    y: int = 0
    width: int = -1
    height: int = -1
    z: int = 0
    t: int = 0

    @classmethod
    def from_geometry(cls, geometry, downsample: float = None) -> "Region2D":
        from ..objects.geometries import to_geometry, to_shapely
        geometry = to_geometry(geometry)
        if geometry is None:
            return None
        bounds = to_shapely(geometry).bounds
        return Region2D(
            downsample=downsample,
            x=int(bounds[0]),
            y=int(bounds[1]),
            width=int(bounds[2] - bounds[0]),
            height=int(bounds[3] - bounds[1]),
            z=getattr(geometry, 'z', 0),
            t=getattr(geometry, 't', 0)
        )

    def scale_region(self, scale_factor: float = None) -> "Region2D":
        if scale_factor is None:
            return self.downsample_region()
        return self.downsample_region(1.0 / scale_factor)

    def downsample_region(self, downsample: float = None) -> "Region2D":
        """
        Downsample the bounding box for the Region2D.
        This can be used to convert coordinates from (for example) the full image resolution 
        to a different pyramidal level.
        """
        if downsample is None:
            downsample = self.downsample

        if downsample is None or downsample == 1:
            return self

        if downsample == 0:
            raise ValueError('Downsample cannot be 0!')

        x = int(self.x / downsample)
        y = int(self.y / downsample)
        # Handle -1 for width & height, i.e. until the full image width
        if self.width == -1:
            x2 = x - 1
        else:
            x2 = int(round(self.x + self.width) / downsample)
        if self.height == -1:
            y2 = y - 1
        else:
            y2 = int(round(self.y + self.height) / downsample)
        return Region2D(downsample=None, x=x, y=y, width=x2 - x, height=y2 - y, z=self.z, t=self.t)


    @property
    def geometry(self) -> Polygon:
        """
        Get a Geometry representation of this region.
        """
        from ..objects.geometries import create_rectangle
        return create_rectangle(self.x, self.y, self.width, self.height, z=self.z, t=self.t)



@dataclass(frozen=True)
class PixelLength:
    """
    Simple data class to store pixel size information, along one dimension.
    Can be thought of as the pixel width, pixel height or pixel depth (z-spacing).
    """
    length: float = 1.0
    unit: str = 'pixels'

    def is_default(self) -> bool:
        """
        Returns True if this is a default value (length is 1.0 and unit is 'pixels'), False otherwise.
        """
        return self.length == 1.0 and self.unit == 'pixels'

    @staticmethod
    def create_default() -> 'PixelLength':
        """
        Create a PixelLength with a default value of 1 pixel.
        """
        return PixelLength(length=1.0, unit='pixels')

    @staticmethod
    def create_microns(length: float) -> 'PixelLength':
        """
        Create a PixelLength with a unit of micrometers (µm).
        """
        return PixelLength(length=length, unit='micrometer')

    @staticmethod
    def create_unknown(length: float) -> 'PixelLength':
        """
        Create a PixelLength with an unknown unit.
        """
        return PixelLength(length=length, unit=None)


@dataclass(frozen=True)
class PixelCalibration:
    """
    Simple data class for storing pixel calibration information.
    Currently only the width, height and depth (z-spacing) are supported.
    """
    length_x: PixelLength = PixelLength()
    length_y: PixelLength = PixelLength()
    length_z: PixelLength = PixelLength()

    @property
    def is_calibrated(self) -> bool:
        for siz in [self.length_x, self.length_y, self.length_z]:
            if not siz.is_default():
                return True
        return False


@dataclass(frozen=True)
class ImageChannel:
    name: str
    color: Tuple[float, float, float]


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


@dataclass
class ImageShape:
    """
    Simple data class to store an image shape.
    Useful to avoid ambiguity about dimension order.

    TODO: This can potentially be removed if we use xarray instead of numpy arrays, and/or embrace AICSImageIO's
          dimension ordering standard.
    """
    x: int
    y: int
    t: int = 1
    c: int = 1
    z: int = 1

    def from_tczyx(*args) -> 'ImageShape':
        return ImageShape(t=args[0], c=args[1], z=args[2], y=args[3], x=args[4])

    def as_tuple(self, dims: str = 'tczyx'):
        return tuple(self.__getattribute__(d) for d in dims)


def _possible_downsample(x1: int, x2: int, downsample: float) -> bool:
    """
    Determine if an image dimension is what you'd expect after downsampling, and then flooring/rounding to decide
    the new dimension.
    """
    return int(int(x1 / downsample) * downsample) == x2 or int(round(int(round(x1 / downsample)) * downsample)) == x2


def _estimate_downsample(main_shape: ImageShape, secondary_shape: ImageShape) -> float:
    """
    Estimate the downsample factor between two ImageShapes.
    This is used to prefer values like 4 rather than 4.000345, which arise due to resolutions having to have
    integer pixel dimensions.
    """
    dx = main_shape.x / secondary_shape.x
    dy = main_shape.y / secondary_shape.y
    downsample = (dx + dy) / 2.0
    downsample_round = round(downsample)
    if _possible_downsample(main_shape.x, secondary_shape.x, downsample_round) and _possible_downsample(main_shape.y,
                                                                                                        secondary_shape.y,
                                                                                                        downsample_round):
        if downsample != downsample_round:
            warnings.warn(f'Returning rounded downsample value {downsample_round} instead of {downsample}')
        return downsample_round
    return downsample


@dataclass
class ImageServerMetadata:
    """
    Simple data class to store core metadata for a pyramidal image.
    """
    path: str
    name: str
    shapes: Tuple[ImageShape, ...]
    pixel_calibration: PixelCalibration
    is_rgb: bool
    dtype: np.dtype
    channels: Tuple[ImageChannel, ...] = None


class ImageServer(ABC):

    def __init__(self, resize_method: Image.Resampling = Image.Resampling.BICUBIC):
        super().__init__()
        self._metadata = None
        self._downsamples = None
        self._channels = None
        self._resize_method = resize_method

    def read_region(self,
                    region: Union[Region2D, Tuple[int, ...]] = None,
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
        """

        if region is None:
            region = Region2D(downsample=downsample,
                              x=x, y=y, width=width, height=height, z=z, t=t)
        elif isinstance(region, Tuple):
            # If we have a tuple, use it along with the downsample if available
            if downsample is None:
                region = Region2D(*region)
            else:
                region = Region2D((downsample,) + region)

        if not isinstance(region, Region2D):
            raise ValueError('No valid region provided to read_region method')

        # Fix negative values for width or height
        if region.width < 0 or region.height < 0:
            w = region.width if region.width >= 0 else self.width - region.x
            h = region.height if region.height >= 0 else self.height - region.y
            region = Region2D(downsample=region.downsample,
                              x=region.x, y=region.y, width=w, height=h, z=region.z, t=region.t)

        all_downsamples = self.downsamples
        level = _get_level(all_downsamples, region.downsample)
        level_downsample = all_downsamples[level]

        block = region.downsample_region(downsample=level_downsample)

        im = self.read_block(level=level, block=block)
        if downsample == level_downsample:
            return im
        target_size = (round(region.width / region.downsample), round(region.height / region.downsample))
        return _resize(im, target_size, resample=self.resize_method)

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

    @abstractmethod
    def read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        """
        Read a block of pixels from a specific level.

        Coordinates are provided in the coordinate space of the level, NOT the full-resolution image.
        This means that the returned image should have the width and height specified.

        Note that this is a lower-level method than :func:`read_region`; usually you should use that method instead.

        TODO: Consider if this should actually return a dask array or xarray
        TODO: Consider if this should return using the dimension ordering of AICSImageIO

        :param level: the pyramidal level to read from
        :param block: a tuple of integers (x, y, width, height, z, t) specifying the block to read
        :return: a numpy array containing the requested pixels from the 2D region, in the order [y, x, c]
        """
        pass

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


class WrappedImageServer(ImageServer):
    """
    Abstract class for an ImageServer that wraps another ImageServer, 
    e.g. to transform the image in some way.
    """

    def __init__(self, base_server: ImageServer, **kwargs):
        super().__init__(**kwargs)
        self._base_server = base_server

    @property
    def base_server(self) -> ImageServer:
        return self._base_server

    def _build_metadata(self) -> ImageServerMetadata:
        return self._base_server.metadata

    @property
    def path(self) -> str:
        return self._base_server.path + ' (wrapped)'


class IccProfileServer(WrappedImageServer):
    """
    Wrap an ImageServer and apply an ICC Profile to the pixels, if possible.

    If no ICC Profile is provided, an attempt is made to read the profile from the image using PIL.
    This isn't guaranteed to succeed.
    To find out if it was successful, test whether self.icc_transform is not None.

    See http://www.andrewjanowczyk.com/application-of-icc-profiles-to-digital-pathology-images/ 
    for a blog post describing where this may be useful, and providing further code.
    """

    def __init__(self, base_server: ImageServer,
                 icc_profile: Union[bytes, ImageCms.ImageCmsProfile, ImageCms.ImageCmsTransform] = None,
                 **kwargs):
        super().__init__(base_server, **kwargs)

        try:
            if isinstance(icc_profile, ImageCms.ImageCmsProfile) or isinstance(icc_profile, bytes):
                srgb = ImageCms.createProfile("sRGB")
                icc_profile = ImageCms.buildTransformFromOpenProfiles(icc_profile, srgb, "RGB", "RGB")

            if isinstance(icc_profile, ImageCms.ImageCmsTransform):
                self._icc = icc_profile
            else:
                self._icc = _read_icc(base_server.metadata.path)
        except:
            warnings.warn(f'No ICC Profile found for {base_server.path}')
            self._icc = None

    @property
    def icc_transform(self) -> ImageCms.ImageCmsTransform:
        """
        Get the transform used to apply the ICC profile.
        If this is None, then the server simply returns the original pixels unchanged.
        """
        return self._icc

    def read_block(self, *args, **kwargs) -> np.ndarray:
        im = self.base_server.read_block(*args, **kwargs)
        if self._icc:
            image = Image.fromarray(im)
            image = ImageCms.applyTransform(image, self._icc)
            im = np.array(image)
        return im

    @property
    def path(self) -> str:
        return self._base_server.path + ' (+ICC Profile)'


def _get_icc_bytes(path) -> bytes:
    # Temporarily remove max pixel limit used to avoid decompression bomb DOS attach error 
    # since there's a good chance we have a very large (whole slide) image
    max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        icc = Image.open(path).info.get('icc_profile', None)
        if None:
            warnings.warn(f'No ICC Profile found')
        return io.BytesIO(icc)
    except Exception as e:
        warnings.warn(f'Unable to read ICC Profile: {e}')
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = max_pixels


def _read_icc(path, **kwargs) -> ImageCms.ImageCmsTransform:
    icc_bytes = _get_icc_bytes(path)
    if icc_bytes is None:
        return None
    srgb = ImageCms.createProfile("sRGB")
    return ImageCms.buildTransformFromOpenProfiles(icc_bytes, srgb, "RGB", "RGB", **kwargs)


def _compute_length(length: float, downsample: float = None) -> int:
    """
    Helper function for computing an image list (width or height) with downsampling,
    using a consistent approach to rounding.
    :param length:
    :param downsample:
    :return:
    """
    return int(round(length / downsample))


def _validate_block(block: Union[Region2D, Tuple[int, ...]]) -> Region2D:
    """
    Convert a variable length tuple into a Region2D, corresponding to the 2D bounding box
    for an image region in the form (x, y, width, height, z, t).
    Note that the tuple should be in the order (x, y, width, height, z, t) and default values 
    will be used for any that are missing.

    The downsample value for the Region2D will be None.
    """
    if isinstance(block, Region2D):
        return block

    return Region2D(*((None,) + block))

    # if len(block) == 6:
    #     return block
    # if len(block) < 4:
    #     raise ValueError(f'Block required as least 4 entries (x, y, width, height): receieved {block}')
    # elif len(block) < 6:
    #     return block + ('',) * (6 - len(block))


def _get_size(im: Union[np.ndarray, Image.Image]):
    """
    Get the size of an image as a two-element tuple (width, height).
    """
    return im.size if isinstance(im, Image.Image) else im.shape[:2][::-1]


def _resize(im: Union[np.ndarray, Image.Image], target_size: Tuple[int, int], resample: int = Image.Resampling.BICUBIC):
    """
    Resize an image to a target size.

    This uses the implementation from PIL.

    :param im: input image; assumed channels-last with 2 or 3 dimensions
    :param target_size: target size in (width, height format)
    :param resample: reampling mode to use
    :return:
    """
    if _get_size(im) == target_size:
        return im

    # If we have a PIL image, just resize normally
    if isinstance(im, Image.Image):
        return im.resize(size=target_size, resample=resample)

    # If we have NumPy, do one channel at a time
    if im.ndim == 2:
        if im.dtype in [np.uint8, np.float32]:
            image = Image.fromarray(im)
        elif np.issubdtype(im.dtype, np.integer):
            image = Image.fromarray(im.astype(np.int32), mode='I')
        else:
            image = Image.fromarray(im.astype(np.float32), mode='F')
        image = _resize(image, target_size=target_size, resample=resample)
        return np.asarray(image).astype(im.dtype)
    else:
        im_channels = [
            _resize(im[:, :, c, ...], target_size=target_size, resample=resample)
            for c in range(im.shape[2])
        ]
        if len(im_channels) == 1:
            return np.atleast_3d(im_channels[0])
        return np.stack(im_channels, axis=-1)


def _get_level(all_downsamples: Tuple[float], downsample: float, abs_tol=1e-3) -> int:
    """
    Get the level (index) from a list of downsamples that is best for fulfilling an image region request.

    This is the index of the entry in all_downsamples that either (almost) matches the requested downsample,
    or relates to the next highest resolution image (so that any required scaling is to reduce resolution).

    :param all_downsamples:  all available downsamples
    :param downsample:       requested downsample value
    :param abs_tol:          absolute tolerance when comparing downsample values; this allows for a stored downsample
                             value to be slightly off due to rounding
                             (e.g. requesting 4.0 would match a level 4 +/- abs_tol)
    :return:
    """
    downsamples = all_downsamples
    if len(downsamples) == 1 or downsample <= downsamples[0]:
        return 0
    elif downsample >= downsamples[-1]:
        return len(downsamples) - 1
    else:
        # Allow a little bit of a tolerance because downsamples are often calculated
        # by rounding the ratio of image dimensions... and can end up a little bit off
        for level, d in reversed(list(enumerate(downsamples))):
            if downsample >= d - abs_tol:
                return level
        return 0
