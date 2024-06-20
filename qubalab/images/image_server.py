import numpy as np
import dask.array as da
from typing import Union
from abc import ABC, abstractmethod
from PIL import Image
from .metadata.region_2d import Region2D
from .metadata.image_server_metadata import ImageServerMetadata
from .pyramid_store import PyramidStore


class ImageServer(ABC):
    """
    An abtract class to read pixels and metadata of an image.

    An image server must be closed (see the close() function) once no longer used.
    """

    def __init__(self, resize_method: Image.Resampling = Image.Resampling.BICUBIC):
        """
        Create the ImageServer.

        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__()
        self._metadata = None
        self._resize_method = resize_method
        self._pyramid_store = None

    @property
    def metadata(self) -> ImageServerMetadata:
        """
        The image metadata.
        """
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    def read_region(self,
                    downsample: float,
                    region: Union[Region2D, tuple[int, ...]] = None,
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
        If a region is passed, the other parameters (except for the downsample) are ignored.

        Important: coordinates and width/height values are given in the coordinate space of the full-resolution image,
        and the downsample is applied before reading the region.

        This means that, except when the downsample is 1.0, the width and height of the returned image will usually
        be different from the width and height passed as parameters.

        TODO: Consider if this should actually return a dask array or xarray
        TODO: Consider if this should return using the dimension ordering of AICSImageIO

        :param downsample: the downsample to use
        :param region: a Region2D object or a tuple of integers (x, y, width, height, z, t)
        :param x: the x coordinate of the region to read
        :param y: the y coordinate of the region to read
        :param width: the width of the region to read
        :param height: the height of the region to read
        :param z: the z index of the region to read
        :param t: the t index of the region to read
        :return: a 3-dimensional numpy array containing the requested pixels from the 2D region.
                 The [y, x, c] index of the returned array returns the channel of index c of the
                 pixel located at [x, y] on the image
        :raises ValueError: when the region to read is not specified
        """

        if region is None:
            region = Region2D(x=x, y=y, width=width, height=height, z=z, t=t)
        elif isinstance(region, tuple):
            # If we have a tuple, use it along with the downsample if available
            region = Region2D(*region)
        if not isinstance(region, Region2D):
            raise ValueError('No valid region provided to read_region method')

        # Fix negative values for width or height
        if region.width < 0 or region.height < 0:
            w = region.width if region.width >= 0 else self.metadata.width - region.x
            h = region.height if region.height >= 0 else self.metadata.height - region.y
            region = Region2D(x=region.x, y=region.y, width=w, height=h, z=region.z, t=region.t)

        level = ImageServer._get_level(self.metadata.downsamples, downsample)
        level_downsample = self.metadata.downsamples[level]
        image = self._read_block(level, region.downsample_region(downsample=level_downsample))
        
        if downsample == level_downsample:
            return image
        else:
            target_size = (round(region.width / downsample), round(region.height / downsample))
            return self._resize(image, target_size, self._resize_method)

    def level_to_dask(self, level: int = 0) -> da.Array:
        """
        Return a dask array representing a single resolution of the image.

        Pixels of the returned array can be accessed with the following order:
        (t, c, z, y, x). There may be less dimensions for simple images: for
        example, an image with a single timepoint and a single z-slice will
        return an array of dimensions (c, y, x).

        Subclasses of ImageServer may override this function if they can provide
        a simpler and faster implementation.

        :param level: the pyramid level (0 is full resolution). Must be less than the number
                      of resolutions of the image
        :returns: a dask array containing all pixels of the provided level
        :raises ValueError: when level is not valid
        """

        if level < 0 or level >= self.metadata.n_resolutions:
            raise ValueError("The provided level ({0}) is outside the valid range ([0, {1}])".format(level, self.metadata.n_resolutions - 1))
        
        if self._pyramid_store is None:
            self._pyramid_store = PyramidStore(self.metadata, self._read_block)

        return da.from_zarr(self._pyramid_store, component=self._pyramid_store.get_path_of_level(level))

    def rebuild_metadata(self):
        """
        Request that the metadata is rebuilt.

        This shouldn't normally be required, but may be useful in some cases where the 
        metadata set at initialization has been updated (i.e. this ImageServer wraps 
        around some mutable instance).
        """
        self._metadata = self._build_metadata()

    @abstractmethod
    def close(self):
        """
        Close this image server.
        
        This should be called whenever this server is not used anymore.
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
    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        """
        Read a block of pixels from a specific level.

        Coordinates are provided in the coordinate space of the level, NOT the full-resolution image.
        This means that the returned image should have the width and height specified.

        Note that this is a lower-level method than :func:`read_region`; usually you should use that method instead.

        TODO: Consider if this should actually return a dask array or xarray
        TODO: Consider if this should return using the dimension ordering of AICSImageIO
        TODO: remove in favour of level_to_dask?
        
        :param level: the pyramidal level to read from
        :param region: the region to read
        :return: a 3-dimensional numpy array containing the requested pixels from the 2D region.
                 The [y, x, c] index of the returned array returns the channel of index c of the
                 pixel located at [x, y] on the image
        """
        pass
    
    @staticmethod
    def _get_level(all_downsamples: tuple[float], downsample: float, abs_tol=1e-3) -> int:
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
        if len(all_downsamples) == 1 or downsample <= all_downsamples[0]:
            return 0
        elif downsample >= all_downsamples[-1]:
            return len(all_downsamples) - 1
        else:
            # Allow a little bit of a tolerance because downsamples are often calculated
            # by rounding the ratio of image dimensions... and can end up a little bit off
            for level, d in reversed(list(enumerate(all_downsamples))):
                if downsample >= d - abs_tol:
                    return level
            return 0
        
    @staticmethod
    def _resize(image: Union[np.ndarray, Image.Image], target_size: tuple[int, int], resample: int = Image.Resampling.BICUBIC) -> Union[np.ndarray, Image.Image]:
        """
        Resize an image to a target size.

        This uses the implementation from PIL.

        :param image: the image to resize. Either a 3-dimensional numpy array with dimensions (y, x, channel)
                      or a PIL image
        :param target_size: target size in (width, height) format
        :param resample: resampling mode to use, by default bicubic
        :return: the resized image, either a 3-dimensional numpy array with dimensions (y, x, channel) or a PIL image
        """
        if ImageServer._get_size(image) == target_size:
            return image

        # If we have a PIL image, just resize normally
        if isinstance(image, Image.Image):
            return image.resize(size=target_size, resample=resample)
        # If we have NumPy, do one channel at a time
        else:
            if image.ndim == 2:
                if image.dtype in [np.uint8, np.float32]:
                    pilImage = Image.fromarray(image)
                elif np.issubdtype(image.dtype, np.integer):
                    pilImage = Image.fromarray(image.astype(np.int32), mode='I')
                else:
                    pilImage = Image.fromarray(image.astype(np.float32), mode='F')
                pilImage = ImageServer._resize(pilImage, target_size=target_size, resample=resample)
                return np.asarray(pilImage).astype(image.dtype)
            else:
                image_channels = [
                    ImageServer._resize(image[:, :, c, ...], target_size=target_size, resample=resample) for c in range(image.shape[2])
                ]
                if len(image_channels) == 1:
                    return np.atleast_3d(image_channels[0])
                else:
                    return np.stack(image_channels, axis=-1)
        
    @staticmethod
    def _get_size(image: Union[np.ndarray, Image.Image]):
        """
        Get the size of an image as a two-element tuple (width, height).
        """
        return image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]