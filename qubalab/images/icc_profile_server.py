import numpy as np
import warnings
import io
import tifffile
from typing import Union
from PIL import Image, ImageCms
from .wrapped_image_server import WrappedImageServer
from .image_server import ImageServer
from .region_2d import Region2D


class IccProfileServer(WrappedImageServer):
    """
    Wrap an ImageServer and apply an ICC Profile to the pixels, if possible.

    If no ICC Profile is provided, an attempt is made to read the profile from the image using PIL.
    This isn't guaranteed to succeed.
    To find out if it was successful, test whether self.icc_transform is not None.

    See http://www.andrewjanowczyk.com/application-of-icc-profiles-to-digital-pathology-images/ 
    for a blog post describing where this may be useful, and providing further code.
    
    Closing this server will close the wrapped server.
    The metadata of this server is equivalent to the metadata of the wrapped server.
    """

    def __init__(
            self,
            base_server: ImageServer,
            icc_profile: Union[bytes, ImageCms.ImageCmsProfile, ImageCms.core.CmsProfile, ImageCms.ImageCmsTransform] = None,
            **kwargs
    ):
        """
        :param base_server: the server to wrap
        :param icc_profile: the ICC profile to apply to the wrapped image server. If omitted, an attempt is made to read the profile from the image.
                            If not successful, a warning will be logged.
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(base_server, **kwargs)

        if icc_profile is None:
            icc_profile = IccProfileServer._get_icc_bytes_from_path(base_server.metadata.path)

        if icc_profile is None:
            self._icc = None
        elif isinstance(icc_profile, ImageCms.ImageCmsTransform):
            self._icc = icc_profile
        else:
            self._icc = ImageCms.buildTransformFromOpenProfiles(icc_profile, ImageCms.createProfile("sRGB"), "RGB", "RGB")

        if self._icc is None:
            warnings.warn(f'No ICC Profile found for {base_server.metadata.path}. Returning original pixels.')

    @property
    def icc_transform(self) -> ImageCms.ImageCmsTransform:
        """
        Get the transform used to apply the ICC profile.

        If this is None, then the server simply returns the original pixels unchanged.
        """
        return self._icc

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = self.base_server._read_block(level, region)

        if self._icc:
            return np.transpose(np.array(ImageCms.applyTransform(Image.fromarray(np.transpose(image, axes=[2, 1, 0])), self._icc)), axes=[2, 1, 0])
        else:
            return image
    
    @staticmethod
    def _get_icc_bytes_from_path(path) -> bytes:
        try:
            with tifffile.TiffFile(path) as tif:
                if 34675 in tif.pages[0].tags:
                    return io.BytesIO(tif.pages[0].tags[34675].value)
                else:
                    return None
        except Exception as error:
            warnings.warn(f"Error while retrieving icc profile from {path}: {error}")
            return None
