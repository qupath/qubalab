import numpy as np
import tempfile
import os
from enum import Enum
from py4j.java_gateway import JavaGateway, JavaObject
from urllib.parse import urlparse, unquote
from .image_server import ImageServer
from .metadata.image_metadata import ImageMetadata
from .metadata.image_shape import ImageShape
from .metadata.image_channel import ImageChannel
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .region_2d import Region2D
from .utils import base64_to_image, bytes_to_image
from ..qupath import qupath_gateway


class QuPathServer(ImageServer):
    """
    An ImageServer that communicates with a QuPath ImageServer through a Gateway.

    Closing this server won't close the underlying QuPath ImageServer.
    """

    def __init__(
        self,
        gateway: JavaGateway = None,
        qupath_server: JavaObject = None,
        pixel_access: str = 'base_64',
        **kwargs
    ):
        """
        :param gateway: the gateway between Python and QuPath. If not specified, the default gateway is used
        :param qupath_server: a Java object representing the QuPath ImageServer. If not specified, the currently
                              opened in QuPath ImageServer is used
        :param pixel_access: how to send pixel values from QuPath to Python. Can be 'base_64' to convert pixels to
                             base 64 encoded text, 'bytes' to to convert pixels to bytes, or 'temp_files' to use
                             temporary files
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        :raises ValueError: when pixel_access has an unexpected value
        """
        super().__init__(**kwargs)

        available_pixel_access = ['base_64', 'bytes', 'temp_files']
        if pixel_access not in available_pixel_access:
            raise ValueError(f'The provided pixel access ({pixel_access}) is not one of {str(available_pixel_access)}')

        self._gateway = qupath_gateway.get_default_gateway() if gateway is None else gateway
        self._qupath_server = qupath_gateway.get_current_image_data(gateway).getServer() if qupath_server is None else qupath_server
        self._pixel_access = pixel_access

    def close(self):
        pass

    def _build_metadata(self) -> ImageMetadata:
        qupath_metadata = self._qupath_server.getMetadata()

        return ImageMetadata(
            path=QuPathServer._find_qupath_server_path(self._qupath_server),
            name=qupath_metadata.getName(),
            shapes=tuple([
                ImageShape(
                    x=level.getWidth(),
                    y=level.getHeight(),
                    c=self._qupath_server.nChannels(),
                    z=self._qupath_server.nZSlices(),
                    t=self._qupath_server.nTimepoints()
                )
                for level in qupath_metadata.getLevels()
            ]),
            pixel_calibration=QuPathServer._find_qupath_server_pixel_calibration(self._qupath_server),
            is_rgb=self._qupath_server.isRGB(),
            dtype=np.dtype(self._qupath_server.getPixelType().toString().lower()),
            channels=tuple([ImageChannel(c.getName(), QuPathServer._unpack_color(c.getColor())) for c in qupath_metadata.getChannels()]),
            downsamples=tuple([d for d in self._qupath_server.getPreferredDownsamples()])
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        if level < 0:
            level = len(self.metadata.downsamples) + level
        downsample = self._qupath_server.getDownsampleForResolution(level)

        request = self._gateway.jvm.qupath.lib.regions.RegionRequest.createInstance(
            self._qupath_server.getPath(),
            downsample,
            int(round(region.x * downsample)),
            int(round(region.y * downsample)),
            int(round(region.width * downsample)),
            int(round(region.height * downsample)),
            region.z,
            region.t
        )

        if self._pixel_access == 'temp_files':
            temp_path = tempfile.mkstemp(prefix='qubalab-', suffix='.tif')[1]

            self._gateway.entry_point.writeImageRegion(self._qupath_server, request, temp_path)
            image = bytes_to_image(temp_path, self.metadata.is_rgb, ImageShape(region.width, region.height, c=self.metadata.n_channels))
            ## on Windows, this fails because the file handle is open elsewhere
            ## slightly bad manners to pollute tempfiles but should be insignificant
            if not os.name == "nt":
                os.remove(temp_path)
        else:
            format = 'png' if self.metadata.is_rgb else "imagej tiff"

            if self._pixel_access == 'bytes':
                image = bytes_to_image(
                    self._gateway.entry_point.getImageBytes(self._qupath_server, request, format),
                    self.metadata.is_rgb,
                    ImageShape(region.width, region.height, c=self.metadata.n_channels)
                )
            else:
                image = base64_to_image(
                    self._gateway.entry_point.getImageBase64(self._qupath_server, request, format),
                    self.metadata.is_rgb,
                    ImageShape(region.width, region.height, c=self.metadata.n_channels)
                )

        return image
    
    @staticmethod
    def _find_qupath_server_path(qupath_server: JavaObject) -> str:
        """
        Try to get the file path for a java object representing an ImageServer.
        This can be useful to get direct access to an image file, rather than via QuPath.
        """

        uris = tuple(str(u) for u in qupath_server.getURIs())
        if len(uris) == 1:
            parsed = urlparse(uris[0])
            if parsed.scheme == 'file':
                return unquote(parsed.path)
        return qupath_server.getPath()
    
    @staticmethod
    def _find_qupath_server_pixel_calibration(qupath_server: JavaObject) -> PixelCalibration:
        pixel_calibration = qupath_server.getPixelCalibration()

        if pixel_calibration.hasPixelSizeMicrons():
            return PixelCalibration(
                PixelLength.create_microns(pixel_calibration.getPixelWidthMicrons()),
                PixelLength.create_microns(pixel_calibration.getPixelHeightMicrons()),
                PixelLength.create_microns(pixel_calibration.getZSpacingMicrons()) if pixel_calibration.hasZSpacingMicrons() else PixelLength()
            )
        else:
            return PixelCalibration()
    
    @staticmethod
    def _unpack_color(rgb: int) -> tuple[float, float, float]:
        r = (rgb >> 16) & 255
        g = (rgb >> 8) & 255
        b = rgb & 255
        return r / 255.0, g / 255.0, b / 255.0
