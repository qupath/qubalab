from ..images import ImageServer, PixelLength, PixelCalibration, ImageServerMetadata, ImageShape
from ..images.servers import _validate_block

from dataclasses import astuple
from typing import Tuple
from py4j.java_gateway import JavaGateway, JavaObject

from imageio.v3 import imread
from imageio import volread
import numpy as np

from urllib.parse import urlparse, unquote
from pathlib import Path


class QuPathServer(ImageServer):

    def __init__(self, gateway: JavaGateway, server_obj: JavaObject, **kwargs):
        super().__init__(**kwargs)
        self._gateway = gateway
        self._server_obj = server_obj

    def _build_metadata(self) -> ImageServerMetadata:
        # Get what we can directly from the server
        server = self._server_obj
        downsamples = tuple([d for d in server.getPreferredDownsamples()])
        
        self._downsamples = downsamples # Use the QuPath values directly
        
        shapes = tuple([ImageShape(x=level.getWidth(), y=level.getHeight(), c=server.nChannels(), z=server.nZSlices(), t=server.nTimepoints())
            for level in server.getMetadata().getLevels()])

        dt = np.dtype(server.getPixelType().toString().lower())
        is_rgb = server.isRGB()

        meta = server.getMetadata()
        name = meta.getName()
        cal = server.getPixelCalibration()

        # Try to get the file path
        path = _find_server_file_path(server)
        if path is None:
            path = server.getPath()
        
        if cal.hasPixelSizeMicrons():
            if cal.getZSpacingMicrons():
                length_z = PixelLength.create_microns(cal.getZSpacingMicrons())
            else:
                length_z = PixelLength()
            pixel_cal = PixelCalibration(
                length_x = PixelLength.create_microns(cal.getPixelWidthMicrons()),
                length_y = PixelLength.create_microns(cal.getPixelHeightMicrons()),
                length_z = length_z
                )
        else:
            pixel_cal = PixelCalibration()

        return ImageServerMetadata(
            path=path,
            name=name,
            pixel_calibration=pixel_cal,
            shapes=shapes,
            dtype=dt,
            is_rgb=is_rgb
        )

    def read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        _, x, y, width, height, z, t = astuple(_validate_block(block))

        gateway = self._gateway
        server = self._server_obj

        # TODO: Explore requesting directly in QuPath - this is awkward and could result in 
        # rounding problems
        if level < 0:
            level = len(self.downsamples) + level
        downsample = server.getDownsampleForResolution(level)

        byte_array = gateway.entry_point.getTiff(server,
            downsample,
            int(round(x * downsample)),
            int(round(y * downsample)),
            int(round(width * downsample)),
            int(round(height * downsample)),
            z,
            t)

        if self.metadata.is_rgb or self.n_channels == 1:
            return imread(byte_array)

        # We can just provide 2D images; using volread move to channels-last
        return np.moveaxis(volread(byte_array), 0, -1)


def _get_server_uris(server: JavaObject) -> Tuple[str]:
    """
    Get URIs for a java object representing an ImageServer.
    """
    return tuple(str(u) for u in server.getURIs())

def _find_server_file_path(server: JavaObject) -> str:
    """
    Try to get the file path for a java object representing an ImageServer.
    This can be useful
    """
    uris = _get_server_uris(server)
    if len(uris) == 1:
        parsed = urlparse(uris[0])
        if parsed.scheme == 'file':
            return unquote(parsed.path)
    return None