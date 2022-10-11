from ..images import ImageServer, PixelCalibration, ImageServerMetadata, Units
from ..images.servers import _validate_block

from dataclasses import astuple
from typing import Tuple
from py4j.java_gateway import JavaGateway, JavaObject

from imageio.v3 import imread
from imageio import volread
import numpy as np


class QuPathServer(ImageServer):

    def __init__(self, gateway: JavaGateway, server_obj: JavaObject):
        super().__init__()
        self._gateway = gateway
        self._server_obj = server_obj

    def _build_metadata(self) -> ImageServerMetadata:
        # Get what we can directly from the server
        server = self._server_obj
        downsamples = tuple([d for d in server.getPreferredDownsamples()])
        shape = [server.getHeight(), server.getWidth(), server.nChannels(), server.nZSlices(), server.nTimepoints()]
        dt = np.dtype(server.getPixelType().toString().lower())
        is_rgb = server.isRGB()

        meta = server.getMetadata()
        name = meta.getName()
        cal = server.getPixelCalibration()

        # TODO: Replace with URIs!
        path = server.getPath()

        if cal.hasPixelSizeMicrons():
            pixel_cal = PixelCalibration(
                pixel_width=cal.getPixelWidthMicrons(),
                pixel_height=cal.getPixelHeightMicrons(),
                units=Units.MICRONS)
        else:
            pixel_cal = PixelCalibration()

        return ImageServerMetadata(
            path=path,
            name=name,
            downsamples=downsamples,
            pixel_calibration=pixel_cal,
            shape=shape,
            dtype=dt,
            is_rgb=is_rgb
        )

    def read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        _, x, y, width, height, z, t = astuple(_validate_block(block))

        gateway = self._gateway
        server = self._server_obj

        if level < 0:
            level = len(self.downsamples) + level
        downsample = server.getDownsampleForResolution(level)
        request = gateway.jvm.qupath.lib.regions.RegionRequest.createInstance(
            server.getPath(),
            downsample,
            int(round(x * downsample)),
            int(round(y * downsample)),
            int(round(width * downsample)),
            int(round(height * downsample)),
            z,
            t)

        byte_array = gateway.entry_point.getTiff(server, request)
        if self.metadata.is_rgb or self.n_channels == 1:
            return imread(byte_array)

        # We can just provide 2D images; using volread move to channels-last
        return np.moveaxis(volread(byte_array), 0, -1)