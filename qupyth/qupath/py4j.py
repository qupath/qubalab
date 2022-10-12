from ..images import ImageServer, PixelLength, PixelCalibration, ImageServerMetadata, ImageShape
from ..images.servers import _validate_block

from dataclasses import astuple
from typing import Tuple
from py4j.java_gateway import JavaGateway, JavaObject

from imageio.v3 import imread
from imageio import volread
import numpy as np


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

        # TODO: Replace with URIs!
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