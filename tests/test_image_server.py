import numpy as np
from qubalab.images.image_server import ImageServerMetadata, ImageServer
from qubalab.images.image_shape import ImageShape
from qubalab.images.pixel_calibration import PixelCalibration, PixelLength


class SampleRGBServer(ImageServer):
    def _build_metadata(self) -> ImageServerMetadata:
        return ImageServerMetadata(
            "/path/to/img.tiff",
            "Image name",
            tuple(
                ImageShape(128, 100),
                ImageShape(64, 50),
                ImageShape(32, 25)
            ),
            PixelCalibration(
                PixelLength.create_microns(2.5),
                PixelLength.create_microns(2.5)
            ),
            True,
            np.uint8
        )

    def read_block(self, level: int, block: tuple[int, ...]) -> np.ndarray:
        return None
    
    def close():
        pass