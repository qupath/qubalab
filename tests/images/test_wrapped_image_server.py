import numpy as np
import dask.array as da
from qubalab.images.image_server import ImageServerMetadata, ImageServer
from qubalab.images.wrapped_image_server import WrappedImageServer
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.region_2d import Region2D


class SampleServer(ImageServer):
    isClosed = False
    
    def close(self):
        self.isClosed = True

    def _build_metadata(self) -> ImageServerMetadata:
        return ImageServerMetadata(
            "/path/to/img.tiff",
            "Image name",
            (
                ImageShape(64, 50),
                ImageShape(32, 25),
                ImageShape(16, 12)
            ),
            PixelCalibration(
                PixelLength.create_microns(2.5),
                PixelLength.create_microns(2.5)
            ),
            True,
            np.uint8
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        return np.array()


class SampleWrappedServer(WrappedImageServer):
    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        return np.array()


def test_wrapped_server():
    expected_sample_server = SampleServer()
    wrapped_server = SampleWrappedServer(expected_sample_server)

    sample_server = wrapped_server.base_server

    assert sample_server == expected_sample_server


def test_sample_server_closed():
    sample_server = SampleServer()
    wrapped_server = SampleWrappedServer(sample_server)

    wrapped_server.close()

    assert sample_server.isClosed


def test_metadata():
    sample_server = SampleServer()
    wrapped_server = SampleWrappedServer(sample_server)
    expected_metadata = sample_server.metadata

    metadata = wrapped_server.metadata

    assert expected_metadata == metadata
