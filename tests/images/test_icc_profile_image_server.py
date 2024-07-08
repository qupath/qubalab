import numpy as np
from PIL import ImageCms
from qubalab.images.image_server import ImageMetadata, ImageServer
from qubalab.images.icc_profile_server import IccProfileServer
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.region_2d import Region2D


sample_RGB_metadata = ImageMetadata(
    "/path/to/img.tiff",
    "Image name",
    (
        ImageShape(16, 12, c=3),
    ),
    PixelCalibration(
        PixelLength.create_microns(2.5),
        PixelLength.create_microns(2.5)
    ),
    True,
    np.uint8
)
sample_RGB_pixels = np.array(
    [[[x / sample_RGB_metadata.width * 255, y / sample_RGB_metadata.height * 255, 0] for x in range(sample_RGB_metadata.width)] for y in range(sample_RGB_metadata.height)],
    np.uint8
)
sample_RGB_pixels = np.array([[[
    x / sample_RGB_metadata.width * 255 if c == 0 else (y / sample_RGB_metadata.height * 255 if c == 1 else 0)
    for x in range(sample_RGB_metadata.width)]
    for y in range(sample_RGB_metadata.height)]
    for c in range(sample_RGB_metadata.n_channels)
], np.uint8)
class SampleRGBServer(ImageServer):
    def close(self):
        pass

    def _build_metadata(self) -> ImageMetadata:
        return sample_RGB_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        return sample_RGB_pixels[:, region.y:region.y+region.height, region.x:region.x+region.width]


def test_transform_when_icc_profile_not_provided():
    sample_rgb_server = SampleRGBServer()
    icc_profile_server = IccProfileServer(sample_rgb_server)

    transform = icc_profile_server.icc_transform

    assert transform == None
    
    icc_profile_server.close()


def test_region_of_pixels_when_icc_profile_not_provided():
    sample_rgb_server = SampleRGBServer()
    icc_profile_server = IccProfileServer(sample_rgb_server)

    image = icc_profile_server.read_region(1, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(image, sample_RGB_pixels)
    
    icc_profile_server.close()


def test_transform_when_icc_profile_provided():
    sample_rgb_server = SampleRGBServer()
    icc_profile_server = IccProfileServer(sample_rgb_server, ImageCms.createProfile("sRGB"))

    transform = icc_profile_server.icc_transform

    assert transform != None
    
    icc_profile_server.close()


def test_region_of_pixels_when_icc_profile_provided():
    sample_rgb_server = SampleRGBServer()
    icc_profile_server = IccProfileServer(sample_rgb_server, ImageCms.createProfile("sRGB"))

    image = icc_profile_server.read_region(1, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(image, sample_RGB_pixels)
    
    icc_profile_server.close()