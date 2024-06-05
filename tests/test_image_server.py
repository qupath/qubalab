import numpy as np
from qubalab.images.image_server import ImageServerMetadata, ImageServer
from qubalab.images.image_shape import ImageShape
from qubalab.images.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.region_2d import Region2D


sample_metadata = ImageServerMetadata(
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
sample_pixels = [[[[x / shape.x * 255, y / shape.y * 255, 0] for x in range(shape.x)] for y in range(shape.y)] for shape in sample_metadata.shapes]


def test_metadata():
    expected_metadata = sample_metadata
    sample_RGB_server = SampleRGBServer()

    metadata = sample_RGB_server.metadata

    assert expected_metadata == metadata


def test_full_resolution_image_with_region():
    expected_image = np.array(sample_pixels[0], dtype=np.uint8)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, Region2D(x=0, y=0, width=sample_metadata.width, height=sample_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_image_with_region_tuple():
    expected_image = np.array(sample_pixels[0], dtype=np.uint8)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, (0, 0, sample_metadata.width, sample_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_image_with_tuple():
    expected_image = np.array(sample_pixels[0], dtype=np.uint8)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, x=0, y=0, width=sample_metadata.width, height=sample_metadata.height)

    np.testing.assert_array_equal(image, expected_image)


def test_lowest_resolution_image():
    expected_image = np.array(sample_pixels[-1], dtype=np.uint8)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(4, Region2D(x=0, y=0, width=sample_metadata.width, height=sample_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_tile_of_full_resolution_image():
    xFrom = int(sample_metadata.width/4)
    xTo = int(3*sample_metadata.width/4)
    yFrom = int(sample_metadata.height/4)
    yTo = int(3*sample_metadata.height/4)
    expected_image = np.array(sample_pixels[0], dtype=np.uint8)[yFrom:yTo+1, xFrom:xTo+1, :]
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, Region2D(x=xFrom, y=yFrom, width=xTo-xFrom+1, height=yTo-yFrom+1))

    np.testing.assert_array_equal(image, expected_image)


def test_image_with_new_downsample():
    width = 48
    height = 37
    expected_image = np.array([[[x / width * 255, y / height * 255, 0] for x in range(width)] for y in range(height)], dtype=np.uint8)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1.5, Region2D(x=0, y=0, width=sample_metadata.width, height=sample_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


class SampleRGBServer(ImageServer):
    def _build_metadata(self) -> ImageServerMetadata:
        return sample_metadata

    def read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_pixels[level], dtype=np.uint8)
        return image[region.y:region.y+region.height, region.x:region.x+region.width, :]
    
    def close():
        pass