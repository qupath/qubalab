import numpy as np
import dask.array as da
from qubalab.images.image_server import ImageServerMetadata, ImageServer
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.metadata.region_2d import Region2D


sample_RGB_metadata = ImageServerMetadata(
    "/path/to/img.tiff",
    "Image name",
    (
        ImageShape(64, 50, c=3),
        ImageShape(32, 25, c=3),
        ImageShape(16, 12, c=3)
    ),
    PixelCalibration(
        PixelLength.create_microns(2.5),
        PixelLength.create_microns(2.5)
    ),
    True,
    np.uint8
)
sample_RGB_pixels = [[[[x / shape.x * 255, y / shape.y * 255, 0] for x in range(shape.x)] for y in range(shape.y)] for shape in sample_RGB_metadata.shapes]

class SampleRGBServer(ImageServer):
    def level_to_dask(self, level: int = 0) -> da.Array:
        return da.from_array(np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype))

    def close(self):
        pass

    def _build_metadata(self) -> ImageServerMetadata:
        return sample_RGB_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
        return image[region.y:region.y+region.height, region.x:region.x+region.width, :]


sample_float32_metadata = ImageServerMetadata(
    "/path/to/img.tiff",
    "Image name",
    (
        ImageShape(64, 50, c=3),
        ImageShape(32, 25, c=3),
        ImageShape(16, 12, c=3)
    ),
    PixelCalibration(
        PixelLength.create_microns(2.5),
        PixelLength.create_microns(2.5)
    ),
    False,
    np.float32
)
sample_float32_pixels = [[[[x / shape.x, y / shape.y, 0] for x in range(shape.x)] for y in range(shape.y)] for shape in sample_RGB_metadata.shapes]

class SampleFloat32Server(ImageServer):
    def level_to_dask(self, level: int = 0) -> da.Array:
        return da.from_array(np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype))
    
    def close():
        pass

    def _build_metadata(self) -> ImageServerMetadata:
        return sample_float32_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
        return image[region.y:region.y+region.height, region.x:region.x+region.width, :]


def test_metadata():
    expected_metadata = sample_RGB_metadata
    sample_RGB_server = SampleRGBServer()

    metadata = sample_RGB_server.metadata

    assert expected_metadata == metadata


def test_full_resolution_RGB_image_with_region():
    expected_image = np.array(sample_RGB_pixels[0], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_RGB_image_with_region_tuple():
    expected_image = np.array(sample_RGB_pixels[0], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, (0, 0, sample_RGB_metadata.width, sample_RGB_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_RGB_image_with_tuple():
    expected_image = np.array(sample_RGB_pixels[0], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_RGB_image_with_dask():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)


def test_lowest_resolution_RGB_image():
    expected_image = np.array(sample_RGB_pixels[-1], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(4, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_lowest_resolution_RGB_image_with_dask():
    level = sample_RGB_metadata.n_resolutions - 1
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)


def test_tile_of_full_resolution_RGB_image():
    xFrom = int(sample_RGB_metadata.width/4)
    xTo = int(3*sample_RGB_metadata.width/4)
    yFrom = int(sample_RGB_metadata.height/4)
    yTo = int(3*sample_RGB_metadata.height/4)
    expected_image = np.array(sample_RGB_pixels[0], dtype=sample_RGB_metadata.dtype)[yFrom:yTo+1, xFrom:xTo+1, :]
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(1, Region2D(x=xFrom, y=yFrom, width=xTo-xFrom+1, height=yTo-yFrom+1))

    np.testing.assert_array_equal(image, expected_image)


def test_dowsampled_RGB_image_size():
    downsample = 1.5
    expected_width = round(sample_RGB_metadata.width / downsample)
    expected_height = round(sample_RGB_metadata.height / downsample)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(downsample, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    assert expected_width == image.shape[1] and expected_height == image.shape[0]


def test_scaled_RGB_image_size():
    downsample = 0.5
    expected_width = round(sample_RGB_metadata.width / downsample)
    expected_height = round(sample_RGB_metadata.height / downsample)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(downsample, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    assert expected_width == image.shape[1] and expected_height == image.shape[0]


def test_full_resolution_float32_image_with_region():
    expected_image = np.array(sample_float32_pixels[0], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(1, Region2D(x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_float32_image_with_region_tuple():
    expected_image = np.array(sample_float32_pixels[0], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(1, (0, 0, sample_float32_metadata.width, sample_float32_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_float32_image_with_tuple():
    expected_image = np.array(sample_float32_pixels[0], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(1, x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height)

    np.testing.assert_array_equal(image, expected_image)


def test_full_resolution_float32_image_with_dask():
    level = 0
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)


def test_lowest_resolution_float32_image():
    expected_image = np.array(sample_float32_pixels[-1], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(4, Region2D(x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_lowest_resolution_float32_image_with_dask():
    level = sample_float32_metadata.n_resolutions - 1
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)


def test_tile_of_full_resolution_float32_image():
    xFrom = int(sample_float32_metadata.width/4)
    xTo = int(3*sample_float32_metadata.width/4)
    yFrom = int(sample_float32_metadata.height/4)
    yTo = int(3*sample_float32_metadata.height/4)
    expected_image = np.array(sample_float32_pixels[0], dtype=sample_float32_metadata.dtype)[yFrom:yTo+1, xFrom:xTo+1, :]
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(1, Region2D(x=xFrom, y=yFrom, width=xTo-xFrom+1, height=yTo-yFrom+1))

    np.testing.assert_array_equal(image, expected_image)


def test_dowsampled_float32_image_size():
    downsample = 1.5
    expected_width = round(sample_float32_metadata.width / downsample)
    expected_height = round(sample_float32_metadata.height / downsample)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(downsample, Region2D(x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height))

    assert expected_width == image.shape[1] and expected_height == image.shape[0]


def test_scaled_float32_image_size():
    downsample = 0.5
    expected_width = round(sample_float32_metadata.width / downsample)
    expected_height = round(sample_float32_metadata.height / downsample)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(downsample, Region2D(x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height))

    assert expected_width == image.shape[1] and expected_height == image.shape[0]
