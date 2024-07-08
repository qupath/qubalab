import numpy as np
from PIL import Image
from qubalab.images.image_server import ImageMetadata, ImageServer
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.region_2d import Region2D


sample_RGB_metadata = ImageMetadata(
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
sample_RGB_pixels = [[[[
    x / shape.x * 255 if c == 0 else (y / shape.y * 255 if c == 1 else 0)
    for x in range(shape.x)]
    for y in range(shape.y)]
    for c in range(shape.c)]
    for shape in sample_RGB_metadata.shapes
]
class SampleRGBServer(ImageServer):
    def close(self):
        pass

    def _build_metadata(self) -> ImageMetadata:
        return sample_RGB_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
        return image[:, region.y:region.y+region.height, region.x:region.x+region.width]


sample_float32_metadata = ImageMetadata(
    "/path/to/img.tiff",
    "Image name",
    (
        ImageShape(64, 50, t=5, z=2, c=3),
        ImageShape(32, 25, t=5, z=2, c=3),
        ImageShape(16, 12, t=5, z=2, c=3)
    ),
    PixelCalibration(
        PixelLength.create_microns(2.5),
        PixelLength.create_microns(2.5)
    ),
    False,
    np.float32
)
sample_float32_pixels = [[[[[[
    x/shape.x + y/shape.y + z/shape.z + c/shape.c + t/shape.t
    for x in range(shape.x)]
    for y in range(shape.y)]
    for z in range(shape.z)]
    for c in range(shape.c)]
    for t in range(shape.t)]
    for shape in sample_float32_metadata.shapes
]
class SampleFloat32Server(ImageServer):
    def close(self):
        pass

    def _build_metadata(self) -> ImageMetadata:
        return sample_float32_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
        return image[region.t, :, region.z, region.y:region.y+region.height, region.x:region.x+region.width]


def test_metadata():
    expected_metadata = sample_RGB_metadata
    sample_RGB_server = SampleRGBServer()

    metadata = sample_RGB_server.metadata

    assert expected_metadata == metadata
    
    sample_RGB_server.close()


def test_full_resolution_RGB_image_with_region():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_full_resolution_RGB_image_with_region_tuple():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(
        sample_RGB_metadata.downsamples[level],
        (0, 0, sample_RGB_metadata.width, sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_full_resolution_RGB_image_with_tuple():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(
        sample_RGB_metadata.downsamples[level],
        x=0,
        y=0,
        width=sample_RGB_metadata.width,
        height=sample_RGB_metadata.height
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_full_resolution_RGB_image_with_dask():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_lowest_resolution_RGB_image():
    level = sample_RGB_metadata.n_resolutions - 1
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_lowest_resolution_RGB_image_with_dask():
    level = sample_RGB_metadata.n_resolutions - 1
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_tile_of_full_resolution_RGB_image():
    level = 0
    xFrom = int(sample_RGB_metadata.width/4)
    xTo = int(3*sample_RGB_metadata.width/4)
    yFrom = int(sample_RGB_metadata.height/4)
    yTo = int(3*sample_RGB_metadata.height/4)
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)[:, yFrom:yTo+1, xFrom:xTo+1]
    sample_RGB_server = SampleRGBServer()

    image = sample_RGB_server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(x=xFrom, y=yFrom, width=xTo-xFrom+1, height=yTo-yFrom+1)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_RGB_server.close()


def test_downsampled_RGB_image():
    downsample = 1.5
    c = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_RGB_pixels[0], sample_RGB_metadata.dtype)[c, ...]) \
        .resize((round(sample_RGB_metadata.width / downsample), round(sample_RGB_metadata.height / downsample)))
    sample_RGB_server = SampleRGBServer()

    pixels = sample_RGB_server.read_region(downsample, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(pixels[c, ...], expected_pixels)
    
    sample_RGB_server.close()


def test_scaled_RGB_image():
    downsample = 0.5
    c = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_RGB_pixels[0], sample_RGB_metadata.dtype)[c, ...]) \
        .resize((round(sample_RGB_metadata.width / downsample), round(sample_RGB_metadata.height / downsample)))
    sample_RGB_server = SampleRGBServer()

    pixels = sample_RGB_server.read_region(downsample, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(pixels[c, ...], expected_pixels)
    
    sample_RGB_server.close()


def test_downsampled_RGB_image_with_dask():
    downsample = 1.5
    c = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_RGB_pixels[0], sample_RGB_metadata.dtype)[c, ...]) \
        .resize((round(sample_RGB_metadata.width / downsample), round(sample_RGB_metadata.height / downsample)))
    sample_RGB_server = SampleRGBServer()

    pixels = sample_RGB_server.read_region(downsample, Region2D(x=0, y=0, width=sample_RGB_metadata.width, height=sample_RGB_metadata.height))

    np.testing.assert_array_equal(pixels[c, ...], expected_pixels)
    
    sample_RGB_server.close()


def test_downsampled_RGB_image_with_dask():
    downsample = 1.5
    c = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_RGB_pixels[0], sample_RGB_metadata.dtype)[c, ...]) \
        .resize((round(sample_RGB_metadata.width / downsample), round(sample_RGB_metadata.height / downsample)))
    sample_RGB_server = SampleRGBServer()

    pixels = sample_RGB_server.to_dask(downsample).compute()

    np.testing.assert_allclose(pixels[c, ...], expected_pixels, atol=2)
    
    sample_RGB_server.close()


def test_full_resolution_float32_image_with_region():
    level = 0
    z = int(sample_float32_metadata.shapes[level].z / 2)
    t = int(sample_float32_metadata.shapes[level].t / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_full_resolution_float32_image_with_region_tuple():
    level = 0
    z = int(sample_float32_metadata.shapes[level].z / 2)
    t = int(sample_float32_metadata.shapes[level].t / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(
        sample_float32_metadata.downsamples[level],
        (0, 0, sample_float32_metadata.width, sample_float32_metadata.height, z, t)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_full_resolution_float32_image_with_tuple():
    level = 0
    z = int(sample_float32_metadata.shapes[level].z / 2)
    t = int(sample_float32_metadata.shapes[level].t / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(
        sample_float32_metadata.downsamples[level],
        x=0,
        y=0,
        width=sample_float32_metadata.width,
        height=sample_float32_metadata.height,
        z=z,
        t=t
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_full_resolution_float32_image_with_dask():
    level = 0
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_lowest_resolution_float32_image():
    level = sample_float32_metadata.n_resolutions - 1
    z = int(sample_float32_metadata.shapes[level].z / 2)
    t = int(sample_float32_metadata.shapes[level].t / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(x=0, y=0, width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_lowest_resolution_float32_image_with_dask():
    level = sample_float32_metadata.n_resolutions - 1
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.level_to_dask(level).compute()

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_tile_of_full_resolution_float32_image():
    level = 0
    z = int(sample_float32_metadata.shapes[level].z / 2)
    t = int(sample_float32_metadata.shapes[level].t / 2)
    xFrom = int(sample_float32_metadata.width/4)
    xTo = int(3*sample_float32_metadata.width/4)
    yFrom = int(sample_float32_metadata.height/4)
    yTo = int(3*sample_float32_metadata.height/4)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, yFrom:yTo+1, xFrom:xTo+1]
    sample_float32_server = SampleFloat32Server()

    image = sample_float32_server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(x=xFrom, y=yFrom, width=xTo-xFrom+1, height=yTo-yFrom+1, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)
    
    sample_float32_server.close()


def test_downsampled_float32_image():
    downsample = 1.5
    z = 0
    c = 0
    t = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_float32_pixels[0], sample_float32_metadata.dtype)[t, c, z, ...]) \
        .resize((round(sample_float32_metadata.width / downsample), round(sample_float32_metadata.height / downsample)))
    sample_float32_server = SampleFloat32Server()

    pixels = sample_float32_server.read_region(
        downsample,
        Region2D(0, 0, sample_float32_metadata.width, sample_float32_metadata.height, z, t)
    )

    np.testing.assert_array_equal(pixels[c, ...], expected_pixels)
    
    sample_float32_server.close()


def test_scaled_float32_image():
    downsample = 0.5
    z = 0
    c = 0
    t = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_float32_pixels[0], sample_float32_metadata.dtype)[t, c, z, ...]) \
        .resize((round(sample_float32_metadata.width / downsample), round(sample_float32_metadata.height / downsample)))
    sample_float32_server = SampleFloat32Server()

    pixels = sample_float32_server.read_region(
        downsample,
        Region2D(0, 0, sample_float32_metadata.width, sample_float32_metadata.height, z, t)
    )

    np.testing.assert_array_equal(pixels[c, ...], expected_pixels)
    
    sample_float32_server.close()


def test_downsampled_float32_image_with_dask():
    downsample = 1.5
    z = 0
    c = 0
    t = 0
    expected_pixels = Image \
        .fromarray(np.array(sample_float32_pixels[0], sample_float32_metadata.dtype)[t, c, z, ...]) \
        .resize((round(sample_float32_metadata.width / downsample), round(sample_float32_metadata.height / downsample)))
    sample_float32_server = SampleFloat32Server()

    pixels = sample_float32_server.to_dask(downsample).compute()

    np.testing.assert_allclose(pixels[t, c, z, ...], expected_pixels, atol=2)
    
    sample_float32_server.close()
