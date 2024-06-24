import numpy as np
import zarr
import pytest
from qubalab.images.pyramid_store import PyramidStore, OME_NGFF_SPECIFICATION_VERSION
from qubalab.images.image_server import ImageServerMetadata, ImageServer
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.metadata.region_2d import Region2D


sample_RGB_metadata = ImageServerMetadata(
    "/path/to/img.tiff",
    "Image name",
    (
        ImageShape(64, 32, c=3),
        ImageShape(32, 16, c=3),
        ImageShape(16, 8, c=3)
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
    def _build_metadata(self) -> ImageServerMetadata:
        return sample_RGB_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
        return image[region.y:region.y+region.height, region.x:region.x+region.width, :]
    
    def close(self):
        pass


sample_float32_metadata = ImageServerMetadata(
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
    def close():
        pass

    def _build_metadata(self) -> ImageServerMetadata:
        return sample_float32_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
        return image[region.t, :, region.z, region.y:region.y+region.height, region.x:region.x+region.width]


def test_group_length_with_server_downsamples():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    length = len(root)

    assert length == sample_RGB_metadata.n_resolutions


def test_group_length_with_custom_downsamples():
    downsamples = [1, 4, 8, 16]
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, downsamples=downsamples)
    root = zarr.Group(store=pyramid_store)

    length = len(root)

    assert length == len(downsamples)


def test_group_read_only():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    with pytest.raises(RuntimeError):
        root.create_group("group")


def test_attributes_multiscales_version():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    version = root.attrs["multiscales"][0]["version"]

    assert version == OME_NGFF_SPECIFICATION_VERSION


def test_attributes_multiscales_name_with_server_name():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    name = root.attrs["multiscales"][0]["name"]

    assert name == sample_RGB_metadata.name


def test_attributes_multiscales_name_with_custom_name():
    expected_name = "some name"
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, name=expected_name)
    root = zarr.Group(store=pyramid_store)

    name = root.attrs["multiscales"][0]["name"]

    assert name == expected_name


def test_attributes_multiscales_axis_length_when_dimensions_squeezed():
    expected_axis_length = 2 + sum(
        n > 1 for n in [sample_RGB_metadata.n_timepoints, sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices]
    )      # number of dimensions that have more than one element
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, squeeze=True)
    root = zarr.Group(store=pyramid_store)

    axis_length = len(root.attrs["multiscales"][0]["axes"])

    assert axis_length == expected_axis_length


def test_attributes_multiscales_axis_length_when_dimensions_not_squeezed():
    expected_axis_length = 5
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, squeeze=False)
    root = zarr.Group(store=pyramid_store)

    axis_length = len(root.attrs["multiscales"][0]["axes"])

    assert axis_length == expected_axis_length


def test_attributes_multiscales_x_axis_name():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    axis_x_name = root.attrs["multiscales"][0]["axes"][-1]["name"]

    assert axis_x_name == "x"


def test_attributes_multiscales_x_axis_type():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    axis_x_type = root.attrs["multiscales"][0]["axes"][-1]["type"]

    assert axis_x_type == "space"


def test_attributes_multiscales_x_axis_unit():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    axis_x_unit = root.attrs["multiscales"][0]["axes"][-1]["unit"]

    assert axis_x_unit == sample_RGB_metadata.pixel_calibration.length_x.unit


def test_attributes_multiscales_datasets_length_with_server_downsamples():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    datasets_length = len(root.attrs["multiscales"][0]["datasets"])

    assert datasets_length == sample_RGB_metadata.n_resolutions


def test_attributes_multiscales_datasets_length_with_custom_downsamples():
    downsamples = [1, 4, 8, 16]
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, downsamples=downsamples)
    root = zarr.Group(store=pyramid_store)

    datasets_length = len(root.attrs["multiscales"][0]["datasets"])

    assert datasets_length == len(downsamples)


def test_attributes_multiscales_datasets_scale_of_full_resolution_with_server_downsamples_and_dimensions_squeezed():
    expected_scales = tuple(sum(
        n > 1 for n in [sample_RGB_metadata.n_timepoints, sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices]
    ) * [1.0] + [sample_RGB_metadata.downsamples[0], sample_RGB_metadata.downsamples[0]])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, squeeze=True)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_attributes_multiscales_datasets_scale_of_full_resolution_with_server_downsamples_and_dimensions_not_squeezed():
    expected_scales = (1.0, 1.0, 1.0, sample_RGB_metadata.downsamples[0], sample_RGB_metadata.downsamples[0])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, squeeze=False)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_attributes_multiscales_datasets_scale_of_full_resolution_with_custom_downsamples():
    downsamples = [2]
    expected_scales = tuple(sum(
        n > 1 for n in [sample_RGB_metadata.n_timepoints, sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices]
    ) * [1.0] + [downsamples[0], downsamples[0]])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, downsamples=downsamples)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_attributes_multiscales_datasets_scale_of_lowest_resolution_with_server_downsamples():
    expected_scales = tuple(sum(
        n > 1 for n in [sample_RGB_metadata.n_timepoints, sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices]
    ) * [1.0] + [sample_RGB_metadata.downsamples[-1], sample_RGB_metadata.downsamples[-1]])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][-1]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_attributes_multiscales_datasets_scale_of_lowest_resolution_with_custom_downsamples():
    downsamples = [1, 4, 8, 16]
    expected_scales = tuple(sum(
        n > 1 for n in [sample_RGB_metadata.n_timepoints, sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices]
    ) * [1.0] + [downsamples[-1], downsamples[-1]])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, downsamples=downsamples)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][-1]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_array_chunks_when_size_not_specified():
    expected_chunks = [
        1 if sample_RGB_metadata.n_timepoints > 1 else None,
        sample_RGB_metadata.n_channels if sample_RGB_metadata.n_channels > 1 else None,
        1 if sample_RGB_metadata.n_z_slices > 1 else None,
        min(1024, sample_RGB_metadata.height),
        min(1024, sample_RGB_metadata.width)
    ]
    expected_chunks = tuple([c for c in expected_chunks if c is not None])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[0]

    chunks = array.chunks

    assert chunks == expected_chunks


def test_array_chunks_when_one_size_specified():
    chunk_size = 4
    expected_chunks = [
        1 if sample_RGB_metadata.n_timepoints > 1 else None,
        sample_RGB_metadata.n_channels if sample_RGB_metadata.n_channels > 1 else None,
        1 if sample_RGB_metadata.n_z_slices > 1 else None,
        min(chunk_size, sample_RGB_metadata.height),
        min(chunk_size, sample_RGB_metadata.width)
    ]
    expected_chunks = tuple([c for c in expected_chunks if c is not None])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, chunk_size=chunk_size)
    root = zarr.Group(store=pyramid_store)
    array = root[0]

    chunks = array.chunks

    assert chunks == expected_chunks


def test_array_chunks_when_two_sizes_specified():
    chunk_size = (4, 8)
    expected_chunks = [
        1 if sample_RGB_metadata.n_timepoints > 1 else None,
        sample_RGB_metadata.n_channels if sample_RGB_metadata.n_channels > 1 else None,
        1 if sample_RGB_metadata.n_z_slices > 1 else None,
        min(chunk_size[1], sample_RGB_metadata.height),
        min(chunk_size[0], sample_RGB_metadata.width)
    ]
    expected_chunks = tuple([c for c in expected_chunks if c is not None])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block, chunk_size=chunk_size)
    root = zarr.Group(store=pyramid_store)
    array = root[0]

    chunks = array.chunks

    assert chunks == expected_chunks


def test_array_dtype():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[0]

    dtype = array.dtype

    assert dtype == sample_RGB_metadata.dtype


def test_array_shape_of_full_resolution():
    level = 0
    expected_shape = [
        sample_RGB_metadata.shapes[level].t if sample_RGB_metadata.shapes[level].t > 1 else None,
        sample_RGB_metadata.shapes[level].c if sample_RGB_metadata.shapes[level].c > 1 else None,
        sample_RGB_metadata.shapes[level].z if sample_RGB_metadata.shapes[level].z > 1 else None,
        sample_RGB_metadata.shapes[level].y,
        sample_RGB_metadata.shapes[level].x
    ]
    expected_shape = tuple([c for c in expected_shape if c is not None])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[level]

    shape = array.shape

    assert shape == expected_shape


def test_array_shape_of_lowest_resolution():
    level = sample_RGB_metadata.n_resolutions-1
    expected_shape = [
        sample_RGB_metadata.shapes[level].t if sample_RGB_metadata.shapes[level].t > 1 else None,
        sample_RGB_metadata.shapes[level].c if sample_RGB_metadata.shapes[level].c > 1 else None,
        sample_RGB_metadata.shapes[level].z if sample_RGB_metadata.shapes[level].z > 1 else None,
        sample_RGB_metadata.shapes[level].y,
        sample_RGB_metadata.shapes[level].x
    ]
    expected_shape = tuple([c for c in expected_shape if c is not None])
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)

    root = zarr.Group(store=pyramid_store)
    array = root[level]

    shape = array.shape

    assert shape == expected_shape


def test_full_resolution_pixel_values_of_rgb_image():
    level = 0
    expected_pixels = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[level]

    pixel_values = array[:]

    np.testing.assert_array_equal(pixel_values, expected_pixels)


def test_lowest_resolution_pixel_values_of_rgb_image():
    level = sample_RGB_metadata.n_resolutions-1
    expected_pixels = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server.metadata, sample_RGB_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[level]

    pixel_values = array[:]

    np.testing.assert_array_equal(pixel_values, expected_pixels)


def test_full_resolution_pixel_values_of_float_image():
    level = 0
    expected_pixels = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
    sample_float_server = SampleFloat32Server()
    pyramid_store = PyramidStore(sample_float_server.metadata, sample_float_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[level]

    pixel_values = array[:]

    np.testing.assert_array_equal(pixel_values, expected_pixels)


def test_lowest_resolution_pixel_values_of_float_image():
    level = sample_float32_metadata.n_resolutions-1
    expected_pixels = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)
    sample_float_server = SampleFloat32Server()
    pyramid_store = PyramidStore(sample_float_server.metadata, sample_float_server._read_block)
    root = zarr.Group(store=pyramid_store)
    array = root[level]

    pixel_values = array[:]

    np.testing.assert_array_equal(pixel_values, expected_pixels)
