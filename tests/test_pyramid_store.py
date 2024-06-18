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
    def _build_metadata(self) -> ImageServerMetadata:
        return sample_RGB_metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
        return image[region.y:region.y+region.height, region.x:region.x+region.width, :]
    
    def close(self):
        pass


def test_group_length():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    length = len(root)

    assert length == sample_RGB_metadata.n_resolutions


def test_group_read_only():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    with pytest.raises(RuntimeError):
        root.create_group("group")


def test_attributes_multiscales_version():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    version = root.attrs["multiscales"][0]["version"]

    assert version == OME_NGFF_SPECIFICATION_VERSION


def test_attributes_multiscales_name_when_not_specified():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    name = root.attrs["multiscales"][0]["name"]

    assert name == sample_RGB_metadata.name


def test_attributes_multiscales_name_when_specified():
    expected_name = "some name"
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server, name=expected_name)
    root = zarr.Group(store=pyramid_store)

    name = root.attrs["multiscales"][0]["name"]

    assert name == expected_name


def test_attributes_multiscales_axis_length():
    expected_axis_length = expected_axis_length = 2 + sum(
        n > 1 for n in [sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices, sample_RGB_metadata.n_timepoints]
    )      # number of dimensions that have more than one element
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    axis_length = len(root.attrs["multiscales"][0]["axes"])

    assert axis_length == expected_axis_length


def test_attributes_multiscales_x_axis_name():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    axis_x_name = root.attrs["multiscales"][0]["axes"][-1]["name"]

    assert axis_x_name == "x"


def test_attributes_multiscales_x_axis_type():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    axis_x_type = root.attrs["multiscales"][0]["axes"][-1]["type"]

    assert axis_x_type == "space"


def test_attributes_multiscales_x_axis_unit():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    axis_x_unit = root.attrs["multiscales"][0]["axes"][-1]["unit"]

    assert axis_x_unit == sample_RGB_metadata.pixel_calibration.length_x.unit


def test_attributes_multiscales_datasets_length():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    datasets_length = len(root.attrs["multiscales"][0]["datasets"])

    assert datasets_length == sample_RGB_metadata.n_resolutions


def test_attributes_multiscales_datasets_scale_of_full_resolution():
    expected_scales = sum(
        n > 1 for n in [sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices, sample_RGB_metadata.n_timepoints]
    ) * [1.0] + [sample_RGB_metadata.downsamples[0], sample_RGB_metadata.downsamples[0]]
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_attributes_multiscales_datasets_scale_of_lowest_resolution():
    expected_scales = sum(
        n > 1 for n in [sample_RGB_metadata.n_channels, sample_RGB_metadata.n_z_slices, sample_RGB_metadata.n_timepoints]
    ) * [1.0] + [sample_RGB_metadata.downsamples[-1], sample_RGB_metadata.downsamples[-1]]
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)

    scales = root.attrs["multiscales"][0]["datasets"][-1]["coordinateTransformations"][0]["scale"]

    assert scales == expected_scales


def test_full_resolution_array_dtype():
    sample_RGB_server = SampleRGBServer()
    pyramid_store = PyramidStore(sample_RGB_server)
    root = zarr.Group(store=pyramid_store)
    array = root[0]

    dtype = array.dtype

    assert dtype == sample_RGB_metadata.dtype

#TODO: check metadata when different parameters sent to PyramidStore