import numpy as np
import tifffile
import imageio.v3 as iio
import base64
from unittest.mock import Mock
from qubalab.images.qupath_server import QuPathServer, PixelAccess
from qubalab.images.region_2d import Region2D
from qubalab.images.metadata.image_server_metadata import ImageServerMetadata
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength


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
sample_RGB_pixels = [[[[
    x / shape.x * 255 if c == 0 else (y / shape.y * 255 if c == 1 else 0)
    for x in range(shape.x)]
    for y in range(shape.y)]
    for c in range(shape.c)]
    for shape in sample_RGB_metadata.shapes
]


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


def _create_qupath_metadata(metadata: ImageServerMetadata):
    qupath_metadata = Mock()
    qupath_metadata.getName.return_value = metadata.name

    qupath_levels = []
    for shape in metadata.shapes:
        level = Mock()
        level.getWidth.return_value = shape.x
        level.getHeight.return_value = shape.y
        qupath_levels.append(level)
    qupath_metadata.getLevels.return_value = qupath_levels

    qupath_channels = []
    for channel in metadata.channels:
        qupath_channel = Mock()
        qupath_channel.getName.return_value = channel.name
        qupath_channel.getColor.return_value = ((255 & 0xff)<<24) + \
            ((int(channel.color[0]) * 255 & 0xff)<<16) + \
            ((int(channel.color[1] * 255) & 0xff)<<8) +  \
            (int(channel.color[2] * 255) & 0xff)
        qupath_channels.append(qupath_channel)
    qupath_metadata.getChannels.return_value = qupath_channels

    return qupath_metadata


def _create_qupath_server(metadata: ImageServerMetadata):
    qupath_server = Mock()
    qupath_server.getMetadata.return_value = _create_qupath_metadata(metadata)
    qupath_server.getURIs.return_value = ("file://" + metadata.path,)
    qupath_server.nChannels.return_value = metadata.n_channels
    qupath_server.nZSlices.return_value = metadata.n_z_slices
    qupath_server.nTimepoints.return_value = metadata.n_timepoints
    qupath_server.isRGB.return_value = metadata.is_rgb
    qupath_server.getPreferredDownsamples.return_value = metadata.downsamples
    qupath_server.getDownsampleForResolution.side_effect = lambda level : metadata.downsamples[level]
    qupath_server.getPath.return_value = metadata.path

    pixel_type_text = Mock()
    pixel_type_text.lower.return_value = metadata.dtype
    pixel_type = Mock()
    pixel_type.toString.return_value = pixel_type_text
    qupath_server.getPixelType.return_value = pixel_type

    pixel_calibration = Mock()
    pixel_calibration.hasPixelSizeMicrons.return_value = metadata.pixel_calibration.length_x.unit == metadata.pixel_calibration.length_y.unit == "micrometer"
    pixel_calibration.hasZSpacingMicrons.return_value = metadata.pixel_calibration.length_z.unit == "micrometer"
    pixel_calibration.getPixelWidthMicrons.return_value = metadata.pixel_calibration.length_x.length
    pixel_calibration.getPixelHeightMicrons.return_value = metadata.pixel_calibration.length_y.length
    pixel_calibration.getZSpacingMicrons.return_value = metadata.pixel_calibration.length_z.length
    qupath_server.getPixelCalibration.return_value = pixel_calibration

    return qupath_server


def _create_gateway(metadata: ImageServerMetadata, pixels: list):
    def _create_region_request(path, downsample, x, y, width, height, z, t):
        return {
            "downsample": downsample,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "z": z,
            "t": t
        }
    
    def _get_tile(request):
        level = metadata.downsamples.index(request["downsample"])
        image = np.array(pixels[level], dtype=metadata.dtype)
        
        if metadata.n_timepoints > 1:
            image = image[request["t"], ...]
        if metadata.n_z_slices > 1:
            image = image[..., request["z"], :, :]
        image = image[:, request["y"]:request["y"]+request["height"], request["x"]:request["x"]+request["width"]]
        image = np.moveaxis(image, 0, -1)   # move channel axis

        return image

    def _write_image_region(server, request, path):
        with tifffile.TiffWriter(path, bigtiff=True) as tif:
            tif.write(_get_tile(request), photometric='rgb')

    def _get_image_bytes(server, request, format):
        return iio.imwrite("<bytes>", _get_tile(request), extension=".tiff", photometric='rgb')

    def _get_image_base64(server, request, format):
        return base64.b64encode(_get_image_bytes(server, request, format))

    gateway = Mock()

    gateway.jvm.qupath.lib.regions.RegionRequest.createInstance.side_effect = _create_region_request
    gateway.entry_point.writeImageRegion.side_effect = _write_image_region
    gateway.entry_point.getImageBytes.side_effect = _get_image_bytes
    gateway.entry_point.getImageBase64.side_effect = _get_image_base64

    return gateway


def test_RGB_metadata():
    metadata = sample_RGB_metadata
    gateway = _create_gateway(metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(metadata)
    server = QuPathServer(gateway, qupath_server)

    metadata = server.metadata

    assert metadata == metadata


def test_RGB_full_resolution_with_temp_file():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.TEMP_FILES)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_lowest_resolution_with_temp_file():
    level = sample_RGB_metadata.n_resolutions - 1
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.TEMP_FILES)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_tile_of_full_resolution_with_temp_file():
    level = 0
    width = 5
    height = 6
    x = 10
    y = 20
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)[:, y:y+height, x:x+width]
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.TEMP_FILES)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(x, y, width, height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_full_resolution_with_bytes():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BYTES)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_lowest_resolution_with_bytes():
    level = sample_RGB_metadata.n_resolutions - 1
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BYTES)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_tile_of_full_resolution_with_bytes():
    level = 0
    width = 5
    height = 6
    x = 10
    y = 20
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)[:, y:y+height, x:x+width]
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BYTES)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(x, y, width, height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_full_resolution_with_base64():
    level = 0
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BASE_64)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_lowest_resolution_with_base64():
    level = sample_RGB_metadata.n_resolutions - 1
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BASE_64)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(width=sample_RGB_metadata.width, height=sample_RGB_metadata.height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_RGB_tile_of_full_resolution_with_base64():
    level = 0
    width = 5
    height = 6
    x = 10
    y = 20
    expected_image = np.array(sample_RGB_pixels[level], dtype=sample_RGB_metadata.dtype)[:, y:y+height, x:x+width]
    gateway = _create_gateway(sample_RGB_metadata, sample_RGB_pixels)
    qupath_server = _create_qupath_server(sample_RGB_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BASE_64)

    image = server.read_region(
        sample_RGB_metadata.downsamples[level],
        Region2D(x, y, width, height)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_metadata():
    metadata = sample_float32_metadata
    gateway = _create_gateway(metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(metadata)
    server = QuPathServer(gateway, qupath_server)

    metadata = server.metadata

    assert metadata == metadata


def test_float32_full_resolution_with_temp_file():
    level = 0
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.TEMP_FILES)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_lowest_resolution_with_temp_file():
    level = sample_float32_metadata.n_resolutions - 1
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.TEMP_FILES)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_tile_of_full_resolution_with_temp_file():
    level = 0
    width = 5
    height = 6
    x = 10
    y = 20
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, y:y+height, x:x+width]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.TEMP_FILES)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(x, y, width, height, z, t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_full_resolution_with_bytes():
    level = 0
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BYTES)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_lowest_resolution_with_bytes():
    level = sample_float32_metadata.n_resolutions - 1
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BYTES)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_tile_of_full_resolution_with_bytes():
    level = 0
    width = 5
    height = 6
    x = 10
    y = 20
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, y:y+height, x:x+width]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BYTES)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(x, y, width, height, z, t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_full_resolution_with_base64():
    level = 0
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BASE_64)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_lowest_resolution_with_base64():
    level = sample_float32_metadata.n_resolutions - 1
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, ...]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BASE_64)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(width=sample_float32_metadata.width, height=sample_float32_metadata.height, z=z, t=t)
    )

    np.testing.assert_array_equal(image, expected_image)


def test_float32_tile_of_full_resolution_with_base64():
    level = 0
    width = 5
    height = 6
    x = 10
    y = 20
    t = int(sample_float32_metadata.n_timepoints / 2)
    z = int(sample_float32_metadata.n_z_slices / 2)
    expected_image = np.array(sample_float32_pixels[level], dtype=sample_float32_metadata.dtype)[t, :, z, y:y+height, x:x+width]
    gateway = _create_gateway(sample_float32_metadata, sample_float32_pixels)
    qupath_server = _create_qupath_server(sample_float32_metadata)
    server = QuPathServer(gateway, qupath_server, PixelAccess.BASE_64)

    image = server.read_region(
        sample_float32_metadata.downsamples[level],
        Region2D(x, y, width, height, z, t)
    )

    np.testing.assert_array_equal(image, expected_image)