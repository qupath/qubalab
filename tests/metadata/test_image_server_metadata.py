import numpy as np
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.image_server_metadata import ImageServerMetadata
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength
from qubalab.images.metadata.image_channel import ImageChannel


def test_shape():
    expected_shape = ImageShape(128, 100)
    metadata = ImageServerMetadata(
        "",
        "",
        (
            expected_shape,
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None
    )

    shape = metadata.shape

    assert expected_shape == shape

def test_width():
    expected_width = 128
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(expected_width, 100),
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None
    )

    width = metadata.width

    assert expected_width == width

def test_height():
    expected_height = 100
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, expected_height),
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None
    )

    height = metadata.height

    assert expected_height == height

def test_n_channels():
    expected_n_channels = 5
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100, c=expected_n_channels),
            ImageShape(64, 50, c=expected_n_channels),
            ImageShape(32, 25, c=expected_n_channels)
        ),
        None,
        True,
        None
    )

    n_channels = metadata.n_channels

    assert expected_n_channels == n_channels

def test_n_timepoints():
    expected_n_timepoints = 50
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100, t=expected_n_timepoints),
            ImageShape(64, 50, t=expected_n_timepoints),
            ImageShape(32, 25, t=expected_n_timepoints)
        ),
        None,
        True,
        None
    )

    n_timepoints = metadata.n_timepoints

    assert expected_n_timepoints == n_timepoints

def test_n_z_slices():
    expected_n_z_slices = 10
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100, z=expected_n_z_slices),
            ImageShape(64, 50, z=expected_n_z_slices),
            ImageShape(32, 25, z=expected_n_z_slices)
        ),
        None,
        True,
        None
    )

    n_z_slices = metadata.n_z_slices

    assert expected_n_z_slices == n_z_slices

def test_n_resolutions():
    expected_n_resolutions = 3
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100),
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None
    )

    n_resolutions = metadata.n_resolutions

    assert expected_n_resolutions == n_resolutions

def test_downsamples():
    expected_downsamples = (1, 2, 4)
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100),
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None
    )

    downsamples = metadata.downsamples

    assert expected_downsamples == downsamples

def test_channels_when_provided():
    expected_channels = (ImageChannel("Channel 1", (1, 1, 1)), ImageChannel("Channel 2", (0, 0, 0)))
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100),
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None,
        expected_channels
    )

    channels = metadata.channels

    assert expected_channels == channels

def test_number_of_channels_when_not_provided_and_RGB():
    expected_n_channels = 3
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100),
            ImageShape(64, 50),
            ImageShape(32, 25)
        ),
        None,
        True,
        None
    )

    n_channels = len(metadata.channels)

    assert expected_n_channels == n_channels

def test_number_of_channels_when_not_provided_and_not_RGB():
    expected_n_channels = 5
    metadata = ImageServerMetadata(
        "",
        "",
        (
            ImageShape(128, 100, c=expected_n_channels),
            ImageShape(64, 50, c=expected_n_channels),
            ImageShape(32, 25, c=expected_n_channels)
        ),
        None,
        False,
        None
    )

    n_channels = len(metadata.channels)

    assert expected_n_channels == n_channels
    