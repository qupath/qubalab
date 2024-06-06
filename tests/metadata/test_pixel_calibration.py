from qubalab.images.metadata.pixel_calibration import PixelLength, PixelCalibration


def test_default_pixel_length():
    pixel_length = PixelLength()

    is_default = pixel_length.is_default()

    assert is_default


def test_default_values_of_pixel_length():
    expected_pixel_length = PixelLength(1.0, 'pixels')

    pixel_length = PixelLength()

    assert expected_pixel_length == pixel_length


def test_length_and_unit_of_pixel_length_different():
    pixel_length = PixelLength()
    pixel_length_microns = PixelLength.create_microns(0.25)

    assert pixel_length != pixel_length_microns


def test_length():
    length = 0.25

    pixel_length = PixelLength.create_microns(length)

    assert pixel_length.length == length


def test_unit():
    unit = 'micrometer'

    pixel_length = PixelLength.create_microns(0)

    assert pixel_length.unit == unit


def test_default_pixel_calibration_not_calibrated():
    pixel_calibration = PixelCalibration()

    assert not pixel_calibration.is_calibrated()


def test_default_pixel_calibration_calibrated():
    pixel_calibration = PixelCalibration(length_x=PixelLength.create_microns(0.25))

    assert pixel_calibration.is_calibrated()
