import pytest
from qubalab.images.image_shape import ImageShape


def test_as_tuple():
    x = 1
    y = 2
    z = 3
    t = 4
    c = 5
    expected_tuple = (t, c, z, y, x)
    image_shape = ImageShape(x=x, y=y, t=t, z=z, c=c)

    tuple = image_shape.as_tuple()

    assert expected_tuple == tuple


def test_as_tuple_with_parameters():
    x = 1
    y = 2
    z = 3
    t = 4
    c = 5
    expected_tuple = (y, x)
    image_shape = ImageShape(x=x, y=y, t=t, z=z, c=c)

    tuple = image_shape.as_tuple('yx')

    assert expected_tuple == tuple


def test_as_tuple_with_invalid_parameters():
    with pytest.raises(AttributeError):
        image_shape = ImageShape(0, 0)

        image_shape.as_tuple('a')


def test_from_tuple():
    x = 1
    y = 2
    z = 3
    t = 4
    c = 5
    expected_image_shape = ImageShape(x=x, y=y, t=t, z=z, c=c)

    image_shape = ImageShape.from_tczyx(t, c, z, y, x)

    assert expected_image_shape == image_shape


def test_from_invalid_tuple():
    with pytest.raises(IndexError):
        ImageShape.from_tczyx()
