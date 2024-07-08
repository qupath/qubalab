import numpy as np
import shapely
from PIL import Image, ImageDraw
from qubalab.objects.draw import draw_geometry


def test_draw_point():
    v = 60
    geometry = shapely.Point(1, 3)
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, v, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_line_string():
    v = 32
    geometry = shapely.LineString([(1, 3), (2, 3), (4, 4), (1, 8)])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, v, v, 0, 0],
         [0, 0, 0, v, v],
         [0, 0, 0, v, 0],
         [0, 0, v, 0, 0],
         [0, 0, v, 0, 0],
         [0, v, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_linear_ring():
    v = 57
    geometry = shapely.LinearRing([(1, 1), (3, 4), (1, 7)])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[0, 0, 0, 0, 0],
         [0, v, 0, 0, 0],
         [0, v, v, 0, 0],
         [0, v, v, 0, 0],
         [0, v, 0, v, 0],
         [0, v, v, 0, 0],
         [0, v, v, 0, 0],
         [0, v, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_polygon_without_holes():
    v = 573
    geometry = shapely.Polygon([(0, 0), (3, 3), (3, 4), (0, 7)])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[v, 0, 0, 0, 0],
         [v, v, 0, 0, 0],
         [v, v, v, 0, 0],
         [v, v, v, v, 0],
         [v, v, v, v, 0],
         [v, v, v, 0, 0],
         [v, v, 0, 0, 0],
         [v, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_polygon_with_holes():
    v = 9
    geometry = shapely.Polygon([(0, 0), (3, 3), (3, 4), (0, 7)], holes=[[(1, 3), (2, 3), (2, 4)]])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[v, 0, 0, 0, 0],
         [v, v, 0, 0, 0],
         [v, v, v, 0, 0],
         [v, 0, 0, v, 0],
         [v, v, 0, v, 0],
         [v, v, v, 0, 0],
         [v, v, 0, 0, 0],
         [v, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_multi_point():
    v = 4
    geometry = shapely.MultiPoint([shapely.Point(1, 3), shapely.Point(3, 6)])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, v, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, v, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_multi_line_string():
    v = 32
    geometry = shapely.MultiLineString([shapely.LineString([(1, 3), (2, 3), (4, 4), (1, 8)]), shapely.LineString([(4, 2), (4, 7)])])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, v],
         [0, v, v, 0, v],
         [0, 0, 0, v, v],
         [0, 0, 0, v, v],
         [0, 0, v, 0, v],
         [0, 0, v, 0, v],
         [0, v, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_multi_polygon():
    v = 573
    geometry = shapely.MultiPolygon([
        shapely.Polygon([(0, 0), (3, 3), (3, 4), (0, 7)]),
        shapely.Polygon([(2, 7), (4, 7), (4, 9), (2, 9)], holes=[[(3, 8), (3, 8), (3, 8)]])
    ])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[v, 0, 0, 0, 0],
         [v, v, 0, 0, 0],
         [v, v, v, 0, 0],
         [v, v, v, v, 0],
         [v, v, v, v, 0],
         [v, v, v, 0, 0],
         [v, v, 0, 0, 0],
         [v, 0, v, v, v],
         [0, 0, v, 0, v],
         [0, 0, v, v, v],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_geometry_collection():
    v = 7
    geometry = shapely.GeometryCollection([shapely.Point(1, 2), shapely.LineString([(1, 3), (2, 3), (4, 4), (1, 8)])])
    pixel_values = np.zeros((10, 5))
    expected_pixel_values = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, v, 0, 0, 0],
         [0, v, v, 0, 0],
         [0, 0, 0, v, v],
         [0, 0, 0, v, 0],
         [0, 0, v, 0, 0],
         [0, 0, v, 0, 0],
         [0, v, 0, 0, 0],
         [0, 0, 0, 0, 0],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)


def test_draw_when_values_already_present():
    i = 6
    v = 3
    geometry = shapely.Point(1, 3)
    pixel_values = np.full((10, 5), i, dtype=np.uint8)
    expected_pixel_values = np.array(
        [[i, i, i, i, i],
         [i, i, i, i, i],
         [i, i, i, i, i],
         [i, v, i, i, i],
         [i, i, i, i, i],
         [i, i, i, i, i],
         [i, i, i, i, i],
         [i, i, i, i, i],
         [i, i, i, i, i],
         [i, i, i, i, i],]
    )
    image = Image.fromarray(pixel_values)
    drawing_context = ImageDraw.Draw(image)

    draw_geometry(image.size, drawing_context, geometry, v)

    np.testing.assert_array_equal(np.asarray(image), expected_pixel_values)
