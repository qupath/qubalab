from typing import Union
from PIL import Image, ImageDraw
import shapely


def draw_geometry(image_size: tuple[int, int], drawing_context: ImageDraw.Draw, geometry: shapely.Geometry, value: int):
    """
    Draw the provided Shapely geometry with the provided drawing context using the provided value.

    :param image_size: the size of the image to draw on
    :param drawing_context: the drawing context to use
    :param geometry: the geometry to draw
    :param value: the value to use when drawing
    """
    if (isinstance(geometry, shapely.Point)):
        _draw_point(drawing_context, geometry, value)
    elif (isinstance(geometry, Union[shapely.LineString, shapely.LinearRing])):
        _draw_line(drawing_context, geometry, value)
    elif (isinstance(geometry, shapely.Polygon)):
        _draw_polygon(image_size, drawing_context, geometry, value)
    elif (isinstance(geometry, shapely.MultiPoint)):
        for point in geometry.geoms:
            _draw_point(drawing_context, point, value)
    elif (isinstance(geometry, shapely.MultiLineString)):
        for line in geometry.geoms:
            _draw_line(drawing_context, line, value)
    elif (isinstance(geometry, shapely.MultiPolygon)):
        for polygon in geometry.geoms:
            _draw_polygon(image_size, drawing_context, polygon, value)
    elif (isinstance(geometry, shapely.GeometryCollection)):
        for g in geometry.geoms:
            draw_geometry(image_size, drawing_context, g, value)

def _draw_point(drawing_context: ImageDraw, point: shapely.Point, value: int):
    drawing_context.point([point.x, point.y], value)

def _draw_line(drawing_context: ImageDraw, line: Union[shapely.LineString, shapely.LinearRing], value: int):
    drawing_context.line(line.coords, value)

def _draw_polygon(image_size: tuple[int, int], drawing_context: ImageDraw, polygon: shapely.Polygon, value: int):
    # If we have holes, we risk overpainting existing pixels with the background color
    # For that reason we need to create a separate binary image and then copy the values
    if len(polygon.interiors) > 0:
        bitmap = Image.new('1', image_size)
        bitmap_draw = ImageDraw.Draw(bitmap)

        bitmap_draw.polygon(polygon.exterior.coords, 1)
        for interior in polygon.interiors:
            bitmap_draw.polygon(interior.coords, 0)

        drawing_context.bitmap((0, 0), bitmap, value)
    else:
        drawing_context.polygon(polygon.exterior.coords, value)
