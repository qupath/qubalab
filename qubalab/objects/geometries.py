from geojson.geometry import Geometry, Polygon
from geojson import GeoJSON
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from collections import namedtuple

"""
ImagePlane is an additional property of a geometry, containing z and t indices.
"""
ImagePlane = namedtuple('ImagePlane', ['z', 't'])


def to_geometry(geometry, z: int = None, t: int = None) -> Geometry:
    """
    Create a GeoJSON Geometry object with an additional 'plane' property containing z and t indices.
    If indices are unspecified, they will be taken from the geometry's 'plane' property, if present,
    or else z or t attributes otherwise.
    If z and t values cannot be found in either of these locations, defaults (z=0 and t=0) will be
    used instesad.
    """
    if geometry is None:
        return None
    if geometry['type'] == 'Feature':
        # If we have a feature, extract the geometry
        feature = geometry
        geometry = feature.geometry
    elif geometry['type'] == 'FeatureCollection':
        raise ValueError('Cannot convert FeatureCollection to single Geometry')

    if z is None or t is None:
        plane = getattr(geometry, 'plane', None)
        if z is None:
            z = plane.z if plane is not None else getattr(geometry, 'z', 0)
        if t is None:
            t = plane.t if plane is not None else getattr(geometry, 't', 0)
    geometry = GeoJSON.to_instance(geometry, strict=False)
    geometry.plane = ImagePlane(z=z, t=t)
    return geometry


def to_shapely(geometry: Geometry) -> BaseGeometry:
    """
    Convert a geometry to a shapely geometry
    """
    if isinstance(geometry, BaseGeometry):
        return geometry
    geom = to_geometry(geometry)
    return shape(geom)


def create_rectangle(x: float, y: float, width: float, height: float, z: int = None, t: int = None, validate=True) -> Polygon:
    """
    Create a rectangular geometry with an additional 'plane' property containing z and t indices.
    """
    polygon = Polygon(coordinates=[
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
        (x, y)], validate=validate)
    return to_geometry(polygon, z=z, t=t)