import geojson.geometry
import geojson
from typing import Union


class ImagePlane(dict):
    """
    An additional property of a geometry containing z and t indices.
    """

    def __init__(self, z: int = 0, t: int = 0):
        """
        :param z: the z-stack of the geometry
        :param t: the time point of the geometry
        """
        dict.__init__(self, z=z, t=t)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError('ImagePlane is immutable')

    def __delattr__(self, name):
        raise AttributeError('ImagePlane is immutable')


def add_plane_to_geometry(
        geometry: Union[geojson.geometry.Geometry, geojson.Feature],
        z: int = None,
        t: int = None,
        preferred_geometry_key: str =None
) -> geojson.geometry.Geometry:
    """
    Create a GeoJSON Geometry object with an additional 'plane' property containing z and t indices.

    If indices are unspecified, they will be taken from the geometry's 'plane' property, if present,
    or else z or t attributes otherwise.
    If z and t values cannot be found in either of these locations, defaults (z=0 and t=0) will be
    used instead.

    The 'plane' property is immutable.

    :param geometry: a GeoJSON feature or geometry. It must not be a FeatureCollection
    :param z: the z-stack index this geometry should have
    :param t: the timepoint this geometry should have
    :param preferred_geometry_key: if the provided geometry is a feature, and this feature has a 'extra_geometries'
                                   property, then this parameter can be used as a key to retrieve the geometry to
                                   consider from the 'extra_geometries' dictionnary
    :return: a GeoJSON Geometry containing an additional 'plane' property containing z and t indices
    :raises ValueError: when the provided geometry is a FeatureCollection
    """
    if geometry is None:
        return None
    
    if geometry['type'] == 'Feature':
        feature = geometry
        if preferred_geometry_key is not None and 'extra_geometries' in feature.properties:
            geometry = feature.properties['extra_geometries'].get(preferred_geometry_key, feature.geometry)
        else:
            geometry = feature.geometry
    elif geometry['type'] == 'FeatureCollection':
        raise ValueError('Cannot convert FeatureCollection to single Geometry')

    if z is None or t is None:
        plane = getattr(geometry, 'plane', None)
        if z is None:
            z = getattr(plane, 'z', 0) if plane is not None else getattr(geometry, 'z', 0)
        if t is None:
            t = getattr(plane, 't', 0) if plane is not None else getattr(geometry, 't', 0)
    
    geometry = geojson.GeoJSON.to_instance(geometry, strict=False)
    geometry.plane = ImagePlane(z, t)
    return geometry
