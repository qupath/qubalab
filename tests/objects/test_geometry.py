import geojson
import pytest
from qubalab.objects.geometry import add_plane_to_geometry


def test_plane_added_to_geometry():
    geometry = geojson.Point((-115.81, 37.24))

    plane_geometry = add_plane_to_geometry(geometry)

    assert 'plane' in plane_geometry


def test_plane_added_to_feature():
    feature = geojson.Feature(geometry=geojson.Point((-115.81, 37.24)))

    plane_geometry = add_plane_to_geometry(feature)

    assert 'plane' in plane_geometry


def test_plane_added_to_feature_collection():
    feature_collection = geojson.FeatureCollection([geojson.Feature(geometry=geojson.Point((-115.81, 37.24)))])

    with pytest.raises(ValueError):
        add_plane_to_geometry(feature_collection)



def test_z_when_not_specified():
    geometry = geojson.Point((-115.81, 37.24))

    plane_geometry = add_plane_to_geometry(geometry)

    assert plane_geometry.plane.z == 0


def test_z_when_specified():
    z = 5
    geometry = geojson.Point((-115.81, 37.24))

    plane_geometry = add_plane_to_geometry(geometry, z=z)

    assert plane_geometry.plane.z == z


def test_t_when_not_specified():
    geometry = geojson.Point((-115.81, 37.24))

    plane_geometry = add_plane_to_geometry(geometry)

    assert plane_geometry.plane.t == 0


def test_z_when_specified():
    t = 3
    geometry = geojson.Point((-115.81, 37.24))

    plane_geometry = add_plane_to_geometry(geometry, t=t)

    assert plane_geometry.plane.t == t


def test_preferred_geometry_used():
    base_geometry = geojson.Point((-115.81, 37.24))
    preferred_geometry = geojson.Point((43, -78))
    preferred_geometry_key = "some_key"
    feature = geojson.Feature(geometry=base_geometry, properties={
        "extra_geometries": {
            preferred_geometry_key: preferred_geometry
        }
    })

    plane_geometry = add_plane_to_geometry(feature, preferred_geometry_key=preferred_geometry_key)

    assert plane_geometry.coordinates == preferred_geometry.coordinates
