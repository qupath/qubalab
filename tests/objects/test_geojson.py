import geojson
from qubalab.objects.geojson import geojson_features_from_string


def test_geojson_from_feature():
    expected_feature = geojson.Feature(geometry=geojson.Point((1.6432, -19.123)))
    text = geojson.dumps(expected_feature)

    features = geojson_features_from_string(text)

    assert features == expected_feature


def test_geojson_from_feature_collection():
    expected_features = [geojson.Feature(geometry=geojson.Point((1.6432, -19.123))), geojson.Feature(geometry=geojson.Point((-80.234, -22.532)))]
    feature_collection = geojson.FeatureCollection(expected_features)
    text = geojson.dumps(feature_collection)

    features = geojson_features_from_string(text)

    assert features == expected_features


def test_geojson_from_feature_with_Nan_value():
    expected_feature = geojson.Feature(geometry=geojson.Point((float('nan'), -19.123)))
    text = geojson.dumps(expected_feature)

    features = geojson_features_from_string(text)

    assert features == expected_feature