import geojson
from typing import Iterable


def geojson_features_from_string(json_string: str, **kwargs) -> Iterable[geojson.Feature]:
    """
    Read features from a GeoJSON string.

    If the string encodes a feature collection, the features themselves will be extracted.

    :param json_string: a string representing a GeoJSON. NaNs values are allowed
    :param \**kwargs: additional parameters to pass to the geojson loader
    :returns: a collection of features representing the provided string
    """
    results = _geojson_from_string(json_string, **kwargs)
    if 'features' in results:
        results = results['features']
    return results


def _geojson_from_string(json: str, **kwargs):
    """
    Read a GeoJSON string.
    This is a wrapper around geojson.loads that allows for NaNs by default (and is generally non-strict with numbers).
    """
    # Default parse constant is _enforce_strict_numbers, which fails on NaNs
    if 'parse_constant' in kwargs:
        return geojson.loads(json, **kwargs)
    else:
        return geojson.loads(json, parse_constant=None, **kwargs)