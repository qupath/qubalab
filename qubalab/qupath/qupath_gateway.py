import numpy as np
from py4j.java_gateway import JavaGateway, GatewayParameters, JavaObject
from py4j.protocol import Py4JNetworkError
from enum import Enum
from typing import Union
import geojson
from ..images import utils
from ..objects.image_feature import ImageFeature
from ..objects.object_type import ObjectType
from ..objects.geojson import geojson_features_from_string


_default_gateway = None


def create_gateway(auto_convert=True, auth_token=None, port=25333, set_as_default=True, **kwargs) -> JavaGateway:
    """
    Create a new JavaGateway to communicate with QuPath.

    This requires launching QuPath and activating Py4J from there first.

    :param auto_convert: if True, the gateway will try to automatically convert Python objects like sequences and maps to Java Objects
    :param auth_token: the authentication token to use to connect. Can be None if no authentication is used
    :param port: the port to use to connect
    :param set_as_default: whether to set the created gateway as the default gateway
    :param kwargs: additional arguments to give to the JavaGateway constructor
    :returns: the created gateway
    :raises RuntimeError: if the connection to QuPath couldn't be established
    """
    if auth_token is None:
        gateway_parameters = GatewayParameters(port=port)
    else:
        gateway_parameters = GatewayParameters(auth_token=auth_token, port=port)
    gateway = JavaGateway(
        auto_convert=auto_convert,
        gateway_parameters=gateway_parameters,
        **kwargs
    )

    try:
        # This will fail if QuPath is not running with a Py4J gateway
        qupath = gateway.entry_point.getQuPath()
        assert qupath is not None
    except (Py4JNetworkError, AssertionError) as err:
        raise RuntimeError('Could not connect to QuPath - is it running, and you have opened a Py4J gateway?') from err
    
    if set_as_default:
        set_default_gateway(gateway)
    return gateway


def set_default_gateway(gateway: JavaGateway):
    """
    Set the default JavaGateway to use if one is not otherwise specified.

    This default gateway will be used if no gateway is provided in some functions of this file.

    :param gateway: the gateway that should be the default gateway
    """
    global _default_gateway
    _default_gateway = gateway


def get_default_gateway() -> JavaGateway:
    """
    Return the default gateway. It will be created if it doesn't already exist.

    :returns: the default gateway
    """
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = create_gateway()
    return _default_gateway


def get_current_image_data(gateway: JavaGateway = None) -> JavaObject:
    """
    Get the current ImageData opened in QuPath through the provided gateway.

    :param gateway: the gateway to use. Can be None to use the default gateway
    :returns: the current ImageData opened in QuPath
    """
    gateway = get_default_gateway() if gateway is None else gateway
    return gateway.entry_point.getQuPath().getImageData()


def get_project(gateway: JavaGateway = None) -> JavaObject:
    """
    Return the currently opened QuPath project.

    :param gateway: the gateway to use. Can be None to use the default gateway
    :return: a Java Object representing the currently opened QuPath project
    """
    gateway = get_default_gateway() if gateway is None else gateway
    return gateway.entry_point.getProject()


def create_snapshot(gateway: JavaGateway = None, snapshot_type: str = 'qupath') -> np.ndarray:
    """
    Create and return a snapshot of QuPath. 

    :param gateway: the gateway to use. Can be None to use the default gateway
    :param snapshot_type: what to include in the snapshot. 'qupath' for the entire qupath window or
                          'viewer' for only the viewer
    :returns: a numpy array with dimensions (y, x, c) representing an RGB image
    :raises ValueError: if the snapshot type was not recognized
    """
    gateway = get_default_gateway() if gateway is None else gateway
    qp = gateway.entry_point

    if snapshot_type == 'qupath':
        image = qp.snapshotBase64(qp.getQuPath())
    elif snapshot_type == 'viewer':
        image = qp.snapshotBase64(qp.getCurrentViewer())
    else:
        raise ValueError(f'Unknown snapshot_type {snapshot_type}')
        
    return utils.base64_to_image(image, True)


def get_objects(
    image_data: JavaObject = None,
    gateway: JavaGateway = None,
    object_type: ObjectType = None,
    converter: str = None
) -> Union[list[geojson.Feature], list[ImageFeature], list[JavaObject]]:
    """
    Get the objects (e.g. detections, annotations) of the current or specified image in QuPath.

    :param image_data: the image_data to retrieve objects from. Can be None to use the current ImageData opened in QuPath
    :param gateway: the gateway to use. Can be None to use the default gateway
    :param object_type: the type of object to get. Can be None to get all objects except the root
    :param converter: can be 'geojson' to get an extended GeoJSON feature represented as a QuBaLab 'ImageObject',
                      or 'simple_feature' to get a regular GeoJSON 'Feature'. By default, the Py4J representation of
                      the QuPath Java object is returned.
    :return: the list of objects of the specified image. The type of the objects depends on the converted parameter
    """
    gateway = get_default_gateway() if gateway is None else gateway
    image_data = get_current_image_data(gateway) if image_data is None else image_data
    hierarchy = image_data.getHierarchy()

    match object_type:
        case None:
            path_objects = hierarchy.getAllObjects(False)
        case ObjectType.ANNOTATION:
            path_objects = hierarchy.getAnnotationObjects()
        case ObjectType.DETECTION:
            path_objects = hierarchy.getDetectionObjects()
        case ObjectType.TILE:
            path_objects = hierarchy.getTileObjects()
        case ObjectType.CELL:
            path_objects = hierarchy.getCellObjects()
        case ObjectType.TMA_CORE:
            tma_grid = hierarchy.getTMAGrid()
            path_objects = [] if tma_grid is None else tma_grid.getTMACoreList()

    if converter == 'geojson' or converter == 'simple_feature':
        # Use toFeatureCollections for performance and to avoid string length troubles
        features = []
        for feature_collection in gateway.entry_point.toFeatureCollections(path_objects, 1000):
            features.extend(geojson_features_from_string(feature_collection, parse_constant=None))

        if converter == 'simple_feature':
            return features
        else:
            return [ImageFeature.create_from_feature(f) for f in features]
    else:
        return path_objects


def count_objects(image_data: JavaObject = None, gateway: JavaGateway = None, object_type: ObjectType = None,) -> int:
    """
    Get a count of all objects in the current or specified image.

    Since requesting the objects can be slow, this can be used to check if a reasonable
    number of objects is available.

    :param image_data: the image_data to retrieve objects from. Can be None to use the current ImageData opened in QuPath
    :param gateway: the gateway to use. Can be None to use the default gateway
    :param object_type: the type of object to get. Can be None to get all objects except the root
    :return: the number of such objects in the current or specified image
    """
    return len(get_objects(image_data, gateway, object_type))


def add_objects(features: Union[list[geojson.Feature], geojson.Feature], image_data: JavaObject = None, gateway: JavaGateway = None):
    """
    Add the provided features to the current or provided ImageData.

    :param features: the features to add. Can be a list or a single Feature
    :param image_data: the image_data to add features to. Can be None to use the current ImageData opened in QuPath
    :param gateway: the gateway to use. Can be None to use the default gateway
    """
    gateway = get_default_gateway() if gateway is None else gateway
    image_data = get_current_image_data(gateway) if image_data is None else image_data

    if isinstance(features, geojson.Feature):
        features = list(features)
    
    features_json = geojson.dumps(features, allow_nan=True)
    image_data.getHierarchy().addObjects(gateway.entry_point.toPathObjects(features_json))


def delete_objects(image_data: JavaObject = None, gateway: JavaGateway = None, object_type: ObjectType = None):
    """
    Delete all specified objects (e.g. annotations, detections) from the current or provided ImageData.

    :param image_data: the image_data to remove the objects from. Can be None to use the current ImageData opened in QuPath
    :param gateway: the gateway to use. Can be None to use the default gateway
    :param object_type: the type of object to remove. Can be None to remove all objects
    """
    gateway = get_default_gateway() if gateway is None else gateway
    image_data = get_current_image_data(gateway) if image_data is None else image_data

    if image_data is not None:
        image_data.getHierarchy().removeObjects(get_objects(image_data, gateway, object_type), True)
        image_data.getHierarchy().getSelectionModel().clearSelection()


def refresh_qupath(gateway: JavaGateway = None):
    """
    Update the current QuPath interface.
    
    This is sometimes needed to update the QuPath window when some changes are
    made to hierarchy.

    :param gateway: the gateway to use. Can be None to use the default gateway
    """
    gateway = get_default_gateway() if gateway is None else gateway
    gateway.entry_point.fireHierarchyUpdate()
