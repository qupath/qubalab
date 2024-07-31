import geojson
import math
import numpy as np
from qubalab.objects.image_feature import ImageFeature
from qubalab.objects.classification import Classification
from qubalab.objects.object_type import ObjectType


def test_geometry():
    expected_geometry = geojson.Point((-115.81, 37.24))
    image_feature = ImageFeature(expected_geometry)

    geometry = image_feature.geometry

    assert geometry == expected_geometry


def test_json_serializable_with_geometry():
    geometry = geojson.Point((-115.81, 37.24))

    image_feature = ImageFeature(geometry)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_id():
    expected_id = 23
    image_feature = ImageFeature(None, id=expected_id)

    id = image_feature.id

    assert id == expected_id


def test_json_serializable_with_id():
    id = 23

    image_feature = ImageFeature(None, id=id)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_classification():
    expected_classification = Classification("name", (1, 1, 1))
    image_feature = ImageFeature(None, classification=expected_classification)

    classification = image_feature.classification

    assert classification == expected_classification


def test_json_serializable_with_classification():
    classification = Classification("name", (1, 1, 1))

    image_feature = ImageFeature(None, classification=classification)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_name():
    expected_name = "name"
    image_feature = ImageFeature(None, name=expected_name)

    name = image_feature.name

    assert name == expected_name


def test_json_serializable_with_name():
    name = "name"

    image_feature = ImageFeature(None, name=name)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_measurements():
    inital_measurements = {
        "some_value": 0.324,
        "nan_value": float('nan')
    }
    expected_measurements = {k: v for k, v in inital_measurements.items() if not math.isnan(v)}    # NaN values are skipped
    image_feature = ImageFeature(None, measurements=inital_measurements)

    measurements = image_feature.measurements

    assert measurements == expected_measurements


def test_json_serializable_with_measurements():
    measurements = {
        "some_value": 0.324,
        "nan_value": float('nan')
    }

    image_feature = ImageFeature(None, measurements=measurements)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_object_type():
    expected_object_type = ObjectType.DETECTION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    object_type = image_feature.object_type

    assert object_type == expected_object_type


def test_json_serializable_with_object_type():
    object_type = ObjectType.DETECTION

    image_feature = ImageFeature(None, object_type=object_type)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_is_detection():
    expected_object_type = ObjectType.DETECTION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_detection = image_feature.is_detection

    assert is_detection


def test_is_not_detection():
    expected_object_type = ObjectType.ANNOTATION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_detection = image_feature.is_detection

    assert not(is_detection)


def test_is_cell():
    expected_object_type = ObjectType.CELL
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_cell = image_feature.is_cell

    assert is_cell


def test_is_not_cell():
    expected_object_type = ObjectType.DETECTION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_cell = image_feature.is_cell

    assert not(is_cell)


def test_is_tile():
    expected_object_type = ObjectType.TILE
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_tile = image_feature.is_tile

    assert is_tile


def test_is_not_tile():
    expected_object_type = ObjectType.DETECTION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_tile = image_feature.is_tile

    assert not(is_tile)


def test_is_annotation():
    expected_object_type = ObjectType.ANNOTATION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_annotation = image_feature.is_annotation

    assert is_annotation


def test_is_not_annotation():
    expected_object_type = ObjectType.DETECTION
    image_feature = ImageFeature(None, object_type=expected_object_type)

    is_annotation = image_feature.is_annotation

    assert not(is_annotation)


def test_color():
    expected_color = (4, 5, 6)
    image_feature = ImageFeature(None, color=expected_color)

    color = image_feature.color

    assert color == expected_color


def test_json_serializable_with_color():
    color = (4, 5, 6)

    image_feature = ImageFeature(None, color=color)

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_nucleus_geometry():
    expected_nucleus_geometry = geojson.Point((-115.81, 37.24))
    image_feature = ImageFeature(None, extra_geometries={
        "nucleus": expected_nucleus_geometry
    })

    nucleus_geometry = image_feature.nucleus_geometry

    assert nucleus_geometry == expected_nucleus_geometry


def test_json_serializable_with_nucleus_geometry():
    nucleus_geometry = geojson.Point((-115.81, 37.24))

    image_feature = ImageFeature(None, extra_geometries={
        "nucleus": nucleus_geometry
    })

    geojson.dumps(image_feature)     # will throw an exception if not serializable


def test_geometry_when_created_from_feature():
    expected_geometry = geojson.Point((-115.81, 37.24))
    feature = geojson.Feature(geometry=expected_geometry)
    image_feature = ImageFeature.create_from_feature(feature)

    geometry = image_feature.geometry

    assert geometry == expected_geometry


def test_id_when_created_from_feature():
    expected_id = 23
    feature = geojson.Feature(id=expected_id)
    image_feature = ImageFeature.create_from_feature(feature)

    id = image_feature.id

    assert id == expected_id


def test_classification_when_created_from_feature():
    expected_classification = Classification("name", (1, 1, 1))
    feature = geojson.Feature(properties={
        "classification": expected_classification
    })
    image_feature = ImageFeature.create_from_feature(feature)

    classification = image_feature.classification

    assert classification == expected_classification


def test_name_when_created_from_feature():
    expected_name = "name"
    feature = geojson.Feature(properties={
        "name": expected_name
    })
    image_feature = ImageFeature.create_from_feature(feature)

    name = image_feature.name

    assert name == expected_name


def test_measurements_when_created_from_feature():
    inital_measurements = {
        "some_value": 0.324,
        "nan_value": float('nan')
    }
    expected_measurements = {k: v for k, v in inital_measurements.items() if not math.isnan(v)}    # NaN values are skipped
    feature = geojson.Feature(properties={
        "measurements": inital_measurements
    })
    image_feature = ImageFeature.create_from_feature(feature)

    measurements = image_feature.measurements

    assert measurements == expected_measurements


def test_object_type_when_created_from_feature():
    expected_object_type = ObjectType.CELL
    feature = geojson.Feature(properties={
        "object_type": expected_object_type.name
    })
    image_feature = ImageFeature.create_from_feature(feature)

    object_type = image_feature.object_type

    assert object_type == expected_object_type


def test_color_when_created_from_feature():
    expected_color = (4, 5, 6)
    feature = geojson.Feature(properties={
        "color": expected_color
    })
    image_feature = ImageFeature.create_from_feature(feature)

    color = image_feature.color

    assert color == expected_color


def test_nucleus_geometry_when_created_from_feature():
    expected_nucleus_geometry = geojson.Point((-115.81, 37.24))
    feature = geojson.Feature(properties={
        "nucleusGeometry": expected_nucleus_geometry
    })
    image_feature = ImageFeature.create_from_feature(feature)

    nucleus_geometry = image_feature.nucleus_geometry

    assert nucleus_geometry == expected_nucleus_geometry


def test_number_of_features_when_created_from_label_image_without_scale():
    label_image = np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],],
        dtype=np.uint8
    )
    expected_number_of_features = 3

    features = ImageFeature.create_from_label_image(label_image)

    assert len(features) == expected_number_of_features


def test_number_of_features_when_created_from_label_image_with_scale():
    scale = 2
    label_image = np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],],
        dtype=np.uint8
    )
    expected_number_of_features = 3

    features = ImageFeature.create_from_label_image(label_image, scale=scale)

    assert len(features) == expected_number_of_features


def test_object_type_when_created_from_label_image():
    expected_object_type = ObjectType.CELL
    label_image = np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],],
        dtype=np.uint8
    )

    features = ImageFeature.create_from_label_image(label_image, object_type=expected_object_type)

    assert all(feature.object_type == expected_object_type for feature in features)


def test_measurement_when_created_from_label_image():
    expected_measurements = [{'Label': float(label)} for label in [1, 2, 3]]
    label_image = np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],],
        dtype=np.uint8
    )

    features = ImageFeature.create_from_label_image(label_image, include_labels=True)

    assert all(feature.measurements in expected_measurements for feature in features)


def test_classification_when_created_from_label_image_and_classification_name_provided():
    expected_classification_name = "name"
    label_image = np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],],
        dtype=np.uint8
    )

    features = ImageFeature.create_from_label_image(label_image, classification_names=expected_classification_name)

    assert all(feature.classification.name == expected_classification_name for feature in features)


def test_classification_when_created_from_label_image_and_classification_dict_provided():
    classification_dict = {
        # no classification for label 1
        2: "name2",
        3: "name3"
    }
    expected_classification_names = classification_dict.values()
    label_image = np.array(
        [[0, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 0, 0],],
        dtype=np.uint8
    )

    features = ImageFeature.create_from_label_image(label_image, classification_names=classification_dict)

    assert all(feature.classification is None or feature.classification.name in expected_classification_names for feature in features)


def test_number_of_features_when_created_from_binary_image_without_scale():
    binary_image = np.array(
        [[False, True,  True,  False, False],
         [False, True,  True,  False, False],
         [False, False, False, False, False],
         [False, False, False, True,  False],
         [False, False, False, True,  False],
         [False, False, False, False, False],
         [False, False, True,  True,  False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, True,  False, False, False],],
        dtype=bool
    )
    expected_number_of_features = 1

    features = ImageFeature.create_from_label_image(binary_image)

    assert len(features) == expected_number_of_features


def test_number_of_features_when_created_from_binary_image_with_scale():
    scale = 2
    binary_image = np.array(
        [[False, True,  True,  False, False],
         [False, True,  True,  False, False],
         [False, False, False, False, False],
         [False, False, False, True,  False],
         [False, False, False, True,  False],
         [False, False, False, False, False],
         [False, False, True,  True,  False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, True,  False, False, False],],
        dtype=bool
    )
    expected_number_of_features = 1

    features = ImageFeature.create_from_label_image(binary_image, scale=scale)

    assert len(features) == expected_number_of_features


def test_object_type_when_created_from_binary_image():
    expected_object_type = ObjectType.TILE
    binary_image = np.array(
        [[False, True,  True,  False, False],
         [False, True,  True,  False, False],
         [False, False, False, False, False],
         [False, False, False, True,  False],
         [False, False, False, True,  False],
         [False, False, False, False, False],
         [False, False, True,  True,  False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, True,  False, False, False],],
        dtype=bool
    )

    features = ImageFeature.create_from_label_image(binary_image, object_type=expected_object_type)

    assert all(feature.object_type == expected_object_type for feature in features)


def test_measurement_when_created_from_binary_image():
    expected_measurement = {'Label': 1.0}
    binary_image = np.array(
        [[False, True,  True,  False, False],
         [False, True,  True,  False, False],
         [False, False, False, False, False],
         [False, False, False, True,  False],
         [False, False, False, True,  False],
         [False, False, False, False, False],
         [False, False, True,  True,  False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, True,  False, False, False],],
        dtype=bool
    )

    features = ImageFeature.create_from_label_image(binary_image, include_labels=True)

    assert all(feature.measurements == expected_measurement for feature in features)


def test_classification_when_created_from_binary_image_and_classification_name_provided():
    expected_classification_name = "name"
    binary_image = np.array(
        [[False, True,  True,  False, False],
         [False, True,  True,  False, False],
         [False, False, False, False, False],
         [False, False, False, True,  False],
         [False, False, False, True,  False],
         [False, False, False, False, False],
         [False, False, True,  True,  False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, True,  False, False, False],],
        dtype=bool
    )

    features = ImageFeature.create_from_label_image(binary_image, classification_names=expected_classification_name)

    assert all(feature.classification.name == expected_classification_name for feature in features)


def test_classification_when_created_from_binary_image_and_classification_dict_provided():
    classification_dict = {
        1: "name1"
    }
    expected_classification_names = classification_dict.values()
    binary_image = np.array(
        [[False, True,  True,  False, False],
         [False, True,  True,  False, False],
         [False, False, False, False, False],
         [False, False, False, True,  False],
         [False, False, False, True,  False],
         [False, False, False, False, False],
         [False, False, True,  True,  False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, True,  False, False, False],],
        dtype=bool
    )

    features = ImageFeature.create_from_label_image(binary_image, classification_names=classification_dict)

    assert all(feature.classification.name in expected_classification_names for feature in features)


def test_classification_when_set_after_creation():
    expected_classification = Classification("name", (1, 1, 1))
    image_feature = ImageFeature(None)
    image_feature.classification = {
        "name": expected_classification.name,
        "color": expected_classification.color
    }

    classification = image_feature.classification

    assert classification == expected_classification


def test_name_when_set_after_creation():
    expected_name = "name"
    image_feature = ImageFeature(None)
    image_feature.name = expected_name

    name = image_feature.name

    assert name == expected_name


def test_measurements_when_set_after_creation():
    inital_measurements = {
        "some_value": 0.324,
        "nan_value": float('nan')
    }
    expected_measurements = {k: v for k, v in inital_measurements.items() if not math.isnan(v)}    # NaN values are skipped
    image_feature = ImageFeature(None)
    image_feature.measurements = inital_measurements

    measurements = image_feature.measurements

    assert measurements == expected_measurements


def test_object_type_when_set_after_creation():
    expected_object_type = ObjectType.CELL
    image_feature = ImageFeature(None)
    image_feature.object_type = expected_object_type

    object_type = image_feature.object_type

    assert object_type == expected_object_type


def test_object_type_name_when_set_after_creation():
    expected_object_type = ObjectType.TILE
    image_feature = ImageFeature(None)
    image_feature.object_type = expected_object_type.name

    object_type = image_feature.object_type

    assert object_type == expected_object_type


def test_color_when_set_after_creation():
    expected_color = (4, 5, 6)
    image_feature = ImageFeature(None)
    image_feature.color = expected_color

    color = image_feature.color

    assert color == expected_color


def test_nucleus_geometry_when_set_after_creation():
    expected_nucleus_geometry = geojson.Point((-115.81, 37.24))
    image_feature = ImageFeature(None)
    image_feature.nucleus_geometry = expected_nucleus_geometry

    nucleus_geometry = image_feature.nucleus_geometry

    assert nucleus_geometry == expected_nucleus_geometry
