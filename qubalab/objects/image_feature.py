from __future__ import annotations
import geojson
import uuid
import math
import numpy as np
import geojson.geometry
from typing import Union, Any
import rasterio
import rasterio.features
import shapely
from .object_type import ObjectType
from .classification import Classification
from .geometry import add_plane_to_geometry


_NUCLEUS_GEOMETRY_KEY = 'nucleus'


class ImageFeature(geojson.Feature):
    """
    GeoJSON Feature with additional properties for image objects.

    The added properties are:

    - A classification (defined by a name and a color).

    - A name.

    - A list of measurements.

    - A type of QuPath object (e.g. detection, annotation).

    - A color.

    - Additional geometries.
    
    - And any other property.
    """

    def __init__(
        self,
        geometry: geojson.geometry.Geometry,
        classification: Union[Classification, dict] = None,
        name: str = None,
        measurements: dict[str, float] = None,
        object_type: ObjectType = ObjectType.ANNOTATION,
        color: tuple[int, int, int] = None,
        extra_geometries: dict[str, geojson.geometry.Geometry] = None,
        id: Union[str, int, uuid.UUID] = None,
        extra_properties: dict[str, Any] = None
    ):
        """
        Except from the geometry and id parameters, all parameters of this
        constructor will be added to the list of properties of this feature
        (if provided).
        
        :param geometry: the geometry of the feature
        :param classification: the classification of this feature, or a dictionnary with the
                               'name' and 'color' properties defining respectively a string
                               and a 3-long int tuple with values between 0 and 255
        :param name: the name of this feature
        :param measurements: a dictionnary containing measurements. Measurements
                             with NaN values will not be added
        :param object_type: the type of QuPath object this feature represents
        :param color: the color of this feature
        :param extra_geometries: a dictionnary containing additional geometries
                                 that represent this feature
        :param id: the ID of the feature. If not provided, an UUID will be generated
        :param extra_properties: a dictionnary of additional properties to add
        """
        props = {}
        if classification is not None:
            if isinstance(classification, Classification):
                props['classification'] = {
                    "name": classification.name,
                    "color": classification.color
                }
            else:
                props['classification'] = {
                    "name": classification.get('name'),
                    "color": classification.get('color')
                }
        if name is not None:
            props['name'] = name
        if measurements is not None:
            props['measurements'] = ImageFeature._remove_NaN_values_from_measurements(measurements)
        if object_type is not None:
            props['object_type'] = object_type.name
        if color is not None:
            props['color'] = color
        if extra_geometries is not None:
            props['extra_geometries'] = {k: add_plane_to_geometry(v) for k, v in extra_geometries.items()}
        if extra_properties is not None:
            props.update(extra_properties)
        
        super().__init__(
            geometry=add_plane_to_geometry(geometry),
            properties=props,
            id=ImageFeature._to_id_string(id)
        )
        self['type'] = 'Feature'
        
    
    @classmethod
    def create_from_feature(cls, feature: geojson.Feature) -> ImageFeature:
        """
        Create an ImageFeature from a GeoJSON feature.

        The ImageFeature properties will be searched in the provided
        feature and in the properties of the provided feature.
        
        :param feature: the feature to convert to an ImageFeature
        :return: an ImageFeature corresponding to the provided feature
        """
        geometry = cls._find_property(feature, 'geometry')
        plane = cls._find_property(feature, 'plane')
        if plane is not None:
            geometry = add_plane_to_geometry(geometry, z=getattr(plane, 'z', None), t=getattr(plane, 't', None))

        object_type_property = cls._find_property(feature, 'object_type')
        if object_type_property is None:
            object_type_property = cls._find_property(feature, 'objectType')
        object_type = next((o for o in ObjectType if o.name.lower() == str(object_type_property).lower()), None)

        args = dict(
            geometry=geometry,
            id=cls._find_property(feature, 'id'),
            classification=cls._find_property(feature, 'classification'),
            name=cls._find_property(feature, 'name'),
            color=cls._find_property(feature, 'color'),
            measurements=cls._find_property(feature, 'measurements'),
            object_type=object_type,
        )

        nucleus_geometry = cls._find_property(feature, 'nucleusGeometry')
        if nucleus_geometry is not None:
            if plane is not None:
                nucleus_geometry = add_plane_to_geometry(nucleus_geometry, z=getattr(plane, 'z', None), t=getattr(plane, 't', None))
            args['extra_geometries'] = dict(nucleus=nucleus_geometry)

        args['extra_properties'] = {k: v for k, v in feature['properties'].items() if k not in args and v is not None}
        return cls(**args)
    
    @classmethod
    def create_from_label_image(
        cls,
        input_image: np.ndarray,
        object_type: ObjectType = ObjectType.ANNOTATION,
        connectivity: int = 4,
        scale: float = 1.0,
        include_labels = False,
        classification_names: Union[str, dict[int, str]] = None
    ) -> list[ImageFeature]:
        """
        Create a list of ImageFeatures from a binary or labeled image.

        The created geometries will be polygons, even when representing points or line.

        :param input_image: a 2-dimensionnal binary (with a boolean type) or labeled
                            (with a uint8 type) image containing the features to create.
                            If a binary image is given, all True pixel values will be
                            considered as potential features. If a labeled image is given,
                            all pixel values greater than 0 will be considered as potential features
        :param object_type: the type of object to create
        :param connectivity: the pixel connectivity for grouping pixels into features (4 or 8)
        :param scale: a scale value to apply to the shapes
        :param include_labels: whether to include a 'Label' measurement in the created features
        :param classification_names: if str, the name of the classification to apply to all features.
                                     if dict, a dictionnary mapping a label to a classification name
        :return: a list of image features representing polygons present in the input image
        """
        features = []

        if input_image.dtype == bool:
            mask = input_image
            input_image = input_image.astype(np.uint8)
        else:
            mask = input_image > 0

        transform = rasterio.transform.Affine.scale(scale)

        existing_features = {}
        for geometry, label in rasterio.features.shapes(input_image, mask=mask, connectivity=connectivity, transform=transform):
            if label in existing_features:
                existing_features[label]['geometry'] = shapely.geometry.shape(geometry).union(
                    shapely.geometry.shape(existing_features[label]['geometry'])
                )
            else:
                if isinstance(classification_names, str):
                    classification_name = classification_names
                elif isinstance(classification_names, dict) and int(label) in classification_names:
                    classification_name = classification_names[int(label)]
                else:
                    classification_name = None

                feature = cls(
                    geometry=geometry,
                    classification=Classification.get_cached_classification(classification_name),
                    measurements={'Label': float(label)} if include_labels else None,
                    object_type=object_type
                )

                existing_features[label] = feature
                features.append(feature)

        # Ensure we have GeoJSON-compatible geometries
        for feature in features:
            feature['geometry'] = geojson.mapping.to_mapping(feature['geometry'])

        return features

    @property
    def classification(self) -> Classification:
        """
        The classification of this feature (or None if not defined).
        """
        if "classification" in self.properties:
            return Classification(self.properties['classification'].get('name'), self.properties['classification'].get('color'))
        else:
            return None

    @property
    def name(self) -> str:
        """
        The name of this feature (or None if not defined).
        """
        return self.properties.get('name')

    @property
    def measurements(self) -> dict[str, float]:
        """
        The measurements of this feature.
        """
        measurements = self.properties.get('measurements')
        if measurements is None:
            measurements = {}
            self.properties['measurements'] = measurements
        return measurements

    @property
    def object_type(self) -> ObjectType:
        """
        The QuPath object type (e.g. detection, annotation) this feature represents
        or None if the object type doesn't exist or is not recognised.
        """
        return next((o for o in ObjectType if o.name.lower() == str(self.properties['object_type']).lower()), None)

    @property
    def is_detection(self) -> bool:
        """
        Wether the QuPath object type (e.g. detection, annotation) represented by this
        feature is a detection, cell, or tile.
        """
        return self.object_type in [ObjectType.DETECTION, ObjectType.CELL, ObjectType.TILE]

    @property
    def is_cell(self) -> bool:
        """
        Wether the QuPath object type (e.g. detection, annotation) represented by this
        feature is a cell.
        """
        return self.object_type == ObjectType.CELL

    @property
    def is_tile(self) -> bool:
        """
        Wether the QuPath object type (e.g. detection, annotation) represented by this
        feature is a tile.
        """
        return self.object_type == ObjectType.TILE

    @property
    def is_annotation(self) -> bool:
        """
        Wether the QuPath object type (e.g. detection, annotation) represented by this
        feature is an annotation.
        """
        return self.object_type == ObjectType.ANNOTATION

    @property
    def color(self) -> tuple[int, int, int]:
        """
        The color of this feature (or None if not defined).
        """
        return self.properties.get('color')

    @property
    def nucleus_geometry(self) -> geojson.geometry.Geometry:
        """
        The nucleus geometry of this feature (or None if not defined).
        It can be defined when passed as an extra_geometry with the 'nucleus'
        key when creating an ImageFeature, by defining the 'nucleus_geometry'
        property of an ImageFeature, or when passed as a 'nucleusGeometry'
        property when creating an ImageFeature from a GeoJSON feature.
        """
        extra = self.properties.get('extra_geometries')
        if extra is not None:
            return extra.get(_NUCLEUS_GEOMETRY_KEY)
        return None

    def __setattr__(self, name, value):
        if name == 'classification':
            if isinstance(value, Classification):
                self.properties['classification'] = {
                    "name": value.name,
                    "color": value.color
                }
            else:
                self.properties['classification'] = value
        elif name == 'name':
            self.properties['name'] = value
        elif name == 'measurements':
            self.properties['measurements'] = ImageFeature._remove_NaN_values_from_measurements(value)
        elif name == 'object_type':
            if isinstance(value, str):
                self.properties['object_type'] = value
            elif isinstance(value, ObjectType):
                self.properties['object_type'] = value.name
        elif name == 'color':
            if len(value) != 3:
                raise ValueError('Color must be a tuple of length 3')
            rgb = tuple(ImageFeature._validate_rgb_value(v) for v in value)
            self.properties['color'] = rgb
        elif name == 'nucleus_geometry':
            if 'extra_geometries' not in self.properties:
                self.properties['extra_geometries'] = {}
            self.properties['extra_geometries'][_NUCLEUS_GEOMETRY_KEY] = add_plane_to_geometry(value)
        else:
            super().__setattr__(name, value)
    
    @staticmethod
    def _remove_NaN_values_from_measurements(measurements: dict[str, float]) -> dict[str, float]:
        return {
            k: float(v) for k, v in measurements.items()
            if isinstance(k, str) and isinstance(v, (int, float)) and not math.isnan(v)
        }
    
    @staticmethod
    def _to_id_string(object_id: Union[int, str, uuid.UUID]) -> str:
        if object_id is None:
            return str(uuid.uuid4())
        elif isinstance(object_id, str) or isinstance(object_id, int):
            return object_id
        else:
            return str(object_id)
    
    @staticmethod
    def _find_property(feature: geojson.Feature, property_name: str):
        if property_name in feature:
            return feature[property_name]
        elif 'properties' in feature and property_name in feature['properties']:
            return feature['properties'][property_name]
        else:
            return None

    @staticmethod
    def _validate_rgb_value(value: Union[int, float]) -> int:
        if isinstance(value, float):
            value = int(math.round(value * 255))
        if isinstance(value, int):
            if value >= 0 and value <= 255:
                return value
        raise ValueError('Color value must be an int between 0 and 255, or a float between 0 and 1')
