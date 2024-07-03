import geojson
import uuid
import math
from typing import Union, Any
import geojson.geometry
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
        classification: Classification = None,
        name: str = None,
        measurements: dict[str, float] = None,
        object_type: ObjectType = ObjectType.ANNOTATION,
        color: tuple[int, int, int] = None,
        extra_geometries: dict[str, geojson.geometry.Geometry] = None,
        id: Union[str, int, uuid.UUID] = None,
        extra_properties: dict[str, Any] = None
    ):
        """
        Create the image feature.

        Except from the geometry and id parameters, all parameters of this
        constructor will be added to the list of properties of this feature
        (if provided).

        :param geometry: the geometry of the feature
        :param classification: the classification of this feature
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
            props['classification'] = classification
        if name is not None:
            props['name'] = name
        if measurements is not None:
            props['measurements'] = ImageFeature._remove_NaN_values_from_measurements(measurements)
        if object_type is not None:
            props['object_type'] = object_type
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
    def create_from_feature(cls, feature: geojson.Feature):
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

        args = dict(
            geometry=geometry,
            id=cls._find_property(feature, 'id'),
            classification=cls._find_property(feature, 'classification'),
            name=cls._find_property(feature, 'name'),
            color=cls._find_property(feature, 'color'),
            measurements=cls._find_property(feature, 'measurements'),
            object_type=cls._find_property(feature, 'object_type'),
        )

        nucleus_geometry = cls._find_property(feature, 'nucleusGeometry')
        if nucleus_geometry is not None:
            if plane is not None:
                nucleus_geometry = add_plane_to_geometry(nucleus_geometry, z=getattr(plane, 'z', None), t=getattr(plane, 't', None))
            args['extra_geometries'] = dict(nucleus=nucleus_geometry)

        args['extra_properties'] = {k: v for k, v in feature['properties'].items() if k not in args and v is not None}
        return cls(**args)

    @property
    def classification(self) -> Classification:
        """
        The classification of this feature (or None if not defined).
        """
        return self.properties.get('classification')

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
        The QuPath object type (e.g. detection, annotation) this feature represents.
        """
        return self.properties['object_type']

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
            self.properties['classification'] = value
        elif name == 'name':
            self.properties['name'] = value
        elif name == 'measurements':
            self.properties['measurements'] = ImageFeature._remove_NaN_values_from_measurements(value)
        elif name == 'object_type':
            self.properties['object_type'] = value
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