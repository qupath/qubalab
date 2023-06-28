from typing import Tuple
import uuid

from . import types as types
from .geometries import to_geometry
from geojson import Feature
from geojson.geometry import Geometry
from typing import Iterable, Union, Dict, Any
import random


_cached_classifications = {}
_NUCLEUS_GEOMETRY_KEY = 'nucleus'


class Classification(object):

    def __init__(self, name: str, color: Tuple[int, int, int], parent: 'Classification' = None):
        self._parent = parent
        self._name = name
        if color is None:
            self._color = [random.randint(0, 255) for ii in range(3)]
        else:
            self._color = color

    @property
    def name(self) -> str:
        return self._name

    @property
    def color(self) -> Tuple[int, int, int]:
        return self._color

    def __str__(self):
        if self._parent is None:
            return self._name
        return self._parent.__str__() + ': ' + self._name


class ImageObject(Feature):
    """
    GeoJSON Feature with additional properties for image objects.
    """

    def __init__(self,
                 geometry,
                 classification: Classification = None,
                 name: str = None,
                 measurements: Dict[str, float] = None,
                 object_type: str = types.ANNOTATION,
                 color: Tuple[int, int, int] = None,
                 extra_geometries: Dict[str, Any] = None,
                 id: Union[str, int, uuid.UUID] = None,
                 extra_properties: Dict[str, Any] = None):

        object_id = self._to_id_string(id)
        props = {}
        if classification is not None:
            props['classification'] = classification
        if name is not None:
            props['name'] = name
        if measurements is not None:
            # Can't store NaN properly in JSON, so try to remove
            import math
            props['measurements'] = {k: float(v) for k, v in measurements.items()
                                     if isinstance(k, str) and isinstance(v, (int, float)) and not math.isnan(v)}
        if object_type is not None:
            props['object_type'] = object_type
        if color is not None:
            props['color'] = color
        if extra_geometries is not None:
            props['extra_geometries'] = {k: to_geometry(v) for k, v in extra_geometries.items()}

        if extra_properties is not None:
            props.update(extra_properties)
        super().__init__(geometry=to_geometry(geometry), properties=props, id=object_id)
        self.type = 'Feature'


    @classmethod
    def _to_id_string(cls, object_id: Union[int, str, uuid.UUID], create_if_missing: bool = True) -> str:
        if object_id is None:
            if create_if_missing:
                return str(uuid.uuid4())
            else:
                return None
        if isinstance(object_id, str) or isinstance(object_id, int):
            return object_id
        else:
            return str(object_id)

    @property
    def object_type(self) -> str:
        return self.properties['object_type']

    @object_type.setter
    def object_type(self, object_type: str) -> str:
        self.properties['object_type'] = object_type

    @property
    def is_detection(self) -> bool:
        return self.object_type in [types.DETECTION, types.CELL, types.TILE]

    @property
    def is_cell(self) -> bool:
        return self.object_type == types.CELL

    @property
    def is_tile(self) -> bool:
        return self.object_type == types.TILE

    @property
    def is_annotation(self) -> bool:
        return self.object_type == types.ANNOTATION

    @property
    def color(self) -> Tuple[int, int, int]:
        # TODO: Consider consistently changing color tuples to float (and supporting alpha)
        return self.properties.get('color')

    @property
    def measurements(self) -> Dict[str, float]:
        measurements = self.properties.get('measurements')
        if measurements is None:
            measurements = {}
            self.properties['measurements'] = measurements
        return measurements

    @property
    def nucleus_geometry(self) -> Geometry:
        extra = self.properties.get('extra_geometries')
        if extra is not None:
            return extra.get(_NUCLEUS_GEOMETRY_KEY)
        return None

    @nucleus_geometry.setter
    def nucleus_geometry(self, geometry: Geometry):
        self.properties['extra_geometries'][_NUCLEUS_GEOMETRY_KEY] = to_geometry(geometry)

    @property
    def name(self) -> str:
        return self.properties

    @name.setter
    def name(self, name: str):
        self.properties['name'] = name

    @property
    def classification(self) -> Classification:
        return self.properties.get('classification')

    @classification.setter
    def classification(self, classification: Classification):
        self.properties['classification'] = classification

    # @property
    # def properties(self) -> Dict[str, Any]:
    #     return self['properties']

    # @property
    # def parent(self) -> 'ImageObject':
    #     return self._parent
    #
    # @parent.setter
    # def parent(self, parent: 'ImageObject'):
    #     self._parent = parent


def create_tile(geometry, classification: Classification = None, name: str = None,
                measurements: Dict[str, float] = None, object_id: uuid.UUID = None):
    return ImageObject(geometry=geometry, classification=classification, name=name, measurements=measurements,
                       object_type=types.TILE, object_id=object_id)


def create_detection(geometry, **kwargs):
    return ImageObject(geometry=geometry, object_type=types.DETECTION, **kwargs)


def create_cell(geometry, nucleus_geometry, **kwargs):
    # We are assuming that no other extra geometries are provided via kwargs
    if nucleus_geometry:
        extra_geometries = {_NUCLEUS_GEOMETRY_KEY: nucleus_geometry}
    else:
        extra_geometries = None
    return ImageObject(geometry=geometry, extra_geometries=extra_geometries, **kwargs)


def create_annotation(geometry, **kwargs):
    return ImageObject(geometry=geometry, object_type=types.ANNOTATION, **kwargs)


def get_classification(name: str, color: Tuple[int, int, int] = None):
    if name is None:
        return None
    classification = _cached_classifications.get(name)
    if classification is None:
        classification = Classification(name=name, color=color)
        _cached_classifications[classification.__str__()] = classification
    return classification


def _parse_color(color: Union[Tuple[int, int, int], Tuple[int, int, int], int]) -> Tuple[int, int, int]:
    """
    Parse a color from a tuple or (packed RGB) integer.
    """
    if isinstance(color, int):
        r = (color >> 16) & 255
        g = (color >> 8) & 255
        b = color & 255
        return (r, g, b)
    elif isinstance(color, Iterable):
        color = tuple(color) # TODO: Consider checking type and/or supporting float?
        if len(color) == 3 or len(color) == 4:
            return color
        else:
            raise ValueError(f'Color tuples should give RGB or RGBA values (length 3 or 4)')
