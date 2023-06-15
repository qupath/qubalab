import io
import warnings
from typing import Tuple
import uuid
from .rois import ROI, create_roi
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from typing import List, Iterable, Union, Dict
import random
import json
from enum import Enum

from geojson import Feature


class ImageObjectType(Enum):
    ROOT = 0,
    ANNOTATION = 1,
    DETECTION = 2,
    TILE = 3,
    CELL = 4,
    TMA = 5


_cached_classifications = {}


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


class ImageObject(object):

    def __init__(self, roi: Union[ROI, BaseGeometry], classification: Classification = None, name: str = None,
                 measurements: Dict[str, float] = None, object_type: ImageObjectType = ImageObjectType.ANNOTATION,
                 nucleus_roi: Union[ROI, BaseGeometry] = None, object_id: uuid.UUID = None):
        # self._parent = None
        # self._children = []
        self._roi = roi if isinstance(roi, ROI) else create_roi(roi)
        if nucleus_roi is not None:
            if object_type != ImageObjectType.CELL:
                raise ValueError(f'Cannot set a nucleus ROI for objects of type {object_type}')
            self._nucleus_roi = nucleus_roi if isinstance(nucleus_roi, ROI) else create_roi(nucleus_roi)
        else:
            self._nucleus_roi = None
        self._classification = classification
        self._name = name
        self._measurements = measurements
        self._object_type = object_type
        self._object_id = object_id if object_id is not None else uuid.uuid4()

    @property
    def __geo_interface__(self):
        # See https://gist.github.com/sgillies/2217756
        # TODO: Consider if this is useful - and if properties should be stored directly in dict
        props = dict(
                        classification=self.classification,
                        name=self.name,
                        measurements=self.measurements,
                        object_type=self.object_type,
                        object_id=self.object_id
                    )
        if self.nucleus_roi is not None:
            props['nucleus_roi'] = self.nucleus_roi.roi # Or __geo_interface__?
        return dict(type='Feature',
                    geometry=shape(self._roi.__geo_interface__),
                    properties=props)

    @property
    def object_type(self) -> bool:
        return self._object_type

    @property
    def is_detection(self) -> bool:
        return self._object_type in [ImageObjectType.DETECTION, ImageObjectType.CELL, ImageObjectType.TILE]

    @property
    def is_cell(self) -> bool:
        return self._object_type == ImageObjectType.CELL

    @property
    def is_tile(self) -> bool:
        return self._object_type == ImageObjectType.TILE

    @property
    def is_annotation(self) -> bool:
        return self._object_type == ImageObjectType.ANNOTATION

    @property
    def measurements(self) -> Dict[str, float]:
        if self._measurements is None:
            self._measurements = {}
        return self._measurements

    @property
    def roi(self) -> ROI:
        return self._roi

    @property
    def nucleus_roi(self) -> ROI:
        return self._nucleus_roi

    @nucleus_roi.setter
    def nucleus_roi(self, nucleus_roi: ROI):
        self._nucleus_roi = nucleus_roi

    @roi.setter
    def roi(self, roi: ROI):
        self._roi = roi

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: ROI):
        self._name = name

    @property
    def classification(self) -> Classification:
        return self._classification

    @classification.setter
    def classification(self, classification: Classification):
        self._classification = classification

    # @property
    # def parent(self) -> 'ImageObject':
    #     return self._parent
    #
    # @parent.setter
    # def parent(self, parent: 'ImageObject'):
    #     self._parent = parent


def create_tile(roi: Union[ROI, BaseGeometry], classification: Classification = None, name: str = None,
                measurements: Dict[str, float] = None, object_id: uuid.UUID = None):
    return ImageObject(roi=roi, classification=classification, name=name, measurements=measurements,
                       object_type=ImageObjectType.TILE, object_id=object_id)


def create_detection(roi: Union[ROI, BaseGeometry], classification: Classification = None, name: str = None,
                     measurements: Dict[str, float] = None, object_id: uuid.UUID = None):
    return ImageObject(roi=roi, classification=classification, name=name, measurements=measurements,
                       object_type=ImageObjectType.DETECTION, object_id=object_id)


def create_cell(roi: Union[ROI, BaseGeometry], nucleus_roi: Union[ROI, BaseGeometry], classification: Classification = None,
                name: str = None,
                measurements: Dict[str, float] = None, object_id: uuid.UUID = None):
    return ImageObject(roi=roi, nucleus_roi=nucleus_roi, classification=classification, name=name, measurements=measurements,
                       object_type=ImageObjectType.CELL, object_id=object_id)


def create_annotation(roi: Union[ROI, BaseGeometry], classification: Classification = None, name: str = None,
                      measurements: Dict[str, float] = None, object_id: uuid.UUID = None):
    return ImageObject(roi=roi, classification=classification, name=name, measurements=measurements,
                       object_type=ImageObjectType.ANNOTATION, object_id=object_id)


def get_classification(name: str, color: Tuple[int, int, int] = None):
    if name is None:
        return None
    classification = _cached_classifications.get(name)
    if classification is None:
        classification = Classification(name=name, color=color)
        _cached_classifications[classification.__str__()] = classification
    return classification


def parse_json_string(json_string: str) -> Union[ImageObject, List[ImageObject]]:
    json_objects = json.loads(json_string)
    return _parse_json_objects(json_objects)


def read_json(file) -> Union[ImageObject, List[ImageObject]]:
    """
    Read one or more objects from a GeoJSON representation.
    :param file:
    :return:
    """
    with open(file, 'r') as f:
        json_objects = json.load(f)
        return _parse_json_objects(json_objects)


def _parse_json_objects(json_objects) -> Union[ImageObject, List[ImageObject]]:
    """
    Read one or more objects from a GeoJSON representation.
    :param file:
    :return:
    """
    image_objects = []
    unwrap_single_object = False
    if isinstance(json_objects, dict):
        # Handle feature collection
        if 'features' in json_objects:
            json_objects = json_objects['features']

    if not isinstance(json_objects, List):
        json_objects = [json_objects]
        unwrap_single_object = True
    for obj in json_objects:
        geometry = obj['geometry']
        if not geometry:
            print('Skipping object without geometry')
            continue
        roi = create_roi(shape(geometry))
        object_id = None
        if 'id' in obj:
            try:
                object_id = uuid.UUID(obj['id'])
            except:
                pass
        nucleus_roi = create_roi(shape(obj['nucleusGeometry'])) if 'nucleusGeometry' in obj else None
        name = None
        measurements = None
        props = obj.get('properties')
        classification = None
        if props is not None:
            name = props.get('name')
            measurements = props.get('measurements')
            if 'classification' in props:
                classification = props['classification']
                if 'color' in classification:
                    color = _parse_color(classification['color'])
                elif 'colorRGB' in classification:
                    color = _parse_color(classification['colorRGB'])
                classification = get_classification(classification['name'], color)
            object_type = None
            if 'objectType' in obj:
                object_type = _get_object_type.get(obj['objectType'])
        path_object = ImageObject(roi=roi, nucleus_roi=nucleus_roi, classification=classification, name=name,
                                  measurements=measurements, object_type=object_type, object_id=object_id)
        image_objects.append(path_object)
    return image_objects[0] if len(image_objects) == 1 and unwrap_single_object else image_objects


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


def _get_object_type(name: str) -> ImageObjectType:
    """
    Try to get an object type from a string.
    """
    name_lower = name.lower().strip()
    if name_lower == 'detection':
        return ImageObjectType.DETECTION
    elif name_lower == 'cell':
        return ImageObjectType.CELL
    elif name_lower == 'tile':
        return ImageObjectType.TILE
    elif name_lower == 'annotation':
        return ImageObjectType.ANNOTATION
    elif name_lower == 'root':
        return ImageObjectType.ROOT
    else:
        warnings.warn(f"Unknown object type {name}")
        return None