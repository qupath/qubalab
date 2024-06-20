import geojson.mapping

from . import ImageServerMetadata
from .servers import WrappedImageServer, ImageServer, Region2D
from ..objects.geometries import to_geometry
from ..objects.objects import _NUCLEUS_GEOMETRY_KEY
from PIL import Image, ImageDraw
from shapely.geometry import *
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from qubalab.objects import utils
import numpy as np
from typing import Union, Iterable, Dict, Tuple
from qubalab.objects import ImageFeature, Classification, get_classification
from shapely.strtree import STRtree
from math import ceil
import rasterio.features
from rasterio.transform import Affine


def labels_to_features(lab: np.ndarray, object_type='annotation', connectivity: int = 4,
                       transform: Affine = None, downsample: float = 1.0, include_labels=False,
                       classification: Union[str, Dict[float, str]] = None):
    """
    Create a GeoJSON FeatureCollection from a labeled image.
    """
    features = []

    # Ensure types are valid
    if lab.dtype == bool:
        mask = lab
        lab = lab.astype(np.uint8)
    else:
        mask = lab > 0

    # Create transform from downsample if needed
    if transform is None:
        transform = Affine.scale(downsample)

    # Trace geometries
    existing_features = {}
    merged_features = False
    for geometry, label in rasterio.features.shapes(lab, mask=mask,
                                      connectivity=connectivity, transform=transform):

        if label in existing_features:
            geom = shape(geometry).union(shape(existing_features[label]['geometry']))
            existing_features[label]['geometry'] = geom
            merged_features = True
        else:
            # Create properties
            props = dict(object_type=object_type)
            if include_labels:
                props['measurements'] = {'Label': float(label)}

            # Add a classification if we have one
            if isinstance(classification, str):
                props['classification'] = classification
            elif isinstance(classification, dict):
                props['classification'] = get_classification(label, classification)

            # Wrap in a dict to effectively create a GeoJSON Feature
            feature = dict(type="Feature", geometry=geometry, properties=props)
            existing_features[label] = feature
            features.append(feature)

    # If we've been merging, we need to ensure we have GeoJSON-compatible geometries
    if merged_features:
        for feature in features:
            feature['geometry'] = geojson.mapping.to_mapping(feature['geometry'])

    return features


def rasterize(image_objects: Iterable[object | BaseGeometry | ImageFeature] = None,
              region: Union[Tuple, Region2D] = None,
              downsample: float = None,
              value: Union[Iterable, float, int] = 255,
              draw_type: str = 'fill',
              im: np.ndarray = None,
              prefer_nucleus: bool = False):
    """
    Create a raster images for one or more ImageObjects, ROIs or Geometries.

    :param image_objects:  Iterable of objects, ROIs or Geometries to draw.
                           These will be filtered according to the region.
    :param region:         Region corresponding to the image that should be generated
    :param downsample:     Downsample factor; not needed if region is a Region2D
    :param value:          Pixel value used for draw an object in the image, or an iterable the same length as image_objects
    :param draw_type:      Either 'fill', 'outline' or 'both'.
    :param im:             Image where the output should be drawn (optional).
    :param prefer_nucleus: If cell objects are used, prefer to use the nucleus rather than cel boundary.
    :return:               An image with the objects drawn
    """

    draw_type = draw_type.lower()
    do_fill = draw_type in ('fill', 'both')
    do_outline = draw_type in ('outline', 'both')
    if not do_fill and not do_outline:
        raise ValueError(f"Rasterize called with draw_type other than 'fill', 'outline' or 'both'")

    image_objects = list(image_objects)

    # TODO: Make this sensible... currently generates GeoJSON geometries because these can include z and t, AND
    #       shapely geometries required for drawing
    preferred_geom = _NUCLEUS_GEOMETRY_KEY if prefer_nucleus else None
    geometries = [to_geometry(o, preferred_geometry=preferred_geom) for o in image_objects]
    shapely_geometries = [utils._to_geometry(geom) for geom in geometries]

    # Generate region bounds around objects, if needed
    if region is None:
        all_bounds = np.row_stack([geom.bounds for geom in shapely_geometries if geom is not None])
        z = list(set([g.plane.z for g in geometries if g is not None]))
        t = list(set([g.plane.t for g in geometries if g is not None]))
        if len(z) != 1 or len(t) != 1:
            raise ValueError('ImageObjects must all fall on a single plane (with equal values of z and t)')
        xy = np.floor(np.min(all_bounds[:, :2], axis=0))
        wh = np.ceil(np.max(all_bounds[:, 2:], axis=0)) - xy
        region = Region2D(downsample=downsample,
                          x=int(xy[0]),
                          y=int(xy[1]),
                          width=int(wh[0]),
                          height=int(wh[1]),
                          z=z[0],
                          t=t[0])

    envelope = Polygon.from_bounds(region.x, region.y, region.x + region.width, region.y + region.height)

    # Filter out geometries beyond the bounds
    geometries = [g for g in shapely_geometries if envelope.intersects(g.envelope)]

    # Identify the maximum value the image should contain
    multiple_values = isinstance(value, Iterable)
    if multiple_values:
        max_value = float('inf')
        gen_values = iter(value)
    else:
        max_value = value

    if im is None:
        width = int(ceil(region.width / region.downsample))
        height = int(ceil(region.height / region.downsample))
        if max_value < 0 or max_value > 255:
            mode = 'F'
        elif max_value == 1:
            mode = '1'
        else:
            mode = 'P'
        image = Image.new(mode, (width, height))
    else:
        image = Image.fromarray(im)

    if geometries:

        # Err on the side of a slightly bigger envelope for clipping
        envelopePadded = Polygon.from_bounds(
            region.x - 1,
            region.y - 1,
            region.x + region.width + 2,
            region.y + region.height + 2)

        draw = ImageDraw.Draw(image)

        for geom in geometries:
            if multiple_values:
                val = next(gen_values)
            else:
                val = value

            # If we are filling and not outlining, we can clip the geometry
            # This may improve performance for very large annotations where only
            # a small part is visible
            if do_fill and not envelopePadded.covers(geom):
                # print('Intersecting!')
                geom = geom.intersection(envelopePadded)
                if geom.is_empty:
                    # print('Skipping!')
                    continue

            if _has_holes(geom):
                # If we have holes, we risk overpainting existing pixels with the background color
                # For that reason we need to create a separate binary image and then copy the values
                # TODO: Check/improve method of handling holes
                fill = 1 if do_fill else None
                outline = 1 if do_outline else None
                bitmap = Image.new('1', image.size)
                draw2 = ImageDraw.Draw(bitmap)
                _draw_geometry(geom, draw2, fill=fill, outline=outline, region=region)
                draw.bitmap((0, 0), bitmap, fill=val)
            else:
                fill = val if do_fill else None
                outline = val if do_outline else None
                _draw_geometry(geom, draw, fill=fill, outline=outline, region=region)
    return np.asarray(image)


def _has_holes(geom: BaseGeometry) -> bool:
    """
    Check if a geometry contains any interior holes.
    """
    if isinstance(geom, Polygon):
        return len(geom.interiors) > 0
    elif isinstance(geom, BaseMultipartGeometry):
        for g2 in geom.geoms:
            if _has_holes(g2):
                return True
    return False


def _draw_geometry(geom: BaseGeometry, draw: ImageDraw, fill=None, outline=None, fill_hole=0,
                   region: Region2D = None):
    if isinstance(geom, BaseMultipartGeometry):
        for g2 in geom.geoms:
            _draw_geometry(g2, draw, fill=fill, outline=outline, fill_hole=fill_hole, region=region)
    elif isinstance(geom, Polygon):
        _draw_linestring(geom.exterior, draw, fill=fill, outline=outline, region=region)
        for hole in geom.interiors:
            _draw_linestring(hole, draw, fill=fill_hole, outline=outline, region=region)
    elif isinstance(geom, LineString):
        _draw_linestring(geom, draw, fill=None, outline=outline, region=region)
    elif isinstance(geom, Point):
        draw.point([geom.x, geom.y], fill=fill)
    else:
        raise ValueError(f'Unsupported geometry {geom}')


def _draw_linestring(line_string: LineString, draw: ImageDraw, close_path: bool = False, fill=None, outline=None,
                     region: Region2D = None):
    if region is None:
        args = {}
    else:
        args = dict(
            downsample=region.downsample, x_origin=region.x, y_origin=region.y
        )
    vertices, codes = utils._linestring_to_vertices(line_string, close_path=close_path, **args)
    draw.polygon(list(vertices.flatten()), fill=fill, outline=outline)


def _ensure_classification(obj):
    if obj is None:
        return obj
    if isinstance(obj, Classification):
        return obj
    return get_classification(name=obj)


class LabeledImageServer(WrappedImageServer):

    def __init__(self,
                 base_server: ImageServer,
                 image_objects: Iterable[ImageFeature],
                 label_map: Dict[Classification, int] = None,
                 downsample: float = None,
                 multichannel: bool = False,
                 prefer_nucleus: bool = False,
                 dtype=np.float32
                 ):
        # super(LabeledImageServer, self).__init__(base_server=base_server)
        super().__init__(base_server=base_server)
        if downsample is None:
            downsamples = self.base_server.downsamples  # (self.base_server.downsamples[0],)
        else:
            downsamples = (downsample,)
        self._multichannel = multichannel
        self._prefer_nucleus = prefer_nucleus
        if self._multichannel:
            n_channels = max(label_map.values())
        else:
            n_channels = 1
        if label_map is None:
            self._unique_labels = True
            self._label_map = None
            if self._multichannel:
                raise ValueError(f'Multichannel output is not supported without a label_map!')
        else:
            self._unique_labels = False
            self._label_map = {_ensure_classification(k): v for k, v in label_map.items()}
        self._build_cache(image_objects)

        path = f'{base_server.path} - {label_map}'
        name = f'{base_server.metadata.name} - labels'
        shape = list(base_server.shape)
        if len(shape) > 2:
            shape[2] = n_channels
        elif shape == 2:
            shape.append(n_channels)
        else:
            raise ValueError(f'Invalid image shape {shape}')
        self._metadata = ImageServerMetadata(
            path=path,
            name=name,
            shape=tuple(shape),
            downsamples=downsamples,
            pixel_calibration=base_server.metadata.pixel_calibration,
            dtype=dtype
        )

    def _build_metadata(self) -> ImageServerMetadata:
        return self._metadata

    def _build_cache(self, image_objects: Iterable[ImageFeature]):
        geometries = []
        lab = 1
        for path_object in image_objects:
            # We only need to store objects that have both ROIs
            # and classifications found in our label map
            if path_object.roi is None:
                continue
            if self._label_map is not None and not path_object.path_class in self._label_map:
                continue
            geom = path_object.roi.geometry.envelope
            geom.path_object = path_object
            if self._unique_labels:
                geom.label = lab
                lab += 1
            else:
                geom.label = self._label_map[path_object.path_class]
            geometries.append(geom)
        print(f'Building tree with {len(geometries)} geometries')
        self._tree = STRtree(geometries)

    def read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        from .servers import _validate_block

        x, y, width, height, z, t = _validate_block(block)
        if self._multichannel:
            shape = (height, width, self.n_channels)
        else:
            shape = (height, width)
        im = np.zeros(shape, dtype=self.dtype)

        level_downsample = self.downsamples[level]
        region = Region2D(downsample=level_downsample,
                          x=int(x * level_downsample),
                          y=int(y * level_downsample),
                          width=int(ceil((x + width) * level_downsample)),
                          height=int(ceil((y + height) * level_downsample))
                          )
        bounds = region.geometry
        geometries = self._tree.query(bounds)

        # If we don't have anything, we can return quickly
        if not geometries:
            return im

        if self._unique_labels:
            labels = [g.label for g in geometries]
            geoms = [utils._to_geometry(g.path_object, prefer_nucleus=self._prefer_nucleus) for g in geometries]
            im = rasterize(geoms, region=region, value=labels, draw_type='fill', im=im)
        else:
            for path_class, value in self._label_map.items():
                geoms = [utils._to_geometry(g.path_object, prefer_nucleus=self._prefer_nucleus) for g in geometries
                         if g.path_object.path_class == path_class]
                if not geoms:
                    continue
                if self._multichannel:
                    im_temp = im[:, :, value]
                    val = 1
                else:
                    im_temp = im
                    val = value
                # print('Will call')
                im = rasterize(geoms, region=region, value=val, draw_type='fill', im=im_temp)
                # print('Did call')
        # print(f'Calculating: {time.time() - start_time}')
        return im

    @property
    def get_label_map(self) -> Dict[Classification, int]:
        """
        Get a copy of the label map.
        :return:
        """
        return self._label_map.copy()
