from shapely.geometry import *
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

import numpy as np
from matplotlib.path import Path

from .objects import ImageObject
from .rois import ROI, create_roi

from typing import Iterable, List, Tuple


def _to_geometry(obj, prefer_nucleus: bool = False) -> BaseGeometry:
    """
    Try to extract a shapely geometry from an object.
    """
    if isinstance(obj, ImageObject):
        obj = obj.nucleus_roi if prefer_nucleus else obj.roi
    if isinstance(obj, ROI):
        return obj.geometry
    if isinstance(obj, BaseGeometry):
        return obj
    if obj is None:
        return None
    return shape(obj)


def _to_roi(obj, prefer_nucleus: bool = False) -> ROI:
    """
    Try to extract a ROI from an object.
    """
    if isinstance(obj, ImageObject):
        return obj.nucleus_roi if prefer_nucleus else obj.roi
    if isinstance(obj, ROI):
        return obj.geometry
    geometry = _to_geometry(obj, prefer_nucleus=prefer_nucleus)
    if geometry is None:
        return None
    else:
        return create_roi(geometry)


def _ensure_iterable(something, wrap_strings: bool = True) -> Iterable:
    """
    Ensure that the input is iterable, according to the following rules:
      * if the input is None, return an empty iterable
      * if the input is already iterable, return it unchanged - unless it is a string, and wrap_strings is True
      * otherwise, return an iterable containing the input
    """
    if something is None:
        return []
    if wrap_strings and isinstance(something, str):
        return [something]
    if isinstance(something, Iterable):
        return something
    return [something]


def _geometry_to_paths(geom: BaseGeometry, paths: List[Path] = None, **kwargs):
    """
    Convert a shapely geometry to a list of matplotlib paths.
    """
    if paths is None:
        paths = []
    if isinstance(geom, BaseMultipartGeometry):
        for geom2 in geom.geoms:
            _geometry_to_paths(geom2, paths=paths, **kwargs)
    else:
        paths.append(_polygon_to_path(geom, **kwargs))
    return paths


def _polygon_to_path(geom: Polygon, **kwargs):
    """
    Convert a shapely polygon to a matplotlib path.
    """
    vertices, codes = _linestring_to_vertices(geom.exterior, close_path=True, **kwargs)
    for hole in geom.interiors:
        v2, c2 = _linestring_to_vertices(hole, close_path=True, **kwargs)
        vertices = np.vstack([vertices, v2])
        codes = np.concatenate([codes, c2], axis=0)
    return Path(vertices, codes=codes)


def _linestring_to_path(line_string: LineString, close_path: bool = False,
                        downsample: float = 1.0, x_origin: float = 0, y_origin: float = 0):
    """
    Convert a shapely linestring to a matplotlib path.
    """
    vertices, codes = _linestring_to_vertices(line_string, close_path=close_path,
                                              downsample=downsample, x_origin=x_origin, y_origin=y_origin)
    return Path(vertices, codes)


def _linestring_to_vertices(line_string: LineString,
                            close_path: bool = False,
                            downsample: float = 1.0,
                            x_origin: float = 0, y_origin: float = 0) -> Tuple[np.ndarray, List]:
    """
    Convert a shapely linestring to a list of vertices and matplotlib path codes.
    """
    x, y = line_string.xy
    x = (np.asarray(x) - x_origin) / downsample
    y = (np.asarray(y) - y_origin) / downsample
    codes = np.ones(x.shape, dtype=np.uint8) * Path.LINETO
    codes[0] = Path.MOVETO
    if close_path and len(x) > 1:
        codes[-1] = Path.CLOSEPOLY
    return np.column_stack([x, y]), codes
