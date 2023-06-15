from shapely.geometry.base import BaseGeometry
from shapely.geometry import *


class ROI(object):

    def __init__(self, geometry: BaseGeometry, z: int = 0, t: int = 0):
        """
        A ROI is a region of interest in an image.
        It is defined by a geometry (a shapely geometry object) and a z and t index
        to support z-stacks and time series.
        """
        self._geometry = geometry
        self._z = z
        self._t = t

    @property
    def __geo_interface__(self):
        # See https://gist.github.com/sgillies/2217756
        # TODO: Consider if this is useful
        return self._geometry.__geo_interface__

    @property
    def t(self) -> int:
        return self._t

    @property
    def z(self) -> int:
        return self._z

    @property
    def geometry(self) -> BaseGeometry:
        return self._geometry


def create_roi(geometry: BaseGeometry, z: int = 0, t: int = 0) -> ROI:
    """
    Create a ROI from a shapely geometry object.
    """
    if not isinstance(geometry, BaseGeometry):
        raise ValueError(f'{geometry} is not an instance of BaseGeometry!')
    return ROI(geometry, z=z, t=t)


def create_rectangle(x: float, y: float, width: float, height: float, z: int = 0, t: int = 0):
    """
    Create a rectangular ROI.
    """
    polygon = Polygon.from_bounds(x, y, x + width, y + height)
    return create_roi(polygon, z=z, t=t)


def create_line(x1: float, y1: float, x2: float, y2: float, z: int = 0, t: int = 0):
    """
    Create a line ROI.
    """
    line = LineString([(x1, y1), (x2, y2)])
    return create_roi(line, z=z, t=t)
