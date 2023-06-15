from ..images import ImageServer, Region2D
from ..objects import ROI, ImageObject
from ..objects import utils
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from typing import Iterable, Tuple, Union
from shapely.geometry import *
from shapely.geometry.base import BaseMultipartGeometry, BaseGeometry
from collections import namedtuple

Color = namedtuple('Color', field_names=('red', 'green', 'blue', 'alpha'),
                   defaults=(0, 0, 0, 0))

def show_image(image: Union[np.ndarray, ImageServer] = None,
               path_objects: Union[ImageObject, Iterable[ImageObject]] = None,
               rois: Union[ROI, BaseGeometry, Iterable[ROI], Iterable[BaseGeometry]] = None,
               region: Region2D = None, downsample: float = None,
               default_color: Tuple = (0.2, 0.8, 0.2), fill_opacity: float = 0.25, ax=None):
    """
    Show an image region, optionally with objects or ROIs on top.

    :param server:
    :param path_objects:
    :param rois:
    :param region:
    :param downsample:
    :param default_color: used for unclassified objects or for ROIs
    :param fill_opacity:
    :param ax:
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots(1)

    if image is not None:
        if isinstance(image, ImageServer) or True:
            if downsample is None and not isinstance(region, Region2D):
                downsample = image.downsamples[0]
                print(f'Downsample {downsample}')
            im = image.read_region(region=region, downsample=downsample)
        else:
            im = np.asarray(image)
        ax.imshow(im)

    if fill_opacity > 0:
        default_face_color = default_color + (fill_opacity,)
    else:
        default_face_color = None

    envelope = None
    if isinstance(region, Region2D):
        args = dict(downsample=downsample, x_origin=region.x, y_origin=region.y)
        envelope = Polygon.from_bounds(region.x, region.y, region.x+region.width, region.y+region.height).envelope
    elif region is None:
        args = dict(downsample=downsample)
    else:
        args = dict(downsample=downsample, x_origin=region[0], y_origin=region[1])
        envelope = Polygon.from_bounds(region[0], region[1], region[0]+region[2], region[1]+region[3]).envelope

    if rois is not None:
        for roi in utils._ensure_iterable(rois):
            geom = roi.geometry if isinstance(roi, ROI) else roi
            if envelope is not None and not envelope.intersects(geom.envelope):
                continue
            for path in utils._geometry_to_paths(geom, **args):
                patch = patches.PathPatch(path, facecolor=default_face_color, edgecolor=default_color, fill=fill_opacity > 0)
                ax.add_patch(patch)

    if path_objects is not None:
        for path_object in utils._ensure_iterable(path_objects):
            for roi in [path_object.roi, path_object.nucleus_roi]:
                if roi is None:
                    continue
                geom = roi.geometry
                if envelope is not None and not envelope.intersects(geom.envelope):
                    continue
                color = default_color
                face_color = default_face_color
                if path_object.path_class is not None:
                    color = tuple(c/255.0 for c in path_object.path_class.color)
                    if fill_opacity > 0:
                        face_color = color + (fill_opacity,)
                    else:
                        face_color = None
                for path in utils._geometry_to_paths(geom, **args):
                    if downsample >= 10:
                        geom = geom.simplify(downsample)
                    patch = patches.PathPatch(path,
                                              facecolor=face_color,
                                              edgecolor=color,
                                              fill=fill_opacity > 0)
                    ax.add_patch(patch)
