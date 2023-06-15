from ..images import ImageServer, Region2D
from ..objects import utils
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from geojson import GeoJSON, Feature
from typing import Iterable, Tuple, Union
from shapely.geometry import Polygon, shape
from collections import namedtuple

Color = namedtuple('Color', field_names=('red', 'green', 'blue', 'alpha'),
                   defaults=(0, 0, 0, 0))


def show_image(image: Union[np.ndarray, ImageServer] = None,
               image_objects: Union[Feature, Iterable[Feature]] = None,
               rois: Union[GeoJSON, Iterable[GeoJSON]] = None,
               region: [Region2D, GeoJSON] = None,
               downsample: float = None,
               default_color: Tuple = (0.2, 0.8, 0.2), fill_opacity: float = 0.25, ax=None):
    """
    Show an image region, optionally with objects or ROIs on top.

    :param server:
    :param image_objects:
    :param rois:
    :param region:
    :param downsample:
    :param default_color: used for unclassified objects or for ROIs
    :param fill_opacity:
    :param ax:
    :return:
    """

    # TODO: Consider removing shapely dependency to (possibly) improve performance
    #       Here we create shapes from ROIs, although this might not be needed
    #       (since we mostly use it to check envelope intersection, then convert again to
    #       a matplotlib path)

    if ax is None:
        fig, ax = plt.subplots(1)

    if isinstance(region, GeoJSON):
        region = Region2D.from_geometry(region, downsample=downsample)

    if image is not None:
        if isinstance(image, ImageServer) or True:
            if downsample is None and not isinstance(region, Region2D):
                downsample = image.downsamples[0]
            im = image.read_region(region=region, downsample=downsample)
        else:
            im = np.asarray(image)
        ax.imshow(im)

    if fill_opacity > 0:
        default_face_color = default_color + (fill_opacity,)
    else:
        default_face_color = None

    envelope = None
    args = dict(downsample=downsample)
    if region is not None:
        if isinstance(region, Region2D):
            args['x_origin'] = region.x
            args['y_origin'] = region.y
            if region.downsample is not None:
                args['downsample'] = region.downsample
            envelope = Polygon.from_bounds(region.x, region.y, region.x+region.width, region.y+region.height).envelope
        else:
            args['x_origin'] = region[0]
            args['y_origin'] = region[1]
            envelope = Polygon.from_bounds(region[0], region[1], region[0]+region[2], region[1]+region[3]).envelope

    if rois is not None:
        for roi in utils._ensure_iterable(rois):
            geom = shape(roi)
            if envelope is not None and not envelope.intersects(geom.envelope):
                continue
            for path in utils._geometry_to_paths(geom, **args):
                patch = patches.PathPatch(path, facecolor=default_face_color, edgecolor=default_color, fill=fill_opacity > 0)
                ax.add_patch(patch)

    if image_objects is not None:
        for image_object in utils._ensure_iterable(image_objects):
            for roi in [getattr(image_object, name, None) for name in ['geometry', 'nucleus_geometry']]:
                if roi is None:
                    continue
                geom = shape(roi)
                if envelope is not None and not envelope.intersects(geom.envelope):
                    continue
                color = default_color
                face_color = default_face_color
                classification = getattr(image_object, 'classification', None)
                if classification is not None and 'color' in classification:
                    color = tuple(c/255.0 for c in classification['color'])
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
