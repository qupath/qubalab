from matplotlib.axes import Axes
from typing import Union
import numpy as np
import shapely
from shapely.geometry import shape
from shapely import plotting
from geojson import GeoJSON
from ..objects.image_feature import ImageFeature


def plotImage(ax: Axes, image: np.ndarray, channel: int = 0, offset: tuple = None) -> Axes:
    """
    A utility function to display a NumPy array returned by an ImageServer.

    :param ax: the Axes to plot to
    :param image: a numpy array with dimensions (c, y, x). If it represents a RGB image, all channels will be displayed.
                  Otherwise, only the specified channel will be displayed
    :param channel: the channel to display if the image is not RGB
    :param offset: an offset [x_min, y_min, x_max, y_max] to apply to the image. Useful if the image should match some features.
                   None to not define any offset
    :return: the Axes where the image was plotted
    """
    
    # Move channel axis
    image = np.moveaxis(image, [0], [2])

    # Keep only specified channel if the image is not RGB
    if image.shape[2] != 3 or image.dtype != "uint8":
        image = image[..., channel]

    # If pixel values are float numbers, they can be greater than 0, while matplotlib only accepts
    # values between 0 and 1, so the array is normalized between 0 and 1
    if "float" in str(image.dtype):
        image /= image.max()

    if offset is not None and len(offset) == 4:
        # y_min and y_max are swapped because otherwise image data is flipped vertically
        ax.imshow(image, extent=(offset[0], offset[2], offset[3], offset[1]))
    else:
        ax.imshow(image)

    return ax


def plotImageFeatures(
    ax: Axes,
    image_features: list[ImageFeature],
    region: GeoJSON = None,
    default_color: tuple = (0.2, 0.8, 0.2),
    fill_opacity: float = 0.25
) -> Axes:
    """
    A utility function to display a image features. This will plot the object's geometries
    and nucleus geometries.
    
    :param ax: the Axes to plot to
    :param image_features: the image features to plot
    :param region: the region to plot. Features outside this region won't be plotted.
                   All features are plotted if this parameter is undefined. 
    :param default_color: the color to use for unclassified features
    :param fill_opacity: the opacity of the fill color. Set to 0 to not fill objects
    :return: the Axes where the image was plotted
    """
    envelope = None if region is None else shape(region).envelope

    for feature in image_features:
        if feature.color is not None:
            edge_color = tuple(c/255.0 for c in feature.color)
        elif feature.classification is not None:
            edge_color = tuple(c/255.0 for c in feature.classification.color)
        else:
            edge_color = default_color

        if fill_opacity > 0:
            face_color = edge_color + (fill_opacity,)
        else:
            face_color = None

        for roi in [getattr(feature, name, None) for name in ['geometry', 'nucleus_geometry']]:
            if roi is not None:
                geometry = shape(roi)
                if envelope is None or envelope.intersects(geometry.envelope):
                    _plot_geometry(ax, shape(roi), face_color, edge_color)

    ax.axis('equal')
    return ax


def _plot_geometry(ax: Axes, geometry: shapely.Geometry, face_color: tuple, edge_color: tuple):
    if (isinstance(geometry, Union[shapely.Point, shapely.MultiPoint])):
        plotting.plot_points(geometry, ax, color=edge_color)
    elif (isinstance(geometry, Union[shapely.LineString, shapely.LinearRing])):
        plotting.plot_line(geometry, ax, add_points=False, color=edge_color)
    elif (isinstance(geometry, Union[shapely.Polygon, shapely.MultiPolygon])):
        plotting.plot_polygon(geometry, ax, add_points=False, facecolor=face_color, edgecolor=edge_color, linewidth=2)
    elif (isinstance(geometry, shapely.MultiLineString)):
        for line in geometry.geoms:
            plotting.plot_line(line, ax, add_points=False, color=edge_color)
    elif (isinstance(geometry, shapely.GeometryCollection)):
        for g in geometry.geoms:
            _plot_geometry(ax, g, face_color, edge_color)
