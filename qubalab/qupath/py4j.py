from ..images import ImageServer, PixelLength, PixelCalibration, ImageServerMetadata, ImageShape, ImageChannel
from ..images.servers import _validate_block, _resize

from dataclasses import astuple, dataclass
from typing import List, Tuple
from py4j.java_gateway import JavaGateway, JavaObject, GatewayParameters

from imageio.v3 import imread
from imageio import volread
import numpy as np

from urllib.parse import urlparse, unquote

import warnings


"""
Store default Gateway so it doesn't need to always be passed as a parameter
"""
_default_gateway: JavaGateway = None


class QuPathServer(ImageServer):
    
    def __init__(self, gateway: JavaGateway = None, server_obj: JavaObject = None, use_temp_files = False, **kwargs):
        """_summary_
        Args:
            gateway (JavaGateway, optional): _description_. Defaults to None.
            server_obj (JavaObject, optional): _description_. Defaults to None.
            use_temp_files (bool, optional): Use temp files when requesting pixels. Temp files can be faster than passing byte arrays with py4j - although not so kind on the hard disk. Defaults to False.

        Raises:
            ValueError: _description_
        """
        super().__init__(**kwargs)
        self._use_temp_files = use_temp_files
        self._gateway = _gateway_or_default(gateway)
        # Get the current server if none is specified
        if server_obj is None:
            server_obj = get_current_image_data(gateway).getServer()
        self._server_obj = server_obj
        if self._gateway is None:
            raise ValueError('A JavaGateway is needed! See create_gateway() for details.')


    def _build_metadata(self) -> ImageServerMetadata:
        # Get what we can directly from the server
        server = self._server_obj
        downsamples = tuple([d for d in server.getPreferredDownsamples()])
        
        self._downsamples = downsamples # Use the QuPath values directly
        
        shapes = tuple([ImageShape(x=level.getWidth(), y=level.getHeight(), c=server.nChannels(), z=server.nZSlices(), t=server.nTimepoints())
            for level in server.getMetadata().getLevels()])

        dt = np.dtype(server.getPixelType().toString().lower())
        is_rgb = server.isRGB()

        meta = server.getMetadata()
        name = meta.getName()
        cal = server.getPixelCalibration()

        channels = tuple([ImageChannel(c.getName(), _unpack_color(c.getColor())) for c in meta.getChannels()])

        # Try to get the file path
        path = _find_server_file_path(server)
        if path is None:
            path = server.getPath()
        
        if cal.hasPixelSizeMicrons():
            if cal.getZSpacingMicrons():
                length_z = PixelLength.create_microns(cal.getZSpacingMicrons())
            else:
                length_z = PixelLength()
            pixel_cal = PixelCalibration(
                length_x = PixelLength.create_microns(cal.getPixelWidthMicrons()),
                length_y = PixelLength.create_microns(cal.getPixelHeightMicrons()),
                length_z = length_z
                )
        else:
            pixel_cal = PixelCalibration()

        return ImageServerMetadata(
            path=path,
            name=name,
            pixel_calibration=pixel_cal,
            shapes=shapes,
            dtype=dt,
            is_rgb=is_rgb,
            channels=channels
        )


    def read_block(self, level: int, block: Tuple[int, ...]) -> np.ndarray:
        _, x, y, width, height, z, t = astuple(_validate_block(block))

        gateway = self._gateway
        server = self._server_obj

        # TODO: Explore requesting directly in QuPath - this is awkward and could result in 
        # rounding problems
        if level < 0:
            level = len(self.downsamples) + level
        downsample = server.getDownsampleForResolution(level)

        if self._use_temp_files:
            import tempfile, os
            temp_path = tempfile.mkstemp(prefix='qubalab-', suffix='.tif')[1]
            request = gateway.jvm.qupath.lib.regions.RegionRequest.createInstance(
                server.getPath(), downsample,
                int(round(x * downsample)),
                int(round(y * downsample)),
                int(round(width * downsample)),
                int(round(height * downsample)),
                z,
                t)
            gateway.entry_point.writeImageRegion(server, request, temp_path)
            if self.metadata.is_rgb or self.n_channels == 1:
                im = imread(temp_path)
            else:
                # We can just provide 2D images; using volread move to channels-last
                im = np.moveaxis(volread(temp_path), 0, -1)
            os.remove(temp_path)
        else:
            # import time
            # start_time = time.time()
            fmt = 'png' if self.is_rgb else "imagej tiff"
            byte_array = gateway.entry_point.getImageBytes(server,
                downsample,
                int(round(x * downsample)),
                int(round(y * downsample)),
                int(round(width * downsample)),
                int(round(height * downsample)),
                z,
                t,
                fmt)
            # end_time = time.time()
            # import threading
            # thread_id = threading.current_thread().ident
            # print(f'Read time: {end_time - start_time:.2f} seconds, length: {len(byte_array)}, thread: {thread_id}')

            if self.metadata.is_rgb or self.n_channels == 1:
                im = imread(byte_array)
            else:
                # We can just provide 2D images; using volread move to channels-last
                im = np.moveaxis(volread(byte_array), 0, -1)

        if height != im.shape[0] or width != im.shape[1]:
            shape_before = im.shape
            im = _resize(im, (width, height), self.resize_method)
            warnings.warn(f'Block needs to be reshaped from {shape_before} to {im.shape}')

        return im




def _unpack_color(rgb: int) -> Tuple[float, float, float]:
    r = (rgb >> 16) & 255
    g = (rgb >> 8) & 255
    b = rgb & 255
    return r/255.0, g/255.0, b/255.0


def _get_server_uris(server: JavaObject) -> Tuple[str]:
    """
    Get URIs for a java object representing an ImageServer.
    """
    return tuple(str(u) for u in server.getURIs())

def _find_server_file_path(server: JavaObject) -> str:
    """
    Try to get the file path for a java object representing an ImageServer.
    This can be useful
    """
    uris = _get_server_uris(server)
    if len(uris) == 1:
        parsed = urlparse(uris[0])
        if parsed.scheme == 'file':
            return unquote(parsed.path)
    return None


def get_current_image_data(gateway: JavaGateway = None) -> JavaObject:
    """
    Get the current ImageData open in QuPath
    """
    return _gateway_or_default(gateway).entry_point.getQuPath().getImageData()


def create_gateway(auto_convert=True, auth_token=None, set_as_default=True, **kwargs) -> JavaGateway:
    """
    Create a new JavaGateway to communicate with QuPath.
    This requires also launching QuPath and activating Py4J from there first.
    """
    if auth_token is not None:
        params = GatewayParameters(auth_token=auth_token)
        gateway = JavaGateway(auto_convert=auto_convert, gateway_parameters=params, **kwargs)
    else:  
        # from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
        # gateway = ClientServer(
        #     java_parameters=JavaParameters(auto_convert=auto_convert),
        #     python_parameters=PythonParameters())
        gateway = JavaGateway(auto_convert=auto_convert, **kwargs)
    if set_as_default:
        set_default_gateway(gateway)
    return gateway

def set_default_gateway(gateway: JavaGateway = None):
    """
    Set the default JavaGateway to use if one is not otherwise specified.
    """
    global _default_gateway
    _default_gateway = gateway

def _gateway_or_default(*args, **kwargs):
    """
    Attempt to get a gateway, by
    * returning one of the input argument, if it's a gateway
    * returning the default gateway, if available
    * creating a new gateway using the **kwargs (if needed, and displaying a warning)
    """
    for arg in args:
        if isinstance(arg, JavaGateway):
            return arg
    gateway = _default_gateway
    if gateway is None:
        warnings.warn('Attempting to create new JavaGateway')
        gateway = create_gateway(**kwargs)
    return gateway


def _get_java_class_name(input: JavaObject) -> str:
    return str(input.getClass().getName())


def _get_java_image_data(input) -> JavaObject:
    """
    Get an ImageData from the input, if possible.
    """
    if input is None:
        return get_current_image_data()
    if isinstance(input, JavaGateway):
        return get_current_image_data(gateway=input)
    if isinstance(input, JavaObject):
        cls = _get_java_class_name(input)
        if cls == 'qupath.lib.images.ImageData':
            return input
    return None


def _get_java_hierarchy(input) -> JavaObject:
    """
    Get a PathObjectHierarchy from the input, if possible.
    """
    if isinstance(input, JavaObject):
        cls = _get_java_class_name(input)
        if cls == 'qupath.lib.objects.hierarchy.PathObjectHierarchy':
            return input
    image_data = _get_java_image_data(input)
    return None if image_data is None else image_data.getHierarchy()


def _get_java_server(input) -> JavaObject:
    """
    Get an ImageServer from the input, if possible.
    """
    # Could use this, but then we definitely need a gateway
#    cls_server = gateway.jvm.Class.forName('qupath.lib.images.servers.ImageServer')
    if isinstance(input, JavaObject):
        for cls in input.getClass().getInterfaces():
            if str(cls.getName()) == 'qupath.lib.images.servers.ImageServer':
                return input
    image_data = _get_java_image_data(input)
    return None if image_data is None else image_data.getServer()


def add_objects(features, image_data: JavaObject=None, gateway: JavaGateway=None):
    image_data = _get_java_image_data(gateway=gateway, image_data=image_data)
    hierarchy = image_data.getHierarchy()
    gateway


def get_server(input = None) -> ImageServer:
    """
    Get an ImageServer from the input
    """
    if isinstance(input, ImageServer):
        return input
    server = _get_java_server(input)
    return None if server is None else QuPathServer(
        gateway = _gateway_or_default(),
        server_obj = server)

import geojson
from geojson import Feature

def get_path_objects(input = None, object_type: str = None, gateway = None) -> List[Feature]:
    gateway = _gateway_or_default(input, gateway)
    hierarchy = _get_java_hierarchy(input)
    if hierarchy is None:
        warnings.warn('No object hierarchy found')
        return []
    if object_type is None:
        path_objects = hierarchy.getObjects(None, None)
    elif object_type == 'annotation':
        path_objects = hierarchy.getAnnotationObjects()
    elif object_type == 'detection':
        path_objects = hierarchy.getDetectionObjects()
    elif object_type == 'tile':
        path_objects = hierarchy.getTileObjects()
    elif object_type == 'cell':
        path_objects = hierarchy.getCellObjects()
    elif object_type == 'tma':
        tma_grid = hierarchy.getTMAGrid()
        pathObjects = [] if tma_grid is None else tma_grid.getTMACoreList()

    feature_list = gateway.entry_point.toGeoJson(path_objects)
    return [geojson.loads(f) for f in feature_list]



"""
class HierarchyWrapper:
    pass

class ImageDataWrapper:

    def __init__(self):
        pass

    def get_hierarchy() -> HierarchyWrapper:
        pass

    def get_server() -> ImageServer:
        pass
"""


# import geojson

# class PathObject(geojson.Feature):

#     def __init__(self, **kwargs):
#         super.__init__(self, **kwargs)