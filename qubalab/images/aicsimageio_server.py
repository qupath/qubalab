import numpy as np
import math
from pathlib import Path
from aicsimageio import AICSImage
from dataclasses import astuple
from .image_server import ImageServer
from .metadata.image_server_metadata import ImageServerMetadata
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .metadata.image_shape import ImageShape
from .metadata.region_2d import Region2D



class AICSImageIoServer(ImageServer):
    """
    An ImageServer using AICSImageIO (https://github.com/AllenCellModeling/aicsimageio).

    What this actually supports will depend upon how AICSImageIO is installed.
    For example, it may provide Bio-Formats or CZI support... or it may not.
    Note that the AICSImage library does not currently handle unit attachment, so the pixel unit
    given by this server will always be 'pixels'.
    Note that the AICSImage library does not properly support pyramids, so you might only get the full
    resolution image when opening a pyramidal image.
    """

    def __init__(self, path: str, scene: int = 0, detect_resolutions=True, aics_kwargs: dict[str, any] = {}, **kwargs):
        """
        Create the server.

        :param path: the local path to the image to open
        :param scene: AICSImageIO divides images into scene. This parameter specifies which scene to consider
        :param detect_resolutions: whether to look at all resolutions of the image (instead of just the full resolution)
        :param aics_kwargs: any specific keyword arguments to pass down to the fsspec created filesystem handled by the AICSImageIO reader
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(**kwargs)
        self._path = path
        self._reader = AICSImage(path, dask_tiles=True, **aics_kwargs)
        self._scene = scene
        self._detect_resolutions = detect_resolutions

    def _build_metadata(self) -> ImageServerMetadata:
        return ImageServerMetadata(
            self._path,
            Path(self._path).name,
            self._get_shapes(self._reader, self._scene) if self._detect_resolutions else (self._get_scene_shape(self._reader, self._reader.scenes[self._scene]),),
            self._get_pixel_calibration(self._reader),
            self._is_rgb(self._reader),
            np.dtype(self._reader.dtype)
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        x, y, width, height, z, t = astuple(region)
        
        self._reader.set_scene(self._reader.scenes[self._scene + level])
        axes = "TZYX" + ("S" if "S" in self._reader.dims.order else "C")

        return self._reader.get_image_dask_data(axes)[t, z, y:y + height, x:x + width, ...].compute()

    def close(self):
        self._reader.close()

    @staticmethod
    def _get_shapes(reader: AICSImage, scene: int) -> tuple[ImageShape, ...]:
        shapes = []
        for scene in reader.scenes[scene:]:
            shape = AICSImageIoServer._get_scene_shape(reader, scene)

            if len(shapes) == 0 or AICSImageIoServer._is_lower_resolution(shapes[-1], shape):
                shapes.append(shape)
            else:
                break
        return tuple(shapes)

    @staticmethod
    def _get_scene_shape(reader: AICSImage, scene: int) -> ImageShape:
        reader.set_scene(scene)

        return ImageShape(
            x=reader.dims.X if 'X' in reader.dims.order else 1,
            y=reader.dims.Y if 'Y' in reader.dims.order else 1,
            z=reader.dims.Z if 'Z' in reader.dims.order else 1,
            c=reader.dims.S if 'S' in reader.dims.order else (reader.dims.C if 'C' in reader.dims.order else 1),
            t=reader.dims.T if 'T' in reader.dims.order else 1,
        )

    @staticmethod
    def _get_pixel_calibration(reader: AICSImage) -> PixelCalibration:
        sizes = reader.physical_pixel_sizes

        if sizes.X or sizes.Y or sizes.Z:
            # The AICSImage library does not currently handle unit attachment, so the pixel unit is returned
            return PixelCalibration(
                PixelLength(sizes.X) if sizes.X is not None else PixelLength(),
                PixelLength(sizes.Y) if sizes.Y is not None else PixelLength(),
                PixelLength(sizes.Z) if sizes.Z is not None else PixelLength()
            )
        else:
            return PixelCalibration()
    
    @staticmethod
    def _is_rgb(reader: AICSImage) -> bool:
        return ('S' in reader.dims.order and reader.dims.S in [3, 4]) or (reader.dtype == np.uint8 and reader.dims.C == 3)

    @staticmethod
    def _is_lower_resolution(base_shape: ImageShape, series_shape: ImageShape) -> bool:
        """
        Calculate if the series shape is a lower resolution than the base shape.

        This involves a bit of guesswork, but it's needed for so long as AICSImageIO doesn't properly support pyramids.
        """
        if base_shape.z == series_shape.z and \
                base_shape.t == series_shape.t and \
                base_shape.c == series_shape.c:
            
            x_ratio = series_shape.x / base_shape.x
            y_ratio = series_shape.y / base_shape.y
            return x_ratio < 1.0 and math.isclose(x_ratio, y_ratio, rel_tol=0.01)
        else:
            return False

