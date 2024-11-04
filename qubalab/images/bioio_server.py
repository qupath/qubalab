import numpy as np
import dask.array as da
import math
from pathlib import Path
from bioio import BioImage
from dataclasses import astuple
from .image_server import ImageServer
from .metadata.image_metadata import ImageMetadata
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .metadata.image_shape import ImageShape
from .region_2d import Region2D


class BioIOServer(ImageServer):
    """
    An ImageServer using BioIO (https://github.com/AllenCellModeling/BioIO).

    What this actually supports will depend upon how BioIO is installed.
    For example, it may provide Bio-Formats or CZI support... or it may not.

    Note that the BioIo library does not currently handle unit attachment, so the pixel unit
    given by this server will always be 'pixels'.
    
    Note that the BioIO library does not properly support pyramids, so you might only get the full
    resolution image when opening a pyramidal image.
    """

    def __init__(self, path: str, scene: int = 0, detect_resolutions=True, bioio_kwargs: dict[str, any] = {}, **kwargs):
        """
        :param path: the local path to the image to open
        :param scene: BioIO divides images into scene. This parameter specifies which scene to consider
        :param detect_resolutions: whether to look at all resolutions of the image (instead of just the full resolution)
        :param bioio_kwargs: any specific keyword arguments to pass down to the fsspec created filesystem handled by the BioIO reader
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(**kwargs)
        self._path = path
        self._reader = BioImage(path, **bioio_kwargs)
        self._scene = scene
        self._detect_resolutions = detect_resolutions

    def level_to_dask(self, level: int = 0) -> da.Array:
        if level < 0 or level >= self.metadata.n_resolutions:
            raise ValueError("The provided level ({0}) is outside the valid range ([0, {1}])".format(level, self.metadata.n_resolutions - 1))

        axes = ("T" if self.metadata.n_timepoints > 1 else "") + \
            (("S" if "S" in self._reader.dims.order else "C") if self.metadata.n_channels > 1 else "") + \
            ("Z" if self.metadata.n_z_slices > 1 else "") + \
            "YX"

        self._reader.set_scene(self._reader.scenes[self._scene + level])
        return self._reader.get_image_dask_data(axes)

    def close(self):
        pass

    def _build_metadata(self) -> ImageMetadata:
        return ImageMetadata(
            self._path,
            Path(self._path).name,
            self._get_shapes(self._reader, self._scene) if self._detect_resolutions else (self._get_scene_shape(self._reader, self._reader.scenes[self._scene]),),
            self._get_pixel_calibration(self._reader, self._scene),
            self._is_rgb(self._reader, self._scene),
            np.dtype(self._reader.dtype)
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        x, y, width, height, z, t = astuple(region)
        
        self._reader.set_scene(self._reader.scenes[self._scene + level])
        axes = "TZ" + ("S" if "S" in self._reader.dims.order else "C") + "YX"

        return self._reader.get_image_dask_data(axes)[t, z, :, y:y + height, x:x + width].compute()

    @staticmethod
    def _get_shapes(reader: BioImage, scene: int) -> tuple[ImageShape, ...]:
        shapes = []
        for scene in reader.scenes[scene:]:
            shape = BioIOServer._get_scene_shape(reader, scene)

            if len(shapes) == 0 or BioIOServer._is_lower_resolution(shapes[-1], shape):
                shapes.append(shape)
            else:
                break
        return tuple(shapes)

    @staticmethod
    def _get_scene_shape(reader: BioImage, scene: int) -> ImageShape:
        reader.set_scene(scene)

        return ImageShape(
            x=reader.dims.X if 'X' in reader.dims.order else 1,
            y=reader.dims.Y if 'Y' in reader.dims.order else 1,
            z=reader.dims.Z if 'Z' in reader.dims.order else 1,
            c=reader.dims.S if 'S' in reader.dims.order else (reader.dims.C if 'C' in reader.dims.order else 1),
            t=reader.dims.T if 'T' in reader.dims.order else 1,
        )

    @staticmethod
    def _get_pixel_calibration(reader: BioImage, scene: int) -> PixelCalibration:
        reader.set_scene(scene)
        sizes = reader.physical_pixel_sizes

        if sizes.X or sizes.Y or sizes.Z:
            # The bioio library does not currently handle unit attachment, so the pixel unit is returned
            return PixelCalibration(
                PixelLength(sizes.X) if sizes.X is not None else PixelLength(),
                PixelLength(sizes.Y) if sizes.Y is not None else PixelLength(),
                PixelLength(sizes.Z) if sizes.Z is not None else PixelLength()
            )
        else:
            return PixelCalibration()
    
    @staticmethod
    def _is_rgb(reader: BioImage, scene: int) -> bool:
        reader.set_scene(scene)
        return ('S' in reader.dims.order and reader.dims.S in [3, 4]) or (reader.dtype == np.uint8 and reader.dims.C == 3)

    @staticmethod
    def _is_lower_resolution(base_shape: ImageShape, series_shape: ImageShape) -> bool:
        """
        Calculate if the series shape is a lower resolution than the base shape.

        This involves a bit of guesswork, but it's needed for so long as BioIO doesn't properly support pyramids.
        """
        if base_shape.z == series_shape.z and \
                base_shape.t == series_shape.t and \
                base_shape.c == series_shape.c:
            
            x_ratio = series_shape.x / base_shape.x
            y_ratio = series_shape.y / base_shape.y
            return x_ratio < 1.0 and math.isclose(x_ratio, y_ratio, rel_tol=0.01)
        else:
            return False

