import numbers

import dask.array

from . import ImageServer, ImageServerMetadata, Region2D, ImageShape
from .servers import _validate_block, PixelCalibration, PixelLength, _get_level

from typing import Tuple, Union, Dict, Any, Iterable
from pathlib import Path
from aicsimageio import AICSImage

import numpy as np
import math


class AICSImageIoServer(ImageServer):
    """
    An ImageServer using AICSImageIO.

    What this actually supports will depend upon how AICSImageIO is installed.
    For example, it may provide Bio-Formats or CZI support... or it may not.
    """

    def __init__(self, path: str, strip_alpha=True, scene: int = 0, detect_resolutions=True,
                 aics_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__(**kwargs)
        self._path = path
        self._strip_alpha = strip_alpha
        self._image = AICSImage(path, dask_tiles=True, **aics_kwargs)
        self._scene = scene
        self._detect_resolutions = detect_resolutions
        self._dask_arrays = None

        # Build dask arrays for each level now
        metadata = self.metadata
        dask_arrays = []
        for level in range(len(metadata.shapes)):
            image = AICSImage(path, dask_tiles=True, **aics_kwargs)
            image.set_scene(image.scenes[self._scene + level])
            if self.is_rgb:
                im = image.get_image_dask_data('TZYXS', C=0)
            elif self.n_channels == 1:
                im = image.get_image_dask_data('TZYX', C=0)
            else:
                im = image.get_image_dask_data('TZYXC')
            dask_arrays.append((image, im))
        self._dask_arrays = dask_arrays

    def _build_metadata(self) -> ImageServerMetadata:
        image = self._image
        name = Path(self._path).name if self._path else image.current_scene
        image.set_scene(image.scenes[self._scene])
        metadata = ImageServerMetadata(
            path=self._path,
            name=name,
            shapes=(_get_current_scene_shape(image),) if not self._detect_resolutions else _get_shapes(image,
                                                                                                       self._scene),
            dtype=image.dtype,
            pixel_calibration=_get_pixel_calibration(image),
            is_rgb='S' in image.dims.order and image.dims['S'][0] in [3, 4]
        )
        return metadata

    def read_block(self, level: int, block: Union[Region2D, Tuple[int, ...]]) -> np.ndarray:
        from dataclasses import astuple
        _, x, y, width, height, z, t = astuple(_validate_block(block))
        im = self._dask_arrays[level][1]
        return im[t, z, y:y + height, x:x + width, ...].compute()


    def get_dask_arrays(self, downsamples: Union[float, Iterable[float]] = None) -> Tuple[dask.array.Array, ...]:
        if downsamples is None:
            return tuple([da[1] for da in self._dask_arrays])
        elif isinstance(downsamples, numbers.Number):
            downsample = downsamples
            level = _get_level(self.downsamples, downsample)
            array = self._dask_arrays[level][1]
            rescale = downsample / self.downsamples[level]
            if rescale == 1.0:
                return (array,)
            else:
                # Couldn't find an easy resizing method for dask arrays... so we try this instead
                from dask_image.ndinterp import affine_transform
                from dask import array as da
                array = da.asarray(array)
                transform = np.eye(array.ndim)
                transform[2, 2] = rescale
                transform[3, 3] = rescale
                output_width = int(array.shape[3] / rescale)
                output_height = int(array.shape[2] / rescale)
                # Not sure why rechunking is necessary here, but it is
                array = da.rechunk(array)
                array = affine_transform(array,
                                         transform,
                                         order=1,
                                         output_chunks=array.chunks)
                return (array[..., :output_height, :output_width, :],)
        else:
            return tuple([self.get_dask_arrays(d)[0] for d in downsamples])


    def close(self):
        self._image.close()


def _get_shapes(image: AICSImage, first_scene: int = 0) -> Tuple[ImageShape, ...]:
    shapes = []
    for scene in image.scenes[first_scene:]:
        image.set_scene(scene)
        shape = _get_current_scene_shape(image)
        if len(shapes) == 0 or _is_lower_resolution(shapes[-1], shape):
            shapes.append(shape)
        else:
            break
    return tuple(shapes)


def _get_current_scene_shape(image: AICSImage) -> ImageShape:
    dims = image.dims
    return ImageShape(
        x=dims['X'][0] if 'X' in dims.order else 1,
        y=dims['Y'][0] if 'Y' in dims.order else 1,
        z=dims['Z'][0] if 'Z' in dims.order else 1,
        c=dims['S'][0] if 'S' in dims.order else (dims['C'][0] if 'C' in dims.order else 1),
        t=dims['T'][0] if 'T' in dims.order else 1,
    )


def _is_lower_resolution(base_shape: ImageShape, series_shape: ImageShape) -> bool:
    if base_shape.z == series_shape.z and \
            base_shape.t == series_shape.t and \
            base_shape.c == series_shape.c:
        x_ratio = series_shape.x / base_shape.x
        y_ratio = series_shape.y / base_shape.y
        return x_ratio < 1.0 and math.isclose(x_ratio, y_ratio, rel_tol=0.01)
    return False


def _get_pixel_calibration(image: AICSImage) -> PixelCalibration:
    sizes = image.physical_pixel_sizes
    if sizes.X or sizes.Y or sizes.Z:
        # TODO: We make a big assumption that any available values are in µm!
        return PixelCalibration(
            x=PixelLength(sizes.X) if sizes.X is not None else PixelLength(),
            y=PixelLength(sizes.Y) if sizes.Y is not None else PixelLength(),
            z=PixelLength(sizes.Z) if sizes.Z is not None else PixelLength()
        )
    else:
        return PixelCalibration()

