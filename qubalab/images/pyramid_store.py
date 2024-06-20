from argparse import ArgumentError
from typing import Union, Callable
from operator import itemgetter
import numpy as np
import traceback as tb
from zarr.storage import Store, init_group, init_array, attrs_key
from zarr.util import json_dumps
from .metadata.pixel_calibration import PixelCalibration
from .metadata.region_2d import Region2D
from .metadata.image_server_metadata import ImageServerMetadata


OME_NGFF_SPECIFICATION_VERSION = "0.4"


class PyramidStore(Store):
    """
    A zarr store to open an image with https://github.com/zarr-developers/zarr-python.

    This class is a Zarr store that represents a zarr group, this one containing
    one or more zarr arrays (depending on the number of resolutions of the image).
    You can use it with zarr.Group(store=PyramidStore(...)) (see the sample notebooks
    or the unit tests for examples).

    Pixels of zarr arrays of this store can be accessed with the following order:
    (t, c, z, y, x). There may be less dimensions for simple images.

    This store is read-only, so functions like zarr.create won't work.

    The Zarr arrays will be divided into chunks whose width and height can be specified.
    The size of chunks along the channel axis will be the total number of channels,
    while the size of chunks along the z-axis and the time-axis will be 1.
    The width and heights of chunks cannot be greater than the width and height
    of the image.

    The returned zarr will conform to version 0.4 of the OME-NGFF specifications
    (see https://ngff.openmicroscopy.org/0.4/). These specifications define
    the Zarr hierarchy and some metadata.

    Inspired by https://github.com/manzt/napari-lazy-openslide
    Copyright (c) 2020, Trevor Manz (BSD-3-clause)
    """

    def __init__(
        self,
        metadata: ImageServerMetadata,
        region_reader: Callable[[int, Region2D], np.ndarray],
        chunk_size: Union[int, tuple[int, int]] = None,
        name: str = None,
        downsamples: tuple[float, ...] = None,
        squeeze=True
    ):
        """
        Create a Zarr-compatible mapping for a multiresolution image reading pixels from an ImageServer.

        :param metadata: the metadata of the image
        :param region_reader: a function that takes a level and a region as parameters, and return pixels
                              of the image belonging to that level and region as a numpy array
        :param chunk_size: the size of the chunks that divide the image. A tuple to set the width/height
                           of the chunks or an integer to set the same value for width/height.
                           By default, chunks will have a size of (y=1024, x=1024)
        :param name: the name of the image. By default, this is the name contained in the image metadata
        :param downsamples: the downsamples of the image. By default, the downsamples of the image server
                            will be used
        :param squeeze: whether to remove dimensions of the image that have only one value (x and y axis
                        not included). For example, if this parameter is True, then an image of dimensions
                        (t=1,c=5,z=2,y=40,x=1) will become an image with dimensions (c=5,z=2,y=40,x=1)
                        (the time dimension was removed, but not the x axis)
        """

        super().__init__()
        self._metadata = metadata
        self._region_reader = region_reader

        if chunk_size is None:
            self._chunk_size = (1024, 1024)
        elif isinstance(chunk_size, int):
            self._chunk_size = (chunk_size, chunk_size)
        else:
            self._chunk_size = chunk_size

        if downsamples is None:
            self._downsamples = self._metadata.downsamples
        else:
            self._downsamples = downsamples

        if squeeze:
            dimensions_to_keep = itemgetter(*[i for i, d in enumerate(self._metadata.shape.as_tuple()) if d > 1 or i > 2])
        else:
            dimensions_to_keep = itemgetter(0, 1, 2, 3, 4)
        
        self._dims = ''.join(dimensions_to_keep('tczyx'))

        self._root = PyramidStore._create_root(
            self._metadata.name if name is None else name,
            dimensions_to_keep,
            self._downsamples,
            self._metadata.pixel_calibration
        )

        PyramidStore._create_arrays(
            self._root,
            self._metadata,
            dimensions_to_keep,
            self._downsamples,
            self._chunk_size
        )

    def get_path_of_level(self, level: int):
        """
        Return the path of a level within the root group described by this store.

        :param level: the pyramid level (0 is full resolution). Must be less than the number
                      of resolutions of the image
        :returns: the path of the provided level within the root group
        :raises ValueError: when level is not valid
        """
        max_level = len(self._root[".zattrs"]["multiscales"][0]["datasets"]) - 1
        if level < 0 or level > max_level:
            raise ValueError("The provided level ({0}) is outside the valid range ([0, {1}])".format(level, max_level))
        
        return self._root[attrs_key]["multiscales"][0]["datasets"][level]["path"]

    def __getitem__(self, key: str):
        # Check if key is in metadata
        if key in self._root:
            return self._root[key]
        
        try:
            # We should have a chunk path, with a chunk level
            ct, cz, cy, cx, level = self._parse_chunk_path(key)

            # Convert the chunk level a downsample value & also to a server level
            downsample = self._downsamples[level]
            level_width = int(self._metadata.width / downsample)
            level_height = int(self._metadata.height / downsample)
            tile_width = min(level_width, self._chunk_size[0])
            tile_height = min(level_height, self._chunk_size[1])

            x = int(round(cx * tile_width * downsample))
            y = int(round(cy * tile_height * downsample))
            w = int(min(self._metadata.width - x, round(tile_width * downsample)))
            h = int(min(self._metadata.height - y, round(tile_height * downsample)))
            tile = self._region_reader(level, Region2D(x, y, w, h, cz, ct))

            # Chunks should be the same size (according to zarr spec), so pad if needed
            pad_y = tile_height - tile.shape[0]
            pad_x = tile_width - tile.shape[1]
            if pad_y > 0 or pad_x > 0:
                pad_dims = tile.ndim - 2
                if pad_dims > 0:
                    pad = ((0, pad_y), (0, pad_x), (0, 0) * pad_dims)
                else:
                    pad = ((0, pad_y), (0, pad_x))
                tile = np.pad(tile, pad)

            # Ensure channels first
            if tile.ndim == 3:
                tile = np.moveaxis(tile, -1, 0)
            elif tile.ndim > 3:
                raise ValueError(f'ndim > 3 not supported! Found shape {tile.shape}')

            return tile.tobytes()
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            print(err)
            tb.print_exc(limit=2)
            raise err
        except Exception as err:
            print('Something unexpected has happened: ' + str(err))
            tb.print_exc(limit=2)
            raise KeyError(key)

    def __contains__(self, key: str):
        return key in self._root

    def __eq__(self, other):
        return (
                isinstance(other, PyramidStore)
                and self._metadata == other._metadata
        )

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(1 for _ in self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __setitem__(self, key, val):
        raise RuntimeError("This store is read-only")

    def __delitem__(self, key):
        raise RuntimeError("This store is read-only")

    def keys(self):
        return self._root.keys()

    @staticmethod
    def _create_root(
        name: str,
        dimensions_to_keep: itemgetter,
        downsamples: tuple[float, ...],
        pixel_calibration: PixelCalibration
    ) -> dict[str, bytes]:
        """
        Create the zarr group that should be at the root of the image files
        according to the OME-NGFF specifications.

        :param dtype: the type of the pixel values of the image
        :param name: the name of the image
        :param dimensions_to_keep: a callable object that returns only the dimensions to keep
                                   when called on a (t,c,z,y,x) iterable. For example, if only the
                                   (t,y,x) dimensions must be kept, dimensions_to_keep([1, 2, 3, 4, 5])
                                   should return (1, 4, 5)
        :param channels: the channels of the image
        :param downsamples: the downsamples of the image
        :param pixel_calibration: the pixel calibration of the image
        :return: a dictionnary of attributes for this image
        """

        group = dict()
        init_group(group)

        group[attrs_key] = dict(
            multiscales=[
                dict(
                    name=name,
                    datasets=[
                        dict(
                            path=str(downsample_index),
                            coordinateTransformations=[dict(
                                type='scale',
                                scale=dimensions_to_keep([
                                    1.0,
                                    1.0,
                                    1.0,
                                    downsample,
                                    downsample
                                ])
                            )]
                        ) for downsample_index, downsample in enumerate(downsamples)
                    ],
                    version=OME_NGFF_SPECIFICATION_VERSION,
                    axes=dimensions_to_keep([
                        dict(name='t', type='time'),
                        dict(name='c', type='channel'),
                        dict(name='z', type='space', unit=pixel_calibration.length_z.unit),
                        dict(name='y', type='space', unit=pixel_calibration.length_y.unit),
                        dict(name='x', type='space', unit=pixel_calibration.length_x.unit),
                    ])
                )
            ]
        )
        return group

    @staticmethod
    def _create_arrays(
        group: dict[str, bytes],
        metadata: ImageServerMetadata,
        dimensions_to_keep: itemgetter,
        downsamples: tuple[float, ...],
        chunk_size: tuple[int, int]
    ):
        """
        Create the zarr arrays that represent the levels of the image, and add them
        to the provided group.

        :param group: the group to add the zarr arrays to
        :param metadata: the metadata of the image
        :param dimensions_to_keep: a callable object that returns only the dimensions to keep
                                   when called on a (t,c,z,y,x) iterable. For example, if only the
                                   (t,y,x) dimensions must be kept, dimensions_to_keep([1, 2, 3, 4, 5])
                                   should return (1, 4, 5)
        :param downsamples: the downsamples of the image
        :param chunk_size: the size (width, height) of the chunks that divide the image
        """

        for downsample_index, downsample in enumerate(downsamples):
            init_array(
                group,
                path=str(downsample_index),
                shape=dimensions_to_keep((
                    metadata.n_timepoints,
                    metadata.n_channels,
                    metadata.n_z_slices,
                    int(metadata.height / downsample),
                    int(metadata.width / downsample)
                )),
                chunks=dimensions_to_keep((
                    1,
                    metadata.n_channels,
                    1,
                    min(int(metadata.height / downsample), chunk_size[1]),
                    min(int(metadata.width / downsample), chunk_size[0])
                )),
                dtype=metadata.dtype,
                compressor=None
            )

    def _parse_chunk_path(self, path: str) -> tuple[int, int, int, int, int]:
        level, chunk = path.split('/')
        chunks = tuple(map(int, chunk.split('.')))
        return tuple([self._extract_chunk_path(chunks, 't'),
                      self._extract_chunk_path(chunks, 'z'),
                      chunks[-2],
                      chunks[-1],
                      int(level)])

    def _extract_chunk_path(self, lengths: tuple[int, ...], target: str):
        if target in self._dims:
            return lengths[self._dims.index(target)]
        return 0
