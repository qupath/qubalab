from argparse import ArgumentError
from typing import Union, Iterable
from operator import itemgetter
import math
import numpy as np
import traceback as tb

from zarr.storage import BaseStore, init_group, init_array, attrs_key
from zarr.util import json_dumps

from .servers import ImageServer, Region2D, _get_level

from .metadata.pixel_calibration import PixelCalibration
from .metadata.image_channel import ImageChannel


OME_NGFF_SPECIFICATION_VERSION = "0.4"


class PyramidStore(BaseStore):
    """
    A wrapper to open an ImageServer with https://github.com/zarr-developers/zarr-python.

    This class is a Zarr store, which means it can be used by specifying it in the
    zarr.open_array function for example.

    This store is read-only, so functions like zarr.create won't work.

    The Zarr arrays will be divided into chunks whose width and height can be specified.
    The size of chunks along the channel axis will be the total number of channels,
    while the size of chunks along the z-axis and the time-axis will be 1.

    The returned zarr will conform to version 0.4 of the OME-NGFF specifications
    (see https://ngff.openmicroscopy.org/0.4/). These specifications define
    the Zarr hierarchy and some metadata.

    Inspired by https://github.com/manzt/napari-lazy-openslide
    Copyright (c) 2020, Trevor Manz (BSD-3-clause)
    """

    def __init__(
        self,
        server: ImageServer,
        tile_size: Union[int, tuple[int, int]] = None,
        name: str = None,
        downsamples: tuple[float, ...] = None,
        squeeze=True
    ):
        """
        Create a Zarr-compatible mapping for a multiresolution image reading pixels from an ImageServer.

        :param server: the image server to open
        :param tile_size: the size of the chunks that divide the image. A tuple to set the width/height
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
        self._server = server

        if tile_size is None:
            self._tile_size = (1024, 1024)
        elif isinstance(tile_size, int):
            self._tile_size = (tile_size, tile_size)
        else:
            self._tile_size = tile_size

        if downsamples is None:
            self._downsamples = server.metadata.downsamples
        else:
            self._downsamples = downsamples

        if squeeze:
            dimensions_to_keep = itemgetter(*[i for i, d in enumerate(self._server.metadata.shape.as_tuple()) if d > 1 or i > 2])
        else:
            dimensions_to_keep = itemgetter(0, 1, 2, 3, 4)
        
        self._dims = ''.join(dimensions_to_keep('tczyx'))

        self._store = PyramidStore._create_group(
            self._server.dtype,
            self._server.metadata.name if name is None else name,
            dimensions_to_keep,
            self._server.metadata.channels,
            self._downsamples,
            self._server.metadata.pixel_calibration
        )

        PyramidStore._create_arrays(
            self._store,
            self._server,
            dimensions_to_keep,
            self._downsamples,
            self._tile_size
        )

    def __getitem__(self, key: str):
        # Check if key is in metadata
        if key in self._store:
            return self._store[key]
        try:
            # We should have a chunk path, with a chunk level
            ct, cz, cy, cx, level = self._parse_chunk_path(key)

            # Convert the chunk level a downsample value & also to a server level
            downsample = self._downsamples[level]
            full_width = int(self._server.width / downsample)
            full_height = int(self._server.height / downsample)
            server_level = _get_level(self._server.downsamples, downsample)
            tile_width = min(full_width, self._tile_size[0])
            tile_height = min(full_height, self._tile_size[1])

            if math.isclose(self._server.downsamples[server_level], downsample, abs_tol=1e-3):
                # If our downsample value is close to what the server can provide directly, use read_block & level
                x = int(cx * tile_width)
                y = int(cy * tile_height)
                w = int(min(full_width - x, tile_width))
                h = int(min(full_height - y, tile_height))
                block = (x, y, w, h, cz, ct)
                tile = self._server.read_block(level=server_level, block=block)
            else:
                # If our downsample value is anything else, use read_region to auto-apply whatever resizing we need (shouldn't be used!)
                x = int(round(cx * tile_width * downsample))
                y = int(round(cy * tile_height * downsample))
                w = int(min(self._server.width - x, round(tile_width * downsample)))
                h = int(min(self._server.height - y, round(tile_height * downsample)))
                region = Region2D(downsample=downsample, x=x, y=y, width=w, height=h, z=cz, t=ct)
                tile = self._server.read_region(region=region)

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
        return key in self._store

    def __eq__(self, other):
        return (
                isinstance(other, PyramidStore)
                and self._server == other._server
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
        raise RuntimeError("__setitem__ not implemented")

    def __delitem__(self, key):
        raise RuntimeError("__setitem__ not implemented")

    def keys(self):
        return self._store.keys()

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

    @staticmethod
    def _create_group(
        dtype: np.dtype,
        name: str,
        dimensions_to_keep: itemgetter[tuple[int, ...]],
        channels: tuple[ImageChannel, ...],
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
        
        if issubclass(dtype, np.integer):
            max_value = np.iinfo(dtype).max
        else:
            max_value = np.finfo(dtype).max

        group[attrs_key] = json_dumps(dict(
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
            ],
            omero= dict(
                name=name,
                version=OME_NGFF_SPECIFICATION_VERSION,
                channels=[
                    dict(
                        active=True,
                        coefficient=1,
                        color="{0:X}{1:X}{2:X}".format(int(channel.color[0] * 255), int(channel.color[1] * 255), int(channel.color[2] * 255)),
                        family="linear",
                        inverted=False,
                        label=channel.name,
                        window=dict(
                            start=0,
                            end=max_value,
                            min=0,
                            max=max_value
                        )
                    ) for channel in channels
                ],
                rdefs=dict(
                    defaultT=0,
                    defaultZ=0,
                    model="color"
                )
            )
        ))
        return group

    @staticmethod
    def _create_arrays(
        group: dict[str, bytes],
        server: ImageServer,
        dimensions_to_keep: itemgetter[tuple[int, ...]],
        downsamples: tuple[float, ...],
        tile_size: tuple[int, int]
    ):
        """
        Create the zarr arrays that represent the levels of the image, and add them
        to the provided group.

        :param group: the group to add the zarr arrays to
        :param server: the image server to represent
        :param dimensions_to_keep: a callable object that returns only the dimensions to keep
                                   when called on a (t,c,z,y,x) iterable. For example, if only the
                                   (t,y,x) dimensions must be kept, dimensions_to_keep([1, 2, 3, 4, 5])
                                   should return (1, 4, 5)
        :param downsamples: the downsamples of the image
        :param tile_size: the size (width, height) of the chunks that divide the image
        """

        for downsample_index, downsample in enumerate(downsamples):
            init_array(
                group,
                path=str(downsample_index),
                shape=dimensions_to_keep((
                    server.metadata.n_timepoints,
                    server.metadata.n_channels,
                    server.metadata.n_z_slices,
                    int(server.metadata.height / downsample),
                    int(server.metadata.width / downsample)
                )),
                chunks=dimensions_to_keep((
                    1,
                    server.metadata.n_channels,
                    1,
                    min(int(server.metadata.height / downsample), tile_size[1]),
                    min(int(server.metadata.width / downsample), tile_size[0])
                )),
                dtype=server.dtype,
                compressor=None
            )