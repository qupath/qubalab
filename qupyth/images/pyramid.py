from argparse import ArgumentError
from typing import Union, Iterable, Tuple, Dict
import uuid
import math
import numpy as np
import traceback as tb

import zarr
from zarr.storage import BaseStore, init_group, init_array, attrs_key
from zarr.util import json_dumps

from .servers import ImageServer, Region2D, _get_level


class PyramidStore(BaseStore):
    """
    Zarr-compatible wrapper for an ImageServer.

    Inspired by https://github.com/manzt/napari-lazy-openslide
       Copyright (c) 2020, Trevor Manz (BSD-3-clause)
    """

    def __init__(self, server: ImageServer, tilesize: int = 512, name: str = None,
                 downsamples: Union[float, Iterable[float]] = None, z=0, t=0, pad_chunks=True):
        """
        Create a Zarr-compatible mapping for a multiresolution image reading pixels from an ImageServer.

        Note that the details of this implementation may change to better align with any future standardized representation
        of pyramidal images.
        In general, if the desired outcome is something array-like it is best to use to_dask instead.


        :param server:
        :param tilesize:
        :param name:
        :param downsamples:
        :param z:
        :param t:
        :param pad_chunks: pad to the preferred chunk size, even at the edge of the image
        """
        super().__init__()
        self._server = server
        self._tilesize = tilesize
        if downsamples is None:
            downsamples = server.downsamples
        elif not isinstance(downsamples, Iterable):
            downsamples = (downsamples,)
        if name is None:
            name = str(uuid.uuid1())
        self._downsamples = downsamples
        self._z = z
        self._t = t
        self._store = _build_store(server, downsamples=downsamples, tilesize=tilesize, name=name)
        self._pad_chunks = pad_chunks

    def __getitem__(self, key: str):
        # Check if key is in metadata
        if key in self._store:
            return self._store[key]
        try:
            # We should have a chunk path, with a chunk level
            inds, level = _parse_chunk_path(key)
            cy = inds[0]
            cx = inds[1]
            cz = 0
            ct = 0
            if len(inds) > 3:
                cz = inds[3]
            if len(inds) > 4:
                ct = inds[4]
                
            # Convert the chunk level a downsample value & also to a server level
            downsample = self._downsamples[level]
            full_width = int(self._server.width / downsample)
            full_height = int(self._server.height / downsample)
            server_level = _get_level(self._server.downsamples, downsample)
            tile_width = min(full_width, self._tilesize)
            tile_height = min(full_height, self._tilesize)

            if math.isclose(self._server.downsamples[server_level], downsample, abs_tol=1e-3):
                # If our downsample value is close to what the server can provide directly, use read_block & level
                x = int(cx * tile_width)
                y = int(cy * tile_height)
                w = int(min(full_width - x, tile_width))
                h = int(min(full_height - y, tile_height))
                block = (x, y, w, h, cz + self._z, ct + self._t)
                tile = self._server.read_block(level=server_level, block=block)
            else:
                # If our downsample value is anything else, use read_region to auto-apply whatever resizing we need (shouldn't be used!)
                x = int(cx * tile_width * downsample),
                y = int(cy * tile_height * downsample),
                w = int(min(full_width - x, round(tile_width * downsample))),
                h = int(min(full_height - y, round(tile_height * downsample))),
                region = Region2D(downsample=downsample, x=x, y=y, width=w, height=h, z=self._z, t=self._t)
                tile = self._server.read_region(region=region)

            # Pad to chunk size if needed
            if self._pad_chunks:
                pad_y = tile_height - tile.shape[0]
                pad_x = tile_width - tile.shape[1]
                if pad_y > 0 or pad_x > 0:
                    pad_dims = tile.ndim - 2
                    if pad_dims > 0:
                        pad = ((0, pad_y), (0, pad_x), (0, 0) * pad_dims)
                    else:
                        pad = ((0, pad_y), (0, pad_x))
                    tile = np.pad(tile, pad)

            return np.array(tile).tobytes()
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



def to_zarr(image: Union[ImageServer, PyramidStore], **kwargs):
    """
    Create & open a read-only Zarr group containing multiresolution data for an ImageServer.

    Note that the details of this implementation may change to better align with any future standardized representation
    of pyramidal images.
    In general, if the desired outcome is something array-like it is best to use to_dask instead.

    :param image:  the image containing pixels
    :param kwargs: passed to PyramidStore if the input is an ImageServer
    :return:       a Zarr group with a 'multiscales' attribute corresponding to any requested downsamples
    """
    if isinstance(image, PyramidStore):
        store = image
    elif isinstance(image, ImageServer):
        store = PyramidStore(image, **kwargs)
    else:
        raise ValueError(f'Unable to convert object of type {type(image)} to Zarr group - '
                         f'only ImageServer and PyramidStore supported')
    return zarr.open(store, mode='r')



def to_dask(image: Union[ImageServer, PyramidStore], **kwargs):
    """
    Create one or more dask arrays for an ImageServer.
    This provides a more pythonic/numpy-esque method to extract pixel data at any arbitrary resolution.

    Internally, the conversion uses a Zarr group, opened in read-only mode.

    :param image:  the image containing pixels
    :param kwargs: passed to PyramidStore if the input is an ImageServer
    :return:       a single dask array if the keyword argument 'downsamples' is a number, or a tuple of dask arrays if
                   'downsamples' is an iterable
    """
    if isinstance(image, PyramidStore):
        store = image
    elif isinstance(image, ImageServer):
        store = PyramidStore(image, **kwargs)
    else:
        raise ValueError(f'Unable to convert object of type {type(image)} to Dask array - '
                         f'only ImageServer and PyramidStore supported')
    from dask import array as da
    grp = to_zarr(store, **kwargs)
    multiscales = grp.attrs["multiscales"][0]
    pyramid = tuple(da.from_zarr(store, component=d["path"]) for d in multiscales["datasets"])
    # If we requested a single downsample, then return it directly (not in a tuple)
    if len(pyramid) == 1 and 'downsamples' in kwargs:
        if not isinstance(kwargs['downsamples'], Iterable) and kwargs['downsamples'] is not None:
            return pyramid[0]
    return pyramid



def _build_store(server: ImageServer, downsamples: Iterable[float], tilesize: int = 512, name: str = None) -> Dict[str, bytes]:
    """
    Build Zarr storage.

    :param server:
    :param downsamples:
    :param tilesize:
    :param name:
    :return:
    """

    if name is None:
        name = server.name
    # TODO: Consider support for single scale?
    root_attrs = dict(
        multiscales=[
            dict(
                name=name,
                datasets=[{'path': str(ii)} for ii in range(len(downsamples))],
                version=0.1
            )
        ]
    )
    store = dict()
    init_group(store)
    store[attrs_key] = json_dumps(root_attrs)
    for ii, downsample in enumerate(downsamples):
        w = int(server.width / downsample)
        h = int(server.height / downsample)
        print(f'{w} x {h}')
        if server.n_z_slices > 1 or server.n_timepoints > 1:
            shape = (h, w, server.n_channels, server.n_z_slices, server.n_timepoints)
            chunks = (min(h, tilesize), min(w, tilesize), server.n_channels, 1, 1)
        else:
            shape = (h, w, server.n_channels)
            chunks = (min(h, tilesize), min(w, tilesize), server.n_channels)
        init_array(
            store,
            path=str(ii),
            shape=shape,
            chunks=chunks,
            dtype=server.dtype,
            compressor=None
        )
    return store


def _parse_chunk_path(path: str) -> Tuple[Tuple[int], int]:
    level, chunk = path.split('/')
    inds = tuple(map(int, chunk.split('.')))
    return inds, int(level)