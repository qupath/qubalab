from argparse import ArgumentError
from typing import Union, Iterable, Tuple, Dict
from operator import itemgetter
import uuid
import math
import numpy as np
import traceback as tb

import zarr
from zarr.storage import BaseStore, init_group, init_array, attrs_key
from zarr.util import json_dumps

from dask import array as da

from .servers import ImageServer, Region2D, _get_level


class PyramidStore(BaseStore):
    """
    Zarr-compatible wrapper for an ImageServer.

    Inspired by https://github.com/manzt/napari-lazy-openslide
       Copyright (c) 2020, Trevor Manz (BSD-3-clause)
    """

    def __init__(self, server: ImageServer, tile_size: Union[int, Tuple[int, int]] = None, name: str = None,
                 downsamples: Union[float, Iterable[float]] = None, squeeze=True):
        """
        Create a Zarr-compatible mapping for a multiresolution image reading pixels from an ImageServer.

        Note that the details of this implementation may change to better align with any future standardized representation
        of pyramidal images.
        In general, if the desired outcome is something array-like it is best to use to_dask instead.

        :param server:
        :param tile_size:
        :param name:
        :param downsamples:
        :param squeeze:
        """
        super().__init__()
        self._server = server

        if not tile_size:
            tile_size = (1024, 1024)
        elif isinstance(tile_size, (int, float)):
            tile_size = (int(tile_size), int(tile_size))
        self._tile_width = tile_size[0]
        self._tile_height = tile_size[1]

        if downsamples is None:
            downsamples = server.downsamples
        elif not isinstance(downsamples, Iterable):
            downsamples = (downsamples,)
        if name is None:
            name = str(uuid.uuid1())

        self._downsamples = downsamples
        self._store = self._build_store(downsamples=downsamples, name=name, squeeze=squeeze)

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
            tile_width = min(full_width, self._tile_width)
            tile_height = min(full_height, self._tile_height)

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

    def _parse_chunk_path(self, path: str) -> Tuple[int, int, int, int, int]:
        level, chunk = path.split('/')
        chunks = tuple(map(int, chunk.split('.')))
        return tuple([self._extract_chunk_path(chunks, 't'),
                      self._extract_chunk_path(chunks, 'z'),
                      chunks[-2],
                      chunks[-1],
                      int(level)])

    def _extract_chunk_path(self, lengths: Tuple[int, ...], target: str):
        if target in self._dims:
            return lengths[self._dims.index(target)]
        return 0

    def _build_store(self, downsamples: Iterable[float], name: str = None, squeeze=True) -> Dict[str, bytes]:
        """
        Build Zarr storage
        """

        server = self._server
        tile_width = self._tile_width
        tile_height = self._tile_height

        if name is None:
            name = server.name

        store = dict()

        for ii, downsample in enumerate(downsamples):
            w = int(server.width / downsample)
            h = int(server.height / downsample)
            c = server.n_channels
            z = server.n_z_slices
            t = server.n_timepoints
            cal = server.metadata.pixel_calibration

            # Default shape and chunks
            # Don't support z or t chunks > 1
            shape = (t, c, z, h, w)
            chunks = (1, c, 1, min(h, tile_height), min(w, tile_width))

            # Determine which dimensions to keep
            if squeeze:
                inds = [ii for ii, s in enumerate(shape) if s > 1 or ii > 2]
                getter = itemgetter(*inds)
            else:
                getter = itemgetter(0, 1, 2, 3, 4)
            axes = [
                dict(name='t', type='time'),
                dict(name='c', type='channel'),
                dict(name='z', type='space', unit=cal.length_z.unit),
                dict(name='y', type='space', unit=cal.length_y.unit),
                dict(name='x', type='space', unit=cal.length_x.unit),
            ]
            self._dims = ''.join(getter('tczyx'))

            # Write main info for highest-resolution
            # See https://ngff.openmicroscopy.org/
            if ii == 0:
                datasets = []
                # Store scales for later
                self._scale = tuple(getter([1.0, 1.0, cal.length_z.length, cal.length_y.length, cal.length_x.length]))

                # Compute scales for downsample (for OME-Zarr in the future)
                for di, d in enumerate(downsamples):
                    scale_for_level = [s for s in self._scale]
                    scale_for_level[-2] = scale_for_level[-2] * d
                    scale_for_level[-1] = scale_for_level[-1] * d
                    datasets.append(dict(path=str(di), coordinateTransformations=[
                        dict(type='scale', scale=scale_for_level)
                    ]))
                root_attrs = dict(
                    multiscales=[
                        dict(
                            name=name,
                            datasets=datasets,
                            version=0.4,
                            axes=getter(axes)
                        )
                    ]
                )
                # print(root_attrs)
                store = dict()
                init_group(store)
                store[attrs_key] = json_dumps(root_attrs)

            init_array(
                store,
                path=str(ii),
                shape=getter(shape),
                chunks=getter(chunks),
                dtype=server.dtype,
                compressor=None
            )
        return store


def _open_zarr_group(image: Union[ImageServer, PyramidStore], **kwargs):
    """
    Create & open a read-only Zarr group containing multiresolution data for an ImageServer.

    Note that the details of this implementation may change to better align with any future standardized representation
    of pyramidal images.
    Currently, it is only intended for internal use with to_dask()

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


def to_dask(image: Union[ImageServer, PyramidStore], rgb=None, as_napari_kwargs=False, squeeze=True,
            downsamples: Union[float, Iterable[float]] = None, **kwargs):
    """
    Create one or more dask arrays for an ImageServer.
    This provides a more pythonic/numpy-esque method to extract pixel data at any arbitrary resolution.

    Internally, the conversion uses a Zarr group, opened in read-only mode.

    :param image:  the image containing pixels
    :param rgb:    optionally specify that the image is RGB; default is None (auto-detect)
    :param as_napari_kwargs:  if True, wrap the output in a dict that can be passed to napari.view_image.
    :param squeeze: if True, remove singleton z, t and c dimensions. This is only used if the image does not provide
                    dask arrays directly.
    :param kwargs: passed to PyramidStore if the input is an ImageServer
    :return:       a single dask array if the keyword argument 'downsamples' is a number, or a tuple of dask arrays if
                   'downsamples' is an iterable; if as_napari_kwargs this is passed as the 'data' value in a dict
    """

    pyramid = None
    # If we have an ImageServer that can directly supply dask arrays, use those
    if isinstance(image, ImageServer):
        if rgb is None:
            rgb = image.is_rgb
        try:
            pyramid = image.get_dask_arrays(downsamples=downsamples)
            if pyramid[0].ndim == 5:
                dims = 'tzyxc'
                cal = image.metadata.pixel_calibration
                scale = [1, cal.length_z.length, cal.length_y.length, cal.length_x.length, 1]
            else:
                print('Cannot create dask arrays directly from ImageServer - falling back to Zarr')
                pyramid = None
        except:
            pyramid = None

    # If that didn't work, try to sort out our own dask arrays
    if pyramid is None:
        if isinstance(image, PyramidStore):
            store = image
        elif isinstance(image, ImageServer):
            store = PyramidStore(image, squeeze=squeeze, downsamples=downsamples, **kwargs)
        else:
            raise ValueError(f'Unable to convert object of type {type(image)} to Dask array - '
                             f'only ImageServer and PyramidStore supported')

        grp = _open_zarr_group(store, squeeze=squeeze, **kwargs)
        multiscales = grp.attrs["multiscales"][0]
        pyramid = tuple(da.from_zarr(store, component=d["path"]) for d in multiscales["datasets"])
        dims = store._dims
        scale = [s for s in store._scale]

        if rgb:
            pyramid = tuple(np.moveaxis(p, 0, -1) for p in pyramid)

    # If we requested a single downsample, then return it directly (not in a tuple)
    data = pyramid
    if len(pyramid):
        if not isinstance(downsamples, Iterable) and downsamples is not None:
            data = pyramid[0]

    # Return either the array, or napari kwargs for display
    if as_napari_kwargs:
        c_axis = None
        colormap = None
        names = None
        if isinstance(image, ImageServer):
            names = image.name
        # Make channels-last if RGB
        if 'c' in dims:
            c_axis = dims.index('c')
            dims = dims.replace('c', '')  # Currently (Napari 0.4.16) needs to remove c when specifying index
            scale.pop(c_axis)
            # Need to remove c_axis again if we have an RGB image
            if rgb:
                c_axis = None
            elif isinstance(image, ImageServer):
                colormap = [c.color for c in image.channels]
                names = [c.name for c in image.channels]
        return dict(data=data, rgb=rgb, channel_axis=c_axis,
                    axis_labels=tuple(dims), scale=scale,
                    colormap=colormap, name=names)
    else:
        return data
