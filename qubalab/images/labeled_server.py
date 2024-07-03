import numpy as np
from .image_server import ImageServer
from .wrapped_image_server import WrappedImageServer
from .metadata.image_server_metadata import ImageServerMetadata
from .metadata.image_shape import ImageShape
from .region_2d import Region2D


class LabeledImageServer(WrappedImageServer):
    """
    TODO
    """

    def __init__(self, base_server: ImageServer, downsample: float = None, dtype=np.float32):
        """
        TODO
        """
        super().__init__(base_server)

        self._metadata = ImageServerMetadata(
            f'{self.base_server.path} - {self._label_map}',
            f'{self.base_server.metadata.name} - labels',
            base_server.metadata.shapes if downsample is None else ImageShape(
                int(base_server.metadata.width / downsample),
                int(base_server.metadata.height / downsample),
                base_server.metadata.n_timepoints,
                base_server.metadata.n_channels,
                base_server.metadata.n_z_slices,
            ),
            base_server.metadata.pixel_calibration,
            base_server.metadata.is_rgb,    #TODO: check
            dtype
        )

    def close(self):
        pass

    def _build_metadata(self) -> ImageServerMetadata:
        return self._metadata

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        pass
