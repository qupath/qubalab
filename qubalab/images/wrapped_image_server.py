from .image_server import ImageServer
from .metadata.image_metadata import ImageMetadata


class WrappedImageServer(ImageServer):
    """
    Abstract class for an ImageServer that wraps another ImageServer,
    e.g. to transform the image in some way.
    
    Closing this server will close the wrapped server.
    The metadata of this server is equivalent to the metadata of the wrapped server.
    """

    def __init__(self, base_server: ImageServer, **kwargs):
        """
        :param base_server: the server to wrap
        :param resize_method: the resampling method to use when resizing the image for downsampling. Bicubic by default
        """
        super().__init__(**kwargs)
        self._base_server = base_server

    @property
    def base_server(self) -> ImageServer:
        """
        Get the wrapped server.
        """
        return self._base_server
    
    def close(self):
        self._base_server.close()

    def _build_metadata(self) -> ImageMetadata:
        return self._base_server.metadata
