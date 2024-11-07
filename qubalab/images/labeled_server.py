import numpy as np
from typing import Iterable
import shapely
import PIL
from .image_server import ImageServer
from .metadata.image_metadata import ImageMetadata
from .metadata.image_shape import ImageShape
from .metadata.pixel_calibration import PixelCalibration, PixelLength
from .region_2d import Region2D
from ..objects.image_feature import ImageFeature
from ..objects.classification import Classification
from ..objects.draw import draw_geometry


class LabeledImageServer(ImageServer):
    """
    An ImageServer where pixel values are labels corresponding to image features (such as annotations)
    present on an image.

    The returned image will have one timepoint and one z-stack. The size of the remaining dimensions depend
    on the metadata provided when creating the server --- usually, the same as the ImageServer that the labeled image corresponds to.

    The image will only have one resolution level; the downsample for this level may be greater than or less than 1, and consequently region requests and downsamples should be considered relative to the metadata provided at server creation, **not** relative to the downsampled (or upsampled) LabeledImageServer coordinates.
    """

    def __init__(
        self,
        base_image_metadata: ImageMetadata,
        features: Iterable[ImageFeature],
        label_map: dict[Classification, int] = None,
        downsample: float = None,
        multichannel: bool = False,
        resize_method=PIL.Image.Resampling.NEAREST,
        **kwargs,
    ):
        """
        :param base_image_metadata: the metadata of the image containing the image features
        :param features: the image features to draw
        :param label_map: a dictionary mapping a classification to a label. The value of pixels where an image feature with
                          a certain classification is present will be taken from this dictionnary. If not provided, each feature
                          will be assigned a unique integer. All labels must be greater than 0
        :param downsample: the downsample to apply to the image. Can be omitted to use the full resolution image
        :param multichannel: if False, the image returned by this server will have a single channel where pixel values will be unsigned
                             integers representing a label (see the label_map parameter). If True, the number of channels will be
                             equal to the highest label value + 1, and the pixel located at (c, y, x) is a boolean indicating if an annotation
                             with label c is present on the pixel located at (x, y)
        :param resize_method: the resampling method to use when resizing the image for downsampling. Nearest neighbour by default for labeled images.
        :raises ValueError: when a label in label_map is less than or equal to 0
        """
        super().__init__(resize_method=resize_method, **kwargs)

        if label_map is not None and any(label <= 0 for label in label_map.values()):
            raise ValueError(
                "A label in label_map is less than or equal to 0: " + str(label_map)
            )

        self._base_image_metadata = base_image_metadata
        self._downsample = 1 if downsample is None else downsample
        self._multichannel = multichannel
        self._features = [
            f for f in features if label_map is None or f.classification in label_map
        ]
        self._geometries = [
            shapely.affinity.scale(
                shapely.geometry.shape(f.geometry),
                1 / self._downsample,
                1 / self._downsample,
                origin=(0, 0, 0),
            )
            for f in self._features
        ]
        self._tree = shapely.STRtree(self._geometries)

        if label_map is None:
            self._feature_index_to_label = {
                i: i + 1 for i in range(len(self._features))
            }
        else:
            self._feature_index_to_label = {
                i: label_map[self._features[i].classification]
                for i in range(len(self._features))
            }

    def close(self):
        pass

    def _build_metadata(self) -> ImageMetadata:
        return ImageMetadata(
            self._base_image_metadata.path,
            f"{self._base_image_metadata.name} - labels",
            (
                ImageShape(
                    int(self._base_image_metadata.width),
                    int(self._base_image_metadata.height),
                    1,
                    max(self._feature_index_to_label.values(), default=0) + 1
                    if self._multichannel
                    else 1,
                    1,
                ),
            ),
            PixelCalibration(
                PixelLength(
                    self._base_image_metadata.pixel_calibration.length_x.length,
                    self._base_image_metadata.pixel_calibration.length_x.unit,
                ),
                PixelLength(
                    self._base_image_metadata.pixel_calibration.length_y.length,
                    self._base_image_metadata.pixel_calibration.length_y.unit,
                ),
                self._base_image_metadata.pixel_calibration.length_z,
            ),
            False,
            bool if self._multichannel else np.uint32,
            downsamples=[self._downsample],
        )

    def _read_block(self, level: int, region: Region2D) -> np.ndarray:
        if self._multichannel:
            full_image = np.zeros(
                (self.metadata.n_channels, region.height, region.width),
                dtype=self.metadata.dtype,
            )
            feature_indices = self._tree.query(region.geometry)
            labels = set(self._feature_index_to_label.values())

            for label in labels:
                image = PIL.Image.new("1", (region.width, region.height))
                drawing_context = PIL.ImageDraw.Draw(image)

                for i in feature_indices:
                    if label == self._feature_index_to_label[i]:
                        draw_geometry(
                            image.size,
                            drawing_context,
                            shapely.affinity.translate(
                                self._geometries[i], -region.x, -region.y
                            ),
                            1,
                        )
                full_image[label, :, :] = np.asarray(image, dtype=self.metadata.dtype)

            return full_image
        else:
            image = PIL.Image.new("I", (region.width, region.height))
            drawing_context = PIL.ImageDraw.Draw(image)
            for i in self._tree.query(region.geometry):
                draw_geometry(
                    image.size,
                    drawing_context,
                    shapely.affinity.translate(
                        self._geometries[i], -region.x, -region.y
                    ),
                    self._feature_index_to_label[i],
                )
            return np.expand_dims(np.asarray(image, dtype=self.metadata.dtype), axis=0)
