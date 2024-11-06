import geojson
import numpy as np
import random
from qubalab.objects.image_feature import ImageFeature
from qubalab.objects.classification import Classification
from qubalab.images.labeled_server import LabeledImageServer
from qubalab.images.region_2d import Region2D
from qubalab.images.metadata.image_metadata import ImageMetadata
from qubalab.images.metadata.image_shape import ImageShape
from qubalab.images.metadata.pixel_calibration import PixelCalibration, PixelLength


sample_metadata = ImageMetadata(
    "/path/to/img.tiff",
    "Image name",
    (ImageShape(10, 6),),
    PixelCalibration(
        PixelLength.create_microns(2.5),
        PixelLength.create_microns(2.5)
    ),
    True,
    np.uint8
)
large_metadata = ImageMetadata(
    "/path/to/img.tiff",
    "Image name",
    (ImageShape(500, 250),),
    PixelCalibration(
        PixelLength.create_microns(2.5),
        PixelLength.create_microns(2.5)
    ),
    True,
    np.uint8
)



def test_image_width_with_downsample():
    downsample = 1.5
    expected_width = sample_metadata.shape.x
    labeled_server = LabeledImageServer(sample_metadata, [], downsample=downsample)

    width = labeled_server.metadata.width

    assert width == expected_width

    labeled_server.close()


def test_image_height_with_downsample():
    downsample = 1.5
    expected_height = sample_metadata.shape.y
    labeled_server = LabeledImageServer(sample_metadata, [], downsample=downsample)

    height = labeled_server.metadata.height

    assert height == expected_height

    labeled_server.close()


def test_image_width_with_no_downsample():
    expected_width = sample_metadata.shape.x
    labeled_server = LabeledImageServer(sample_metadata, [])

    width = labeled_server.metadata.width

    assert width == expected_width

    labeled_server.close()


def test_image_height_with_no_downsample():
    expected_height = sample_metadata.shape.y
    labeled_server = LabeledImageServer(sample_metadata, [])

    height = labeled_server.metadata.height

    assert height == expected_height

    labeled_server.close()


def test_number_of_channels_when_not_multichannel():
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((2, 5)), some_classification),
        ImageFeature(geojson.Point((5, 7)), some_classification),
        ImageFeature(geojson.Point((17, 7)), some_other_classification),
    ]
    expected_number_of_channels = 1
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False)

    n_channels = labeled_server.metadata.n_channels

    assert n_channels == expected_number_of_channels

    labeled_server.close()


def test_number_of_channels_when_multi_channel_and_no_label_map_given():
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((2, 5)), some_classification),
        ImageFeature(geojson.Point((5, 7)), some_classification),
        ImageFeature(geojson.Point((17, 7)), some_other_classification),
    ]
    expected_number_of_channels = 4
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=True)

    n_channels = labeled_server.metadata.n_channels

    assert n_channels == expected_number_of_channels

    labeled_server.close()


def test_number_of_channels_when_multi_channel_and_label_map_given():
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((2, 5)), some_classification),
        ImageFeature(geojson.Point((5, 7)), some_classification),
        ImageFeature(geojson.Point((17, 7)), some_other_classification),
    ]
    label_map = {
        some_classification: 1,
        some_other_classification: 2,
    }
    expected_number_of_channels = 3
    labeled_server = LabeledImageServer(sample_metadata, features, label_map=label_map, multichannel=True)

    n_channels = labeled_server.metadata.n_channels

    assert n_channels == expected_number_of_channels

    labeled_server.close()


def test_image_n_timepoints():
    expected_n_timepoints = 1
    labeled_server = LabeledImageServer(sample_metadata, [])

    n_timepoints = labeled_server.metadata.n_timepoints

    assert n_timepoints == expected_n_timepoints

    labeled_server.close()


def test_image_n_z_slices():
    expected_n_z_slices = 1
    labeled_server = LabeledImageServer(sample_metadata, [])

    n_z_slices = labeled_server.metadata.n_z_slices

    assert n_z_slices == expected_n_z_slices

    labeled_server.close()


def test_image_n_resolutions():
    expected_n_resolutions = 1
    labeled_server = LabeledImageServer(sample_metadata, [])

    n_resolutions = labeled_server.metadata.n_resolutions

    assert n_resolutions == expected_n_resolutions

    labeled_server.close()


def test_x_pixel_length_with_downsample():
    downsample = 1.5
    expected_length_x = sample_metadata.pixel_calibration.length_x.length
    labeled_server = LabeledImageServer(sample_metadata, [], downsample=downsample)

    length_x = labeled_server.metadata.pixel_calibration.length_x.length

    assert length_x == expected_length_x

    labeled_server.close()


def test_x_pixel_length_with_no_downsample():
    expected_length_x = sample_metadata.pixel_calibration.length_x.length
    labeled_server = LabeledImageServer(sample_metadata, [])

    length_x = labeled_server.metadata.pixel_calibration.length_x.length

    assert length_x == expected_length_x

    labeled_server.close()


def test_dtype_when_not_multi_channel():
    expected_dtype = np.uint32
    labeled_server = LabeledImageServer(sample_metadata, [], multichannel=False)

    dtype = labeled_server.metadata.dtype

    assert dtype == expected_dtype

    labeled_server.close()


def test_dtype_when_multi_channel():
    expected_dtype = bool
    labeled_server = LabeledImageServer(sample_metadata, [], multichannel=True)

    dtype = labeled_server.metadata.dtype

    assert dtype == expected_dtype

    labeled_server.close()


def test_read_points_in_single_channel_image_without_label_map_without_downsample():
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((5, 2)), some_classification),
        ImageFeature(geojson.Point((7, 4)), some_classification),
        ImageFeature(geojson.Point((1, 0)), some_other_classification),
    ]
    expected_image = np.array(
        [[[0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False)

    image = labeled_server.read_region(1, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_points_in_multi_channel_image_without_label_map_without_downsample():
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((5, 2)), some_classification),
        ImageFeature(geojson.Point((7, 4)), some_classification),
        ImageFeature(geojson.Point((1, 0)), some_other_classification),
    ]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

         [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=True)

    image = labeled_server.read_region(1, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_points_in_single_channel_image_with_label_map_without_downsample():
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((5, 2)), some_classification),
        ImageFeature(geojson.Point((7, 4)), some_classification),
        ImageFeature(geojson.Point((1, 0)), some_other_classification),
    ]
    label_map = {
        some_classification: 1,
        some_other_classification: 2,
    }
    expected_image = np.array(
        [[[0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, label_map=label_map, multichannel=False)

    image = labeled_server.read_region(1, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_points_in_single_channel_image_without_label_map_with_downsample():
    downsample = 2
    some_classification = Classification("Some classification")
    some_other_classification = Classification("Some other classification")
    features = [
        ImageFeature(geojson.Point((6, 2)), some_classification),
        ImageFeature(geojson.Point((8, 4)), some_classification),
        ImageFeature(geojson.Point((2, 0)), some_other_classification),
    ]
    expected_image = np.array(
        [[[0, 3, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 2]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False, downsample=downsample)

    image = labeled_server.read_region(downsample, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_line_in_single_channel_image_without_label_map_without_downsample():
    features = [ImageFeature(geojson.LineString([(6, 2), (8, 2)]), Classification("Some classification"))]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False)

    image = labeled_server.read_region(1, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_line_in_single_channel_image_without_label_map_with_downsample():
    downsample = 2
    features = [ImageFeature(geojson.LineString([(6, 2), (8, 2)]), Classification("Some classification"))]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1],
          [0, 0, 0, 0, 0]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False, downsample=downsample)

    image = labeled_server.read_region(downsample, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_polygon_in_single_channel_image_without_label_map_without_downsample():
    features = [ImageFeature(geojson.Polygon([[(6, 2), (8, 2), (8, 4), (4, 4)]]), Classification("Some classification"))]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False)

    image = labeled_server.read_region(1, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)


def test_read_polygon_in_single_channel_image_without_label_map_with_downsample():
    downsample = 2
    features = [ImageFeature(geojson.Polygon([[(6, 2), (8, 2), (8, 4), (4, 4)]]), Classification("Some classification"))]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False, downsample=downsample)

    image = labeled_server.read_region(downsample, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    np.testing.assert_array_equal(image, expected_image)

def test_label_can_hold_many_values():
    downsample = 1
    max_objects = 1000
    random.seed(1)
    def rands():
        x = random.randint(0, int(large_metadata.shape.x / downsample))
        y = random.randint(0, int(large_metadata.shape.x / downsample))
        return (
            (x, y),
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1)
        )
    
    features = [ImageFeature(geojson.Polygon([rands()])) for i in range(max_objects)]
    labeled_server = LabeledImageServer(large_metadata, features, multichannel=False, downsample=downsample)

    image = labeled_server.read_region(1, Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height))

    assert np.max(image) == max_objects


def test_single_channel_labeled_image_with_region_request():
    downsample = 1
    features = [ImageFeature(geojson.LineString([(7, 5), (9, 5)]))]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False, downsample=downsample)
    region = Region2D(5, 3, labeled_server.metadata.width-5, labeled_server.metadata.height-3)
    image = labeled_server.read_region(1, region)

    np.testing.assert_array_equal(image, expected_image)



def test_single_channel_labeled_image_with_starting_downsample():
    features = [ImageFeature(geojson.LineString([(6, 5), (9, 5)]))]
    # when resizing, we lose the labels with bicubic
    expected_image = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False, downsample=1)
    downsample = 2
    region = Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height)
    image = labeled_server.read_region(downsample, region)
    
    np.testing.assert_array_equal(image, expected_image)


def test_single_channel_labeled_image_with_request_downsample():
    # we downsample 
    features = [ImageFeature(geojson.LineString([(6, 5), (9, 5)]))]
    expected_image = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=False, downsample=1)
    region = Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height)
    image = labeled_server.read_region(2, region)

    np.testing.assert_array_equal(image, expected_image)



def test_multi_channel_labeled_image_with_region_request():
    features = [ImageFeature(geojson.LineString([(7, 5), (9, 5)]))]
    expected_image = np.array(
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, False, False]],
          [[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, True, True, True]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=True, downsample=1)
    region = Region2D(5, 3, labeled_server.metadata.width-5, labeled_server.metadata.height-3)
    image = labeled_server.read_region(1, region)

    np.testing.assert_array_equal(image, expected_image)




def test_multi_channel_labeled_image_with_starting_downsample():
    # we downsample the feature, then request at the same downsample
    features = [ImageFeature(geojson.LineString([(6, 5), (9, 5)]))]
    expected_image = np.array(
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, False, False]],
          [[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, True, True]]]
    )
    downsample = 2
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=True, downsample=downsample)
    region = Region2D(0, 0, sample_metadata.width, sample_metadata.height)
    image = labeled_server.read_region(2, region)

    np.testing.assert_array_equal(image, expected_image)

def test_multi_channel_labeled_image_with_request_downsample():
    features = [ImageFeature(geojson.LineString([(6, 5), (9, 5)]))]
    ## because we resize the image after reading, we lose the small region
    expected_image = np.array(
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, False, False]],
          [[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, False, False]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=True, downsample=1)
    downsample = 2
    region = Region2D(0, 0, labeled_server.metadata.width, labeled_server.metadata.height)
    image = labeled_server.read_region(downsample, region)

    np.testing.assert_array_equal(image, expected_image)


def test_multi_channel_labeled_image_with_starting_downsample_upsampled():
    # we downsample the feature, then request at a downsample of 1, so upsampled!
    # therefore the feature gets much bigger
    features = [ImageFeature(geojson.LineString([(5, 5), (9, 5)]))]
    expected_image = np.array(
        [[[False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]],

       [[False, False, False, False, False],
        [False, False, False, False, False],
        [False, False,  True,  True,  True]]]
    )
    labeled_server = LabeledImageServer(sample_metadata, features, multichannel=True, downsample=2)
    image = labeled_server.read_region(2)

    np.testing.assert_array_equal(image, expected_image)
