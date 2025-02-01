import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from sharpedge import seam_carve

test_image_dir = os.path.join(os.path.dirname(__file__), "test_image")

# Test data
# Image input_1 is sourced from DSCI 512: Algorithms and Data Structures Lab 4
input_1 = plt.imread(os.path.join(test_image_dir, "seam_carve_input_1.png"))[:, :, :3]
input_2 = plt.imread(os.path.join(test_image_dir, "seam_carve_input_2.png"))[:, :, :3]
input_3 = plt.imread(os.path.join(test_image_dir, "seam_carve_input_3.png"))[:, :, :3]
output_1_1 = plt.imread(os.path.join(test_image_dir, "seam_carve_output_1_1.png"))[:, :, :3]
output_1_2 = plt.imread(os.path.join(test_image_dir, "seam_carve_output_1_2.png"))[:, :, :3]
output_1_3 = plt.imread(os.path.join(test_image_dir, "seam_carve_output_1_3.png"))[:, :, :3]
output_2 = plt.imread(os.path.join(test_image_dir, "seam_carve_output_2.png"))[:, :, :3]
output_3 = plt.imread(os.path.join(test_image_dir, "seam_carve_output_3.png"))[:, :, :3]


# Expected cases
@pytest.mark.parametrize("img, target_height, target_width, expected_shape", [
    (np.random.rand(10, 10, 3), 5, 5, (5, 5, 3)),
    (np.random.rand(8, 6, 3), 6, 4, (6, 4, 3)),
    (np.random.rand(100, 100, 3), 50, 50, (50, 50, 3))
])
def test_seam_carve_expected_shape(img, target_height, target_width, expected_shape):
    result = seam_carve(img, target_height, target_width)
    assert result.shape == expected_shape


@pytest.mark.parametrize("img, target_height, target_width, expected_array", [
    (input_1, 7, 7, output_1_1),
    (input_1, 5, 4, output_1_2),
    (input_2, 1000, 1000, output_2)
])
def test_seam_carve_expected_array(img, target_height, target_width, expected_array):
    result = seam_carve(img, target_height, target_width)
    if expected_array.size > 0:
        np.testing.assert_array_equal(result, expected_array)


# Edge cases
# Edge case 1: no resizing in height or width needed, expect warnings
@pytest.mark.parametrize("img, target_height, target_width, warning_msg", [
    (np.random.rand(10, 10, 3), 10, 10, "Both target height and width are the same as that of the original image. No resizing needed."),
    (np.random.rand(10, 10, 3), 10, 5, "Target height is the same as the original height."),
    (np.random.rand(10, 10, 3), 5, 10, "Target width is the same as the original width."),
])
def test_seam_carve_no_resize(img, target_height, target_width, warning_msg):
    with pytest.warns(UserWarning, match=warning_msg):
        result = seam_carve(img, target_height, target_width)


# Edge case 2: target width and height are 1, expect warnings
@pytest.mark.parametrize("img, target_height, target_width, warning_msg", [
    (np.random.rand(5, 8, 3), 1, 1, "Warning! Resizing to a single pixel."),
    (np.random.rand(10, 4, 3), 1, 1, "Warning! Resizing to a single pixel."),
    (np.random.rand(20, 15, 3), 1, 1, "Warning! Resizing to a single pixel."),
    (np.random.rand(2, 2, 3), 1, 1, "Warning! Resizing to a single pixel."),
])
def test_seam_carve_single_pixel_target_warning(img, target_height, target_width, warning_msg):
    with pytest.warns(UserWarning, match=warning_msg):
        result = seam_carve(img, target_height, target_width)


@pytest.mark.parametrize("img, target_height, target_width, expected_array, warning_msg", [
    (input_1, 1, 1, output_1_3, "Warning! Resizing to a single pixel."),
    (input_3, 1, 1, output_3, "Warning! Resizing to a single pixel.")
])
def test_seam_carve_single_pixel_target_output(img, target_height, target_width, expected_array, warning_msg):
    with pytest.warns(UserWarning, match=warning_msg):
        result = seam_carve(img, target_height, target_width)
        np.testing.assert_array_equal(result, expected_array)


# Edge case 3: seam carving with significant resizing, expect warnings
@pytest.mark.parametrize("img, target_height, target_width, warning_msg", [
    (np.random.rand(202, 40, 3), 2, 30, "Significant resizing is required. It may take a long while."),
    (np.random.rand(10, 300, 3), 5, 100, "Significant resizing is required. It may take a long while."),
    (np.random.rand(203, 204, 3), 3, 4, "Significant resizing is required. It may take a long while."),
])
def test_seam_carve_significant_resizing(img, target_height, target_width, warning_msg):
    with pytest.warns(UserWarning, match=warning_msg):
        result = seam_carve(img, target_height, target_width)


# Erroneous Cases
# Error case 1: invalid image
@pytest.mark.parametrize("img, target_height, target_width, error_type, error_msg", [
    ("not_an_array", 5, 5, TypeError, "Image format must be a numpy array."),
    (12345, 5, 5, TypeError, "Image format must be a numpy array."),
    (None, 5, 5, TypeError, "Image format must be a numpy array."),
    ([[]], 5, 5, TypeError, "Image format must be a numpy array."),
    (np.random.rand(10, 10), 5, 5, ValueError, "Input image must be a 3D numpy array with 3 channels."),
    (np.random.rand(10, 10, 2), 5, 5, ValueError, "Input image must be a 3D numpy array with 3 channels."),
    (np.random.rand(10, 10, 4), 5, 5, ValueError, "Input image must be a 3D numpy array with 3 channels.")
])
def test_seam_carve_invalid_image_type(img, target_height, target_width, error_type, error_msg):
    with pytest.raises(error_type, match=error_msg):
        result = seam_carve(img, target_height, target_width)


# Error case 2: invalid target dimensions
@pytest.mark.parametrize("img, target_height, target_width, error_msg", [
    (np.random.rand(10, 10, 3), [3], 4, "Target dimensions must be integers."),
    (np.random.rand(10, 10, 3), 5.5, 5, "Target dimensions must be integers."),
    (np.random.rand(10, 10, 3), 5, "target_width", "Target dimensions must be integers."),
    (np.random.rand(10, 10, 3), 5, np.array([3]), "Target dimensions must be integers."),
    (np.random.rand(10, 10, 3), None, 5, "Target dimensions must be integers."),
    (np.random.rand(10, 10, 3), 7, (1, 2, 3), "Target dimensions must be integers."),
    (np.random.rand(10, 10, 3), 7, 0, "Target width must be at least 1."),
    (np.random.rand(10, 10, 3), -5, 8, "Target height must be at least 1."),
    (np.empty((0, 10, 3)), 5, 5, "Image size must not be zero in any dimension."),
    (np.empty((10, 0, 3)), 5, 5, "Image size must not be zero in any dimension."),
    (np.empty((10, 10, 0)), 5, 5, "Image size must not be zero in any dimension."),
    (np.empty((0, 0, 3)), 5, 5, "Image size must not be zero in any dimension."),
])
def test_seam_carve_invalid_target_dimensions(img, target_height, target_width, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        result = seam_carve(img, target_height, target_width)


# Error case 3: seam carving with target height/width exceeding original height/width
@pytest.mark.parametrize("img, target_height, target_width, error_msg", [
    (np.random.rand(10, 10, 3), 11, 5, "Target height cannot be greater than original height."),
    (np.random.rand(5, 10, 3), 4, 15, "Target width cannot be greater than original width."),
])
def test_seam_carve_target_height_exceeds_original(img, target_height, target_width, error_msg):
    """Test seam carving with target height exceeding original height."""
    with pytest.raises(ValueError, match=error_msg):
        result = seam_carve(img, target_height, target_width)
