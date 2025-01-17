import numpy as np
import pytest
import matplotlib.pyplot as plt
from sharpedge.seam_carving import seam_carve


# Test data
input_1 = plt.imread("test_image/seam_carve_input_1.png")[:, :, :3]
input_3 = plt.imread("test_image/seam_carve_input_3.png")[:, :, :3]
output_1 = plt.imread("test_image/seam_carve_output_1.png")[:, :, :3]
output_2 = plt.imread("test_image/seam_carve_output_2.png")[:, :, :3]
output_3 = plt.imread("test_image/seam_carve_output_3.png")[:, :, :3]


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
    (input_1, 7, 7, output_1),
    (input_1, 5, 4, output_2),
    (input_3, 1000, 1000, output_3)
])
def test_seam_carve_expected_array(img, target_height, target_width, expected_array):
    result = seam_carve(img, target_height, target_width)
    if expected_array.size > 0:
        np.testing.assert_array_almost_equal(result, expected_array)
