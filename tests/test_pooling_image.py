import pytest
import numpy as np
from sharpedge.pooling_image import pooling_image

# Valid cases: Testing pooling behavior with different pooling functions
@pytest.mark.parametrize("img, window_size, pooling_method, expected", [
    (np.array([[1, 2], [3, 4]]), 2, np.max, np.array([[4]])),   # 2x2 pooling with max
    (np.array([[1, 2], [3, 4]]), 2, np.min, np.array([[1]])),   # 2x2 pooling with min
    (np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), 2, np.mean,
     np.array([[[5.5, 6.5, 7.5]]]))  # RGB image pooling with mean
])
def test_valid_pooling(img, window_size, pooling_method, expected):
    result = pooling_image(img, window_size, pooling_method)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

# Edge cases: Single-pixel output, very small input sizes
@pytest.mark.parametrize("img, window_size, pooling_method, expected", [
    (np.array([[10]]), 1, np.mean, np.array([[10]])),  # Single pixel, no change
    (np.array([[5, 15], [10, 20]]), 2, np.mean, np.array([[12.5]])),  # Entire image pooling
    (np.array([[0, 0], [0, 0]]), 2, np.max, np.array([[0]])),  # All-zero image
])
def test_edge_pooling(img, window_size, pooling_method, expected):
    result = pooling_image(img, window_size, pooling_method)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

# Erroneous cases: Testing logic errors not caught by _input_checker
@pytest.mark.parametrize("img, window_size, pooling_method, expected_error", [
    (np.array([[1, 2, 3], [4, 5, 6]]), 2, np.mean, ValueError),  # Non-divisible dimensions
    (np.array([[1, 2], [3, 4]]), 3, np.mean, ValueError),  # Window size larger than image
    (np.array([[1, 2], [3, 4]]), 2, "not_a_function", TypeError),  # Non-callable function
    (np.array([[1, 2], [3, 4]]), 2.5, np.mean, TypeError)  # Non-integer window size
])
def test_erroneous_pooling(img, window_size, pooling_method, expected_error):
    with pytest.raises(expected_error):
        pooling_image(img, window_size, pooling_method)

# New tests: Image dimensions not divisible by window_size
@pytest.mark.parametrize("img, window_size, pooling_method, expected_error", [
    (np.array([[1, 2, 3], [4, 5, 6]]), 4, np.mean, ValueError),  # Height not divisible by window size
    (np.array([[1, 2], [3, 4], [5, 6]]), 2, np.mean, ValueError),  # Width not divisible by window size
])
def test_non_divisible_dimensions(img, window_size, pooling_method, expected_error):
    with pytest.raises(expected_error):
        pooling_image(img, window_size, pooling_method)

# New tests: Image is not square
@pytest.mark.parametrize("img, window_size, pooling_method, expected_error", [
    (np.array([[1, 2, 3], [4, 5, 6]]), 2, np.mean, ValueError),  # Rectangular image
    (np.array([[1], [2], [3]]), 1, np.mean, ValueError),  # Column vector
    (np.array([[1, 2, 3]]), 1, np.mean, ValueError)  # Row vector
])
def test_non_square_images(img, window_size, pooling_method, expected_error):
    with pytest.raises(expected_error):
        pooling_image(img, window_size, pooling_method)
