import pytest
import numpy as np
from sharpedge import pooling_image

# Valid cases: Testing pooling behavior with different pooling functions
@pytest.mark.parametrize("img, window_size, pooling_method, expected", [
    (np.array([[1, 2], [3, 4]]), 2, np.max, np.array([[4]])),   # 2x2 pooling with max
    (np.array([[1, 2], [3, 4]]), 2, np.min, np.array([[1]])),   # 2x2 pooling with min
    (np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), 2, np.mean,
     np.array([[[5.5 / 255, 6.5 / 255, 7.5 / 255]]]))  # RGB image pooling with mean, adjust by 255
])
def test_valid_pooling(img, window_size, pooling_method, expected):
    result = pooling_image(img, window_size, pooling_method)
    # Adjust expected to account for the 255 normalization in the pooling function
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

# Edge cases: Single-pixel output, very small input sizes, non-square images
@pytest.mark.parametrize("img, window_size, pooling_method, expected", [
    (np.array([[10]]), 1, np.mean, np.array([[10]])),  # Single pixel, no change
    (np.array([[5, 15], [10, 20]]), 2, np.mean, np.array([[12.5]])),  # Entire image pooling
    (np.array([[0, 0], [0, 0]]), 2, np.max, np.array([[0]])),  # All-zero image
    # Rectangular image (valid case), expect reshaping result
    (np.array([[1, 2, 3], [4, 5, 6]]), 1, np.mean, np.array([[1, 2, 3], [4, 5, 6]])), 
    # Column vector (valid case), expect reshaping result
    (np.array([[1], [2], [3]]), 1, np.mean, np.array([[1], [2], [3]])),
    # Row vector (valid case), expect reshaping result
    (np.array([[1, 2, 3]]), 1, np.mean, np.array([[1, 2, 3]]))
])
def test_edge_pooling(img, window_size, pooling_method, expected):
    result = pooling_image(img, window_size, pooling_method)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

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

# Image is not square
@pytest.mark.parametrize("img, window_size, pooling_method, expected_error", [
    (np.array([[1, 2, 3], [4, 5, 6]]), 1, np.mean, None),  # Rectangular image (valid case)
    (np.array([[1], [2], [3]]), 1, np.mean, None),  # Column vector (valid case)
    (np.array([[1, 2, 3]]), 1, np.mean, None)  # Row vector (valid case)
])
def test_non_square_images(img, window_size, pooling_method, expected_error):
    result = pooling_image(img, window_size, pooling_method)
    assert result is not None, f"Expected a result for non-square image"
