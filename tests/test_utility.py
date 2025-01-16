import pytest
import numpy as np
from sharpedge._utils.utility import Utility

## Part 4 of the input_checker: data color range
# Expected Test Cases:
@pytest.mark.parametrize("valid_image", [
    (np.array([[[100, 150, 200], [200, 150, 100]], [[50, 50, 50], [0, 0, 0]]], dtype=np.uint8)),
    (np.array([[100, 150], [200, 50]], dtype=np.uint8))
])
def test_valid_img(valid_image):
    # Valid data should pass the test without raising any errors
    try: 
        Utility._input_checker(valid_image)
    except ValueError as e:
        pytest.fail(f"Valid data raised an error: {e}")


# Edge Cases:
@pytest.mark.parametrize("edge_case_image", [
    (np.array([[[255, 0, 0]]], dtype=np.uint8)),  # Single-pixel RGB
    (np.array([[128]], dtype=np.uint8)),  # Single-pixel Grayscale
    (np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)),  # RGB all zero
    (np.array([[[255, 255, 255], [255, 255, 255]], [[255, 255, 255], [255, 255, 255]]], dtype=np.uint8)),  # RGB all max
    (np.array([[0, 0], [0, 0]], dtype=np.uint8)),  # Grayscale all zero
    (np.array([[255, 255], [255, 255]], dtype=np.uint8))  # Grayscale all max
])
def test_edge_values(edge_case_image):
    # Edge values should pass the test without raising any errors
    try: 
        Utility._input_checker(edge_case_image)
    except ValueError as e:
        pytest.fail(f"Edge data raised an error: {e}")


# Error Cases:
@pytest.mark.parametrize("error_image, expected_exception", [
    (np.array([[[255, 0, 0], [0, 256, 0]], [[0, 0, 255], [255, 255, 0]]]), ValueError),  # RGB w/ 256 out of range
    (np.array([[255, 128], [300, 0]]), ValueError),  # Grayscale w/ 300 out of range
    (np.array([[[-255, 0, 0], [0, -255, 0]], [[0, 0, 255], [255, 255, 0]]]), ValueError),  # RGB w/ negative values
    (np.array([[-255, -128], [-255, 0]]), ValueError),  # Grayscale w/ negative values
    (np.array([[[255.5, 0.0, 0.0], [0.0, 255.5, 0.0]], [[0.0, 0.0, 255.5], [255.5, 255.5, 0.0]]]), ValueError),  # RGB floats present
    (np.array([[255.5, 128], [64.2, 0]], dtype=np.float32), ValueError),  # Grayscale floats present
    (np.array([[[255, 0, 0], [0, np.nan, 0]], [[0, 0, 255], [255, 255, 0]]]), ValueError),  # RGB NaN values
    (np.array([[np.nan, 128], [64, 0]]), ValueError),  # Grayscale NaN values
])
def test_invalid_cases(error_image, expected_exception):
    with pytest.raises(expected_exception, match="Color values must be integers between 0 and 255."):
        Utility._input_checker(error_image)
