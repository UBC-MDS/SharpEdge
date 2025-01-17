import pytest
import numpy as np
from sharpedge._utils.utility import Utility

# Part 1 of the input_checker: image format
# Expected Test Cases
@pytest.mark.parametrize("valid_array", [
    np.array([[1, 2], [3, 4]]),
    np.array([[[1, 2, 3], [4, 5, 6]]])
])
def test_valid_image_format(valid_array):
    try:
        Utility._input_checker(valid_array)
    except Exception as e:
        pytest.fail(f"Valid data raised an error: {e}")


# Edge cases
@pytest.mark.parametrize("edge_array", [
    np.array([[255]]),
    np.array([[[255, 0, 0]]]),
    np.array([[1.5, 2.5]]),
    np.array([[[2.35, 3.45, 4.55]]]),
    np.array([]),
    np.array([[]]),
])
def test_edge_image_format(edge_array):
    try:
        Utility._input_checker(edge_array)
    except Exception as e:
        pytest.fail(f"Edge data raised an error: {e}")


# Erroneous Cases
@pytest.mark.parametrize("invalid_input, expected_error", [
    ([1, 2, 3], "Image format must be a numpy array."),
    ("invalid", "Image format must be a numpy array."),
    (123, "Image format must be a numpy array."),
    ((1, 2, 3), "Image format must be a numpy array."),
    (None, "Image format must be a numpy array.")
])
def test_erroneous_image_format(invalid_input, expected_error):
    with pytest.raises(TypeError, match=expected_error):
        Utility._input_checker(invalid_input)

        
# Part 3 of the input_checker: Empty Array & Missing Dimensions      
# Valid Cases: Test the utility function with valid edge cases
@pytest.mark.parametrize("edge_array", [
    np.array([[255]]),  # 1 pixel 2D array
    np.array([[[255, 0, 0]]]),  # 1 pixel 3D array
    np.array([[1.5, 2.5]]),  # 2D array with float values
    np.array([[[2.35, 3.45, 4.55]]]),  # 1 pixel 3D array with float values
    np.array([[np.nan, np.nan], [np.nan, np.nan]]),  # 2D array with NaN values to simulate "empty"
    np.array([[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]])  # 3D array with NaN values to simulate "empty"
])
def test_edge_image_format(edge_array):
    result = Utility._input_checker(edge_array)
    assert result is None  # Expect no error for valid arrays

# Erroneous Cases: Test with invalid input and check if expected error is raised
@pytest.mark.parametrize("invalid_input, expected_error", [
    (np.array([1, 2, 3]), "Image array must be 2D or 3D."),  # 1D array
    (np.random.rand(2, 2, 2, 2), "Image array must be 2D or 3D."),  # 4D array
    (np.array([]), "Image size must not be zero in any dimension."),  # Empty array
    (np.empty((0, 10)), "Image size must not be zero in any dimension."),  # Zero-sized dimension in 2D array
    (np.empty((0, 10, 3)), "Image size must not be zero in any dimension."),  # Zero-sized dimension in 3D array
    (np.empty((0, 0, 3)), "Image size must not be zero in any dimension."),  # Zero-sized dimensions in multiple axes
    (np.empty((1, 0)), "Image size must not be zero in any dimension."),  # Zero-sized column in 2D
    (np.empty((0, 0)), "Image size must not be zero in any dimension.")  # Zero-sized 2D array
])
def test_erroneous_image_format(invalid_input, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        Utility._input_checker(invalid_input)

# Valid Case: Test for 2D array
@pytest.mark.parametrize("valid_input", [
    np.array([[1, 2], [3, 4]]),  # Standard 2D array
    np.array([[1.1, 2.2], [3.3, 4.4]]),  # 2D array with float values
    np.array([[255, 255], [0, 0]])  # 2D array with image-like values
])
def test_valid_2d_array(valid_input):
    result = Utility._input_checker(valid_input)
    assert result is None  # Expect no error for valid arrays

# Valid Case: Test for 3D array
@pytest.mark.parametrize("valid_input", [
    np.array([[[1, 2, 3]]]),  # 1 pixel 3D array
    np.array([[[1.1, 2.2, 3.3]]]),  # 1 pixel 3D array with float values
    np.array([[[255, 0, 0]]])  # 1 pixel 3D array with image-like values
])
def test_valid_3d_array(valid_input):
    result = Utility._input_checker(valid_input)
    assert result is None  # Expect no error for valid arrays

# Erroneous Case: Check for ValueError on zero-sized dimension
@pytest.mark.parametrize("invalid_input", [
    np.empty((0, 10)),  # Zero-sized row in 2D array
    np.empty((10, 0)),  # Zero-sized column in 2D array
    np.empty((0, 10, 3)),  # Zero-sized dimension in 3D array
    np.empty((0, 0)),  # Fully empty 2D array
    np.empty((0, 0, 0))  # Fully empty 3D array
])
def test_zero_size_dimensions(invalid_input):
    with pytest.raises(ValueError, match="Image size must not be zero in any dimension."):
      Utility._input_checker(invalid_input)

     
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
