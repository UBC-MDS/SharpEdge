import pytest
import numpy as np
from sharpedge._utils.utility import Utility

err_msgs = {
    'Type_1': "Image format must be a numpy array.",
    'Type_2': "Image array must be 2D or 3D.",
    'Type_3': "Image size must not be zero in any dimension.",
    'Type_4': "Image array must have integer data type.",
    'Type_5': "Color values must be integers between 0 and 255."
}

# Expected Test Cases
@pytest.mark.parametrize("valid_array", [
    np.array([[1, 2], [3, 4]]),
    np.array([[[1, 2, 3], [4, 5, 6]]]),
    np.random.randint(1, 256, (100, 100)),
    np.random.randint(1, 256, (100, 200, 3))
])
def test_valid_image_format(valid_array):
    try:
        assert Utility._input_checker(valid_array) == True
    except Exception as e:
        pytest.fail(f"Valid data raised an error: {e}")

# Edge Cases

@pytest.mark.parametrize("valid_array", [
    np.array([[0, 0], [0, 0]]),  # Gray-scale images of 0s
    np.array([[255, 255], [255, 255]]),  # Gray-scale images of 255s
    np.array([[255, 255], [255, 255]]),  # Gray-scale images of 0s and 255s
    np.array([[[0, 0, 0], [0, 0, 0]]]),  # 3D images of 255s
    np.array([[[255, 255, 255], [255, 255, 255]]]),  # 3D images of 255s
    np.array([[[0, 0, 0], [255, 255, 255]]]),  # 3D images of 0s and 255s
])
def test_valid_image_format(valid_array):
    try:
        assert Utility._input_checker(valid_array) == True
    except Exception as e:
        pytest.fail(f"Valid data raised an error: {e}")

# Erroneous Cases

# Part 1: erroneous input array format
@pytest.mark.parametrize("invalid_input, expected_error", [
    ([1, 2, 3], err_msgs['Type_1']),
    ("invalid", err_msgs['Type_1']),
    (123, err_msgs['Type_1']),
    ((1, 2, 3), err_msgs['Type_1']),
    (None, err_msgs['Type_1'])
])
def test_erroneous_image_format(invalid_input, expected_error):
    with pytest.raises(TypeError, match=expected_error):
        Utility._input_checker(invalid_input)

# Part 2: erroneous input array shape
@pytest.mark.parametrize("invalid_input, expected_error", [
    (np.random.randint(1, 256, (100)), err_msgs["Type_2"]),
    (np.random.randint(1, 256, (100, 200, 3, 4)), err_msgs["Type_2"]),
    (np.random.randint(1, 256, (100, 200, 2)), err_msgs["Type_2"]),
    (np.random.randint(1, 256, (1, 3, 4, 2)), err_msgs["Type_2"]),
])
def test_erroneous_image_shape(invalid_input, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        Utility._input_checker(invalid_input)

# Part 3: the array is empty or contains zero-sized dimensions
@pytest.mark.parametrize("invalid_input", [
    np.empty((0, 10)),  # Zero-sized row in 2D array
    np.empty((10, 0)),  # Zero-sized column in 2D array
    np.empty((0, 10, 3)),  # Zero-sized dimension in 3D array
    np.empty((0, 0)),  # Fully empty 2D array
])
def test_zero_sized_dimensions(invalid_input):
    with pytest.raises(ValueError, match=err_msgs["Type_3"]):
        Utility._input_checker(invalid_input)

# Part 4: data type is not integer
@pytest.mark.parametrize("invalid_input", [
    np.array([[[2.35, 3.45, 4.55]]]),  # Floating-point 3D array
    np.array([[1.1, 2.2], [3.3, 4.4]]),  # Floating-point 2D array
    np.array([[1+2j, 3+4j], [5+6j, 7+8j]]),  # Complex numbers
    np.random.rand(10, 10),  # Random floating-point array
    np.array([[[255, 0, 0], [0, np.nan, 0]], [[0, 0, 255], [255, 255, 0]]]),  # RGB NaN values
    np.array([[np.nan, 128], [64, 0]]),  # Grayscale NaN values

])
def test_non_integer(invalid_input):
    with pytest.raises(TypeError, match=err_msgs["Type_4"]):
        Utility._input_checker(invalid_input)

# Part 5: any values are out of range (0 to 255)
@pytest.mark.parametrize("error_image", [
    np.array([[[255, 0, 0], [0, 256, 0]], [[0, 0, 255], [255, 255, 0]]]),  # RGB w/ 256 out of range
    np.array([[255, 128], [300, 0]]),  # Grayscale w/ 300 out of range
    np.array([[[-255, 0, 0], [0, -255, 0]], [[0, 0, 255], [255, 255, 0]]]),  # RGB w/ negative values
    np.array([[-255, -128], [-255, 0]]),  # Grayscale w/ negative values
    np.array([[256, 128], [64, 0]]),  # Grayscale out of range
])
def test_color_value_range(error_image):
    with pytest.raises(ValueError, match=err_msgs["Type_5"]):
        Utility._input_checker(error_image)