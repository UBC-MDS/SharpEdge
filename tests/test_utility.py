from sharpedge._utils import Utility

result = Utility._input_checker([1]) # call the _input_checker function
print(result)

## Part 4 of the input_checker: data color range
# function and docstring as placeholder, to be removed or combined during consolidation
import numpy as np
import pytest

# Expected Test Cases
# `validate_color_values` as a placeholder function name
# function name should be updated to `_input_checker` upon consolidation`
def test_valid_rgb_img():
    rgb_img = np.array([[[255, 0, 0], [0, 255, 0]], 
                        [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    assert validate_color_values(rgb_img) == True

def test_valid_grayscale_img():
    grayscale_img = np.array([[255, 128], [64, 0]], dtype=np.uint8)
    assert validate_color_values(grayscale_img) == True


# Edge Cases
def test_edge_single_pixel_rgb():
    single_pixel_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Single pixel RGB image
    assert validate_color_values(single_pixel_rgb) == True

def test_edge_single_pixel_grayscale():
    single_pixel_grayscale = np.array([[255]], dtype=np.uint8)  # Single pixel grayscale image
    assert validate_color_values(single_pixel_grayscale) == True

def test_edge_rgb_zero_values():
    rgb_img_zero = np.array([[[0, 0, 0], [0, 0, 0]], 
                             [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)  # All zero values
    assert validate_color_values(rgb_img_zero) == True

def test_edge_rgb_max_values():
    rgb_img_max = np.array([[[255, 255, 255], [255, 255, 255]], 
                            [[255, 255, 255], [255, 255, 255]]], dtype=np.uint8)  # All max values (255)
    assert validate_color_values(rgb_img_zero) == True

def test_edge_grayscale_zero_values():
    grayscale_img_zero = np.array([[0, 0], [0, 0]], dtype=np.uint8)  # All zero values for grayscale
    assert validate_color_values(grayscale_img_zero) == True

def test_edge_grayscale_max_values():
    grayscale_img_max = np.array([[255, 255], [255, 255]], dtype=np.uint8)  # All max values (255)
    assert validate_color_values(grayscale_img_max) == True


# Erroneous Cases
def test_rgb_with_out_of_range_values():
    rgb_img_oor = np.array([[[255, 0, 0], [0, 256, 0]], 
                            [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)  # 256 is out of range
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(rgb_img_oor)

def test_grayscale_with_out_of_range_values():
    grayscale_img_oor = np.array([[255, 128], [300, 0]], dtype=np.uint8)  # 300 is out of range
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(grayscale_img_oor)

def test_rgb_with_float_values():
    rgb_img_float = np.array([[[255.5, 0.0, 0.0], [0.0, 255.5, 0.0]], 
                              [[0.0, 0.0, 255.5], [255.5, 255.5, 0.0]]])  # Floats present
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(rgb_img_float)

def test_grayscale_with_float_values():
    grayscale_img_float = np.array([[255.5, 128], [64.2, 0]], dtype=np.float32)  # Floats present
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(grayscale_img_float)

def test_rgb_with_nan_values():
    rgb_img_nan = np.array([[[255, 0, 0], [0, np.nan, 0]], 
                            [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)  # NaN values in rgb
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(rgb_img_nan)

def test_grayscale_with_nan_values():
    grayscale_img_nan = np.array([[np.nan, 128], [64, 0]], dtype=np.float32)  # NaN values in grayscale
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(grayscale_img_nan)

# Additional Erroneous Case (Empty Image)
# Can be removed if properly tested in Part 3
def test_empty_img():
    empty_img = np.array([])  # Empty image
    with pytest.raises(ValueError, match="Color values must be integers between 0 and 255."):
        validate_color_values(empty_img)
