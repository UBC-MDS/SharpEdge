import numpy as np
from sharpedge._utils.utility import Utility


# Part 1 of the input_checker: image format
# Expected Test Cases
def test_valid_2d_numpy_array():
    # Arrange
    valid_2d_array = np.array([[1, 2], [3, 4]])
    # Act
    result = Utility._input_checker(valid_2d_array)
    # Assert
    assert result


def test_valid_3d_numpy_array():
    # Arrange
    valid_3d_array = np.array([[[1, 2, 3], [4, 5, 6]]])
    # Act
    result = Utility._input_checker(valid_3d_array)
    # Assert
    assert result


# Edge cases
def test_edge_case_single_pixel_grayscale():
    # Arrange
    single_pixel_grayscale = np.array([[255]])
    # Act
    result = Utility._input_checker(single_pixel_grayscale)
    # Assert
    assert result


def test_edge_case_single_pixel_rgb():
    # Arrange
    single_pixel_rgb = np.array([[[255, 0, 0]]])
    # Act
    result = Utility._input_checker(single_pixel_rgb)
    # Assert
    assert result


# Erroneous Cases
def test_invalid_type_list():
    # Arrange
    invalid_list = [1, 2, 3]
    # Act
    try:
        Utility._input_checker(invalid_list)
    # Assert
    except TypeError as e:
        assert str(e) == "Image format must be a numpy array."


def test_invalid_type_string():
    # Arrange
    invalid_string = "invalid"
    # Act
    try:
        Utility._input_checker(invalid_string)
    # Assert
    except TypeError as e:
        assert str(e) == "Image format must be a numpy array."


def test_invalid_type_int():
    # Arrange
    invalid_int = 123
    # Act
    try:
        Utility._input_checker(invalid_int)
    # Assert
    except TypeError as e:
        assert str(e) == "Image format must be a numpy array."
