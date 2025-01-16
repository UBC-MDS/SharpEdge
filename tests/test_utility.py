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
    assert Utility._input_checker(valid_array)


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
    assert Utility._input_checker(edge_array)


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
