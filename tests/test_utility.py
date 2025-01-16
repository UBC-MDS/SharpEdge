from sharpedge._utils.utility import Utility
import numpy as np
import pytest

def test_invalid_dimensionality_1d_array():
    invalid_1d_array = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="Image array must be 2D or 3D."):
        Utility._input_checker(invalid_1d_array)

def test_invalid_dimensionality_4d_array():
    invalid_4d_array = np.random.rand(2, 2, 2, 2)

    with pytest.raises(ValueError, match="Image array must be 2D or 3D."):
        Utility._input_checker(invalid_4d_array)

def test_zero_size_array():
    zero_size_array = np.array([])

    with pytest.raises(ValueError, match="Image size must not be zero in any dimension."):
        Utility._input_checker(zero_size_array)

def test_zero_dimension_2d_array():
    zero_dim_2d_array = np.empty((0, 10))

    with pytest.raises(ValueError, match="Image size must not be zero in any dimension."):
        Utility._input_checker(zero_dim_2d_array)

def test_zero_dimension_3d_array():
    zero_dim_3d_array = np.empty((0, 10, 3))

    with pytest.raises(ValueError, match="Image size must not be zero in any dimension."):
        Utility._input_checker(zero_dim_3d_array)

def test_zero_dimension_multiple_axes():
    zero_dim_multi_axis = np.empty((0, 0, 3))

    with pytest.raises(ValueError, match="Image size must not be zero in any dimension."):
        Utility._input_checker(zero_dim_multi_axis)

def test_valid_2d_array():
    valid_2d_array = np.random.rand(10, 10)

    result = Utility._input_checker(valid_2d_array)
    assert result

def test_valid_3d_array():
    valid_3d_array = np.random.rand(10, 10, 3)
