from sharpedge._utils.utility import Utility
import numpy as np
import pytest

def test_invalid_dimensionality_1d_array():
    invalid_1d_array = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="Image array must be 2D or 3D."):
        Utility._input_checker(invalid_1d_array)

