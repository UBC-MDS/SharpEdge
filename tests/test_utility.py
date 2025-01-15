from sharpedge._utils import Utility
import numpy as np
import pytest

result = Utility._input_checker([1]) # call the _input_checker function
print(result)

def test_invalid_dimensionality_1d_array():
    invalid_1d_array = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="Image array must be 2D or 3D."):
        Utility._input_checker(invalid_1d_array)

