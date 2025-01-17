import pytest
import numpy as np
from sharpedge.pca_compression import pca_compression

# Expected Test Cases
@pytest.mark.parametrize("input_img, preservation_rate, expected_output", [
    (np.array([[25, 15], [15, 25]]), 0.5, np.array([[20, 20], [20, 20]])),  # Reduce by 50%    
    (np.array([[25, 15], [15, 25]]), 1.0, np.array([[25, 15], [15, 25]])),  # Reduce by 0%    
])
def test_valid_inputs(input_img, preservation_rate, expected_output):
    compressed_img = pca_compression(input_img, preservation_rate)
    assert compressed_img.shape == input_img.shape, (
        f"Compressed image shape mismatch for preservation_rate={preservation_rate}"
    )
    np.testing.assert_array_almost_equal(
        compressed_img, expected_output, decimal=0,
        err_msg=f"Output mismatch for input {input_img} with preservation_rate={preservation_rate}"
    )

# Edge Cases: 
def test_low_preservation_rate_warning():
    """Test that a warning is raised for very low preservation_rate (< 0.1)."""
    img = np.ones((100, 100))
    with pytest.warns(UserWarning, match="Very low preservation_rate may result in significant quality loss."):
        pca_compression(img, preservation_rate=0.05)

# Error Cases:
def test_invalid_preservation_rate_type():
    """Test that a non-numeric preservation_rate raises a TypeError."""
    img = np.ones((100, 100))
    with pytest.raises(TypeError, match="preservation_rate must be a number."):
        pca_compression(img, preservation_rate="0.5")

def test_invalid_preservation_rate_negative():
    """Test that a negative preservation_rate raises a ValueError."""
    img = np.ones((100, 100))
    with pytest.raises(ValueError, match="preservation_rate must be a float between 0 and 1."):
        pca_compression(img, preservation_rate=-0.1)

def test_invalid_input_not_2d():
    """Test that a non-2D input raises a ValueError."""
    img = np.ones((100, 100, 3))
    with pytest.raises(ValueError, match="Input image must be a 2D array."):
        pca_compression(img)