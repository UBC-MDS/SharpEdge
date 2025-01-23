import pytest
import numpy as np
from sharpedge.pca_compression import pca_compression

# Expected Test Cases
@pytest.mark.parametrize("input_img, preservation_rate, expected_output", [
    (np.array([[25, 15], [15, 25]]), 0.9, np.array([[20, 20], [20, 20]])),  # Reduce by 90%  
    (np.array([[25, 15], [15, 25]]), 0.5, np.array([[20, 20], [20, 20]])),  # Reduce by 50%    
    (np.array([[25, 15], [15, 25]]), 1.0, np.array([[25, 15], [15, 25]])),  # Reduce by 0% 
    (np.array([[100, 200], [200, 100]]), 0.75, np.array([[150, 150], [150, 150]])),  # Reduce by 25%
    (np.array([[50, 150, 200], [120, 180, 160], [80, 160, 140]]), 0.2, np.array([[ 85., 165., 168.], [ 90., 176., 179.], [ 77., 149., 152.]])),  # Reduce by 80%   
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
@pytest.mark.parametrize("img, preservation_rate, warning_message", [
    (np.ones((100, 100), dtype=int), 0.05, "Very low preservation_rate may result in significant quality loss."),
    (np.ones((99, 101), dtype=int), 0.01, "Very low preservation_rate may result in significant quality loss."),
    (np.ones((98, 102), dtype=int), 0.09, "Very low preservation_rate may result in significant quality loss."),
])
def test_low_preservation_rate_warning(img, preservation_rate, warning_message):
    """Test that a warning is raised for very low preservation_rate (< 0.1)."""
    with pytest.warns(UserWarning, match=warning_message):
        pca_compression(img, preservation_rate)

# Error Cases:
@pytest.mark.parametrize("img, preservation_rate, expected_exception, expected_message", [
    # Invalid preservation_rate type
    (np.ones((100, 100), dtype=int), "0.5", TypeError, "preservation_rate must be a number."),
    # Negative preservation_rate
    (np.ones((100, 100), dtype=int), -0.1, ValueError, "preservation_rate must be a float between 0 and 1."),
    # preservation_rate larger than 1
    (np.ones((100, 100), dtype=int), 1.1, ValueError, "preservation_rate must be a float between 0 and 1."),
    # Non-2D input
    (np.ones((100, 100, 3), dtype=int), 0.5, ValueError, "Input image must be a 2D array."),
])
def test_invalid_preservation_rate_and_input(img, preservation_rate, expected_exception, expected_message):
    """Test invalid preservation_rate values and input image shapes."""
    with pytest.raises(expected_exception, match=expected_message):
        pca_compression(img, preservation_rate)