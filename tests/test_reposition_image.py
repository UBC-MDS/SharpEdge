import pytest
import numpy as np
from sharpedge.reposition_image import reposition_image

# Test for valid inputs
@pytest.mark.parametrize("flip, rotate, shift_x, shift_y, expected", [
    ('none', 'up', 0, 0, np.array([[1, 2], [3, 4]])),  # No transformation
    ('horizontal', 'up', 0, 0, np.array([[2, 1], [4, 3]])),  # Horizontal flip
    ('vertical', 'up', 0, 0, np.array([[3, 4], [1, 2]])),  # Vertical flip
    ('both', 'up', 0, 0, np.array([[4, 3], [2, 1]])),  # Both flips
    ('none', 'left', 0, 0, np.array([[2, 4], [1, 3]])),  # Rotate left
    ('none', 'right', 0, 0, np.array([[3, 1], [4, 2]])),  # Rotate right
    ('none', 'down', 0, 0, np.array([[4, 3], [2, 1]])),  # Rotate 180 degrees
    ('none', 'up', 1, 0, np.array([[2, 1], [4, 3]])),  # Shift along x-axis
    ('none', 'up', 0, 1, np.array([[3, 4], [1, 2]])),  # Shift along y-axis
])
def test_valid_inputs(flip, rotate, shift_x, shift_y, expected):
    img = np.array([[1, 2], [3, 4]])
    result = reposition_image(img, flip=flip, rotate=rotate, shift_x=shift_x, shift_y=shift_y)
    assert np.array_equal(result, expected), f"Unexpected result for flip={flip}, rotate={rotate}, shift_x={shift_x}, shift_y={shift_y}"


# Test edge cases
@pytest.mark.parametrize("flip, rotate, shift_x, shift_y", [
    ('none', 'up', 0, 0),  # Single-pixel image
    ('horizontal', 'down', 1, 1),  # Shift larger than dimensions
    ('vertical', 'right', -1, -1),  # Negative shift
    ('both', 'left', 0, 0),  # 1D edge case
])
def test_edge_cases(flip, rotate, shift_x, shift_y):
    img = np.array([[255]])
    result = reposition_image(img, flip=flip, rotate=rotate, shift_x=shift_x, shift_y=shift_y)
    assert result.shape == img.shape, f"Unexpected shape for flip={flip}, rotate={rotate}, shift_x={shift_x}, shift_y={shift_y}"


# Test erroneous inputs
@pytest.mark.parametrize("img, flip, rotate, shift_x, shift_y, error_type, error_msg", [
    (None, 'none', 'up', 0, 0, TypeError, "Input image must be a numpy array."),
    ("invalid", 'none', 'up', 0, 0, TypeError, "Input image must be a numpy array."),
    (np.array([[1, 2], [3, 4]]), 'invalid', 'up', 0, 0, ValueError, "flip must be one of 'none', 'horizontal', 'vertical', or 'both'."),
    (np.array([[1, 2], [3, 4]]), 'none', 'invalid', 0, 0, ValueError, "rotate must be one of 'up', 'left', 'right', or 'down'."),
    (np.array([[1, 2], [3, 4]]), 'none', 'up', 'invalid', 0, TypeError, "shift_x must be an integer."),
    (np.array([[1, 2], [3, 4]]), 'none', 'up', 0, 'invalid', TypeError, "shift_y must be an integer."),
    (np.array([]), 'none', 'up', 0, 0, ValueError, "Image array must not be empty."),
])
def test_erroneous_inputs(img, flip, rotate, shift_x, shift_y, error_type, error_msg):
    with pytest.raises(error_type, match=error_msg):
        reposition_image(img, flip=flip, rotate=rotate, shift_x=shift_x, shift_y=shift_y)
