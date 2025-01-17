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

# Test erroneous cases that pass
@pytest.mark.parametrize("flip, rotate, shift_x, shift_y, expected", [
    ('none', 'up', 0, 0, np.array([[1, 2], [3, 4]])),  # No transformation
    ('horizontal', 'down', 0, 0, np.array([[3, 4], [1, 2]])),  # Horizontal flip, down rotation
    ('vertical', 'left', 0, 0, np.array([[4, 2], [3, 1]])),  # Vertical flip, left rotation
    ('both', 'right', 0, 0, np.array([[2, 4], [1, 3]])),  # Both flips, right rotation
])
def test_erroneous_cases_that_pass(flip, rotate, shift_x, shift_y, expected):
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

#Test erroneous cases 

# Test invalid flip parameter
def test_invalid_flip():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="flip must be one of 'none', 'horizontal', 'vertical', or 'both'."):
        reposition_image(img, flip='invalid')

# Test invalid rotate parameter
def test_invalid_rotate():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="rotate must be one of 'up', 'left', 'right', or 'down'."):
        reposition_image(img, rotate='invalid')

# Test invalid shift_x parameter
def test_invalid_shift_x():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="shift_x must be an integer."):
        reposition_image(img, shift_x='a')

# Test invalid shift_y parameter
def test_invalid_shift_y():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="shift_y must be an integer."):
        reposition_image(img, shift_y='b')

# Test invalid image type
def test_invalid_image_type():
    img = 'not a numpy array'
    with pytest.raises(TypeError, match="Image format must be a numpy array."):
        reposition_image(img)


# Test invalid image dimensions
def test_invalid_image_dimensions():
    img = np.array([1, 2, 3])  # 1D array
    with pytest.raises(ValueError):
        reposition_image(img)
