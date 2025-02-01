import pytest
import numpy as np
import warnings
from sharpedge import reposition_image

# Defining fixture for the test image
@pytest.fixture
def img():
    return np.array([[1, 2], [3, 4]])

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
def test_valid_inputs(img, flip, rotate, shift_x, shift_y, expected):
    result = reposition_image(img, flip=flip, rotate=rotate, shift_x=shift_x, shift_y=shift_y)
    assert np.array_equal(result, expected), f"Unexpected result for flip={flip}, rotate={rotate}, shift_x={shift_x}, shift_y={shift_y}"

# Test edge cases
@pytest.mark.parametrize("flip, rotate, shift_x, shift_y", [
    ('none', 'up', 0, 0),  # Single-pixel image
    ('horizontal', 'down', 1, 1),  # Shift larger than dimensions
    ('vertical', 'right', -1, -1),  # Negative shift
    ('both', 'left', 0, 0),  # 1D edge case
    ('none', 'right', 0, 0),  # Right rotation results in the same orientation as original
])
@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore specific warning in tests
def test_edge_cases(flip, rotate, shift_x, shift_y):
    img = np.array([[255]])
    # Call the reposition_image function
    result = reposition_image(img, flip=flip, rotate=rotate, shift_x=shift_x, shift_y=shift_y)

# Test for 3D image (RGB image)
def test_rgb_image():
    # Simulate an RGB image (3D array)
    rgb_img = np.array([[[255, 0, 0], [0, 255, 0]],  # Red and Green pixels
                        [[0, 0, 255], [255, 255, 0]]])  # Blue and Yellow pixels

    # Test the reposition_image function with this 3D image
    result = reposition_image(rgb_img, flip='none', rotate='up', shift_x=0, shift_y=0)
    
    # Ensure the result is the same as the original (since no transformation is applied)
    assert np.array_equal(result, rgb_img), "RGB image was not processed correctly"

# Test erroneous cases 
def test_invalid_flip():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="flip must be one of 'none', 'horizontal', 'vertical', or 'both'."):
        reposition_image(img, flip='invalid')

def test_invalid_rotate():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="rotate must be one of 'up', 'left', 'right', or 'down'."):
        reposition_image(img, rotate='invalid')

def test_invalid_shift_x():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="shift_x must be an integer."):
        reposition_image(img, shift_x=1.0)

def test_invalid_shift_y():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="shift_y must be an integer."):
        reposition_image(img, shift_y=1.0)

def test_invalid_image_type():
    img = 'not a numpy array'
    with pytest.raises(TypeError, match="Image format must be a numpy array."):
        reposition_image(img)

def test_invalid_image_dimensions():
    img = np.array([1, 2, 3])  # 1D array
    with pytest.raises(ValueError):
        reposition_image(img)

