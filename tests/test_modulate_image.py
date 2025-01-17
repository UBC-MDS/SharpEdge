import pytest
import numpy as np
import warnings
from sharpedge.modulate_image import modulate_image

@pytest.fixture
def img_dict():
    """
    Fixture that returns a dictionary with input and expected output arrays for testing.
    """
    # Dictionary of inputs and expected outputs
    img_dict = {
        "img_rgb": np.full((5, 5, 3), [100, 150, 200], dtype=np.uint8),  # Creates a 5x5 RGB image with same values,
        "img_gray": np.full((5, 5), 100, dtype=np.uint8),  # Creates a 5x5 grayscale image with value 100 for all pixels
        "expected_rgb_to_gray": np.full((5, 5), 150, dtype=np.uint8),  # Averaging RGB values
        "expected_gray_to_rgb": np.full((5, 5, 3), [100, 100, 100], dtype=np.uint8),  # Grayscale to RGB conversion
        "expected_rgb_swap": np.full((5, 5, 3), [200, 150, 100], dtype=np.uint8),  # Swap Red and Blue channels
        "expected_rgb_extract": np.full((5, 5, 2), [100, 150], dtype=np.uint8),  # Extract Red and Green channels
        "expected_rgb_swap_extract": np.full((5, 5, 2), [200, 150], dtype=np.uint8),  # Swap and Extract Red and Green
    }
    return img_dict

# Expected Test Cases:
@pytest.mark.parametrize(
    "test_img, mode, ch_swap, ch_extract, expected_output",
    [
        ("img_gray", 'rgb', None, None, "expected_gray_to_rgb"),  # Grayscale image to RGB
        ("img_rgb", 'gray', None, None, "expected_rgb_to_gray"),  # RGB image to grayscale
        ("img_gray", 'rgb', [2, 1, 0], None, "expected_rgb_swap"),  # Swap Red and Blue channels
        ("img_rgb", 'rgb', None, [0, 1], "expected_rgb_extract"),  # Extract Red and Green channels
        ("img_gray", 'rgb', [2, 1, 0], [0, 1], "expected_rgb_swap_extract"),  # Swap and Extract Red and Green channels
    ]
)
def test_valid_inputs(img_dict, test_img, mode, ch_swap, ch_extract, expected_output):
    """
    Test the modulate_image function using various modes and transformations.
    """
    # Retrieve the actual image and expected output from the fixture
    test_img_array = img_dict[test_img]
    expected_output_array = img_dict[expected_output]

    # Run the function
    result = modulate_image(test_img_array, mode=mode, ch_swap=ch_swap, ch_extract=ch_extract)
    
    # Check that the result matches the expected output exactly
    np.testing.assert_array_equal(result, expected_output_array)


# Edge Cases:**Test for Warnings**
@pytest.fixture
def warn_msg():
    """
    Provides the common warnings for the edge test cases.
    """
    COMMON_WARN = {
        "user_warn_rgb": "Input is already RGB. No conversion needed.",
        "user_warn_gray": "Input is already grayscale. No conversion needed.",
        "user_warn_swap_extr": "Grayscale images have no channels to swap or extract."
    }
    return COMMON_WARN

@pytest.mark.parametrize(
    "img, mode, ch_swap, ch_extract, expected_output, expected_warning, msg_key",
    [
        # Scenario 1: No change needed when input and conversion mode the same
        ("img_rgb", 'rgb', None, None, "img_rgb", UserWarning, "user_warn_rgb"),  # RGB to RGB 
        ("img_gray", 'gray', None, None, "img_gray", UserWarning,"user_warn_gray" )  # Grayscale to Grayscale 

        # Scenario 2: Grayscale not qualified for color swap or extraction
        ("img_gray", 'gray', [2, 1, 0], None, "img_gray", UserWarning, "user_warn_swap_extr")
        ("img_gray", 'gray', None, [0, 1], "img_gray", UserWarning, "user_warn_swap_extr")

    ]
)
def test_edge_cases(image_data, test_img, mode, ch_swap, ch_extract, expected_output, expected_warning, msg):
    """
    Test edge cases where no transformation should occur or where warnings should be raised.
    """
    # Retrieve the actual image and expected output from the fixture dictionary
    test_img_array = image_data[test_img]  # Access input image array
    expected_output_array = image_data[expected_output]  # Access expected output array
    msg = COMMON_WARN[msg_key]  # Access expected warning msg

    # Use pytest's `warns` to check that a warning is raised during the function call
    with pytest.warns(expected_warning, match=msg):
        result = modulate_image(test_img_array, mode=mode, ch_swap=ch_swap, ch_extract=ch_extract)
    
    # Check that the result matches the expected output exactly
    np.testing.assert_array_equal(result, expected_output_array)


# Error Cases: These are the cases where the function should raise errors
@pytest.mark.parametrize(
    "img, mode, ch_extract, ch_swap, expected_exception",
    [
        # Invalid mode
        ('rgb', 'invalid_mode', None, None, ValueError),  # Invalid mode value
        # Invalid ch_extract
        ('rgb', 'rgb', [3], None, ValueError),  # Invalid channel index in ch_extract
        ('rgb', 'rgb', 'wrong_type', None, TypeError),  # ch_extract should be a list/tuple
        # Invalid ch_swap
        ('rgb', 'rgb', None, [0, 1], ValueError),  # Invalid ch_swap (not enough elements)
        ('rgb', 'rgb', None, [0, 1, 3], ValueError),  # Invalid ch_swap (index out of range)
        ('rgb', 'rgb', None, [0, 1, 1], ValueError),  # Invalid ch_swap (duplicate indices)
        ('gray', 'rgb', None, [0, 1, 2], UserWarning),  # Trying to swap channels on grayscale
    ]
)
def test_error_cases(sample_images, img, mode, ch_extract, ch_swap, expected_exception):
    img_rgb, img_gray = sample_images

    # Select the image based on the mode
    if img == 'rgb':
        test_img = img_rgb
    else:
        test_img = img_gray

    if ch_extract is not None:
        with pytest.raises(expected_exception):
            modulate_image(test_img, mode=mode, ch_extract=ch_extract, ch_swap=ch_swap)
    else:
        with pytest.raises(expected_exception):
            modulate_image(test_img, mode=mode, ch_extract=ch_extract, ch_swap=ch_swap)

