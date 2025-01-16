import pytest
import numpy as np
from frame_image import frame_image  # TO BE UPDATED

# Expected Test Cases:
@pytest.mark.parametrize("valid_img, h_border, w_border, inside, color, expected_size", [
    (np.random.randint(0, 255, (100, 100)), None, None, None, None, (140, 140)),  # Grayscale, outside black 20 x 20 border    
    (np.random.randint(0, 255, (100, 100, 3)), None, None, None, (255, 0, 0), (140, 140, 3)),  # RGB, outside 20 x 20 red border
    (np.random.randint(0, 255, (100, 100)), 10, 10, True, 255, (100, 100)),  # Grayscale, inside 10 x 10 white border
    (np.random.randint(0, 255, (100, 100, 3)), 15, 10, True, (0, 255, 0), (100, 100, 3)),  # RGB, inside 15 x 10 green border
])
def test_valid_inputs(valid_img, h_border, w_border, inside, color, expected_size):
    framed_img = frame_image(valid_img, h_border=h_border, w_border=w_border, inside=inside, color=color)
    assert framed_img.shape == expected_size


# Edge Cases: 
# Common warnings that will be reused
COMMON_WARN = {
    "user_warn_img": "The image is too small for meaningful visual information. Proceeding may not yield interpretable results.",
    "user_warn_zero": "Border size of 0 doesn't add any visual effect to the image.",
    "user_warn_inside": "The inside border exceeds 50% image size and may shrink the image significantly.",
    "user_warn_outside": "Single side border size exceeds image size."
}

@pytest.fixture
def warn_msg():
    """
    Provides the common warnings for the edge test cases.
    """
    return COMMON_WARN

@pytest.mark.parametrize("img_shape, h_border, w_border, inside, expected_shape, expected_warning, msg", [
    # Test case 1: Single Pixel Image
    (np.array([[0]]), 1, 1, None, (3, 3), UserWarning, COMMON_WARN["user_warn_img"]),  # Expecting a 3x3 image (outside border)
    (np.array([[[0, 255, 0]]]), 1, 1, None, (3, 3, 3), UserWarning, COMMON_WARN["user_warn_img"]),  # Expecting a 3x3x3 image (outside border)
    
    # Test case 2: Minimum Border Size 
    (np.random.randint(0, 255, (100, 100)), 0, 0, None, (100, 100), UserWarning, COMMON_WARN["user_warn_zero"]),  # No border, original size
    (np.random.randint(0, 255, (100, 100, 3)), 0, 0, True, (100, 100, 3), UserWarning, COMMON_WARN["user_warn_zero"]),  # No border, original size (100x100 with 3 channels)
    (np.random.randint(0, 255, (100, 100, 3)), 0, 20, True, (100, 100, 3), UserWarning, COMMON_WARN["user_warn_zero"]),  # No height border, original size on height (100x140 with 3 channels)
    
    # Test case 3: Large Outside Border
    (np.random.randint(0, 255, (100, 100)), 200, 200, None, (500, 500), UserWarning, COMMON_WARN["user_warn_outside"]),  # Explicit border size, expect 500x500
    (np.random.randint(0, 255, (100, 100, 3)), 200, 200, None, (500, 500, 3), UserWarning, COMMON_WARN["user_warn_outside"]),  # Valid border, expect 500x500 with 3 channels
    
    # Test case 4: Very Small Image with Large Inside Border 
    (np.random.randint(0, 255, (5, 5)), 1, 1, True, (3, 3), UserWarning, COMMON_WARN["user_warn_inside"]),  # Large inside border, expect 3x3 image
    (np.random.randint(0, 255, (10, 10, 3)), 3, 3, True, (4, 4, 3), UserWarning, COMMON_WARN["user_warn_inside"]),  # Large inside border, expect 3x3 image with 3 channels
])
def test_edge_cases(img_shape, h_border, w_border, inside, expected_shape, expected_warning, msg):
    """
    Combined edge cases for frame_image function (with no errors expected).
    This test combines multiple edge cases in one function using pytest parametrize.
    """
    with pytest.warns(expected_warning, match = msg):
        framed_img = frame_image(img_shape, h_border=h_border, w_border=w_border, inside=inside, color=0)

    # Check if the output shape matches the expected shape
    assert framed_img.shape == expected_shape


# Error Cases:
# Common error descriptions that will be reused
COMMON_DESC = {
    "type_err_int": "Each color component must be an integer.",
    "type_err_oth": "Color must be either an integer for grayscale frames or a tuple/list of 3 integers for RGB frames.",
    "type_err_border": "Both h_border and w_border must be integers.",
    "value_err_len": "For RGB frames, color must be a tuple or list of 3 integers.",
    "value_err_rgb": "Each color component must be in the range 0 to 255.",
    "value_err_gs": "For grayscale frames, color must be an integer in the range 0 to 255.",
    "value_err_rel": "The inside border is too large for this small image. The image cannot be processed.",
    "value_err_border": "Both h_border and w_border must be non-negative integers."
}

@pytest.fixture
def error_description():
    """
    Provides the common error descriptions for the error test cases.
    """
    return COMMON_DESC

@pytest.mark.parametrize("img_shape, color, h_border, w_border, inside, expected_exception, description", [
    # Category I: Invalid `color` Argument
    # 1. Invalid `color` format
    (np.random.randint(0, 255, (5, 5)), "red", 20, 20, False, TypeError, COMMON_DESC["type_err_oth"]),  
    (np.random.randint(0, 255, (5, 5, 3)), (255, '255', 0), 20, 20, False, TypeError, COMMON_DESC["type_err_oth"]), 
    (np.random.randint(0, 255, (5, 5, 3)), np.array([255, 0, 0]), 20, 20, False, TypeError, COMMON_DESC["type_err_oth"]),   

    # 2. Missing or redundant `color` channels
    (np.random.randint(0, 255, (5, 5)), (255, 0), 2, 2, False, ValueError, COMMON_DESC["value_err_rgb"]),  
    (np.random.randint(0, 255, (5, 5, 3)), (255, 0), 2, 2, False, ValueError, COMMON_DESC["value_err_rgb"]),  
    (np.random.randint(0, 255, (5, 5)), (255, 0, 0, 0), 2, 2, False, ValueError, COMMON_DESC["value_err_rgb"]),  
    (np.random.randint(0, 255, (5, 5, 3)), (255, 0, 0, 0), 2, 2, False, ValueError, COMMON_DESC["value_err_rgb"]),  

    # 3. Unsupported `color` data type
    (np.random.randint(0, 255, (5, 5)), (111.1, 111.1, 0), 20, 20, False, TypeError, COMMON_DESC["type_err_int"]),  
    (np.random.randint(0, 255, (5, 5, 3)), (111.1, 111.1, 0), 20, 20, False, TypeError, COMMON_DESC["type_err_int"]),  
    (np.random.randint(0, 255, (5, 5)), 3.5, 20, 20, False, TypeError, COMMON_DESC["type_err_oth"]),  
    (np.random.randint(0, 255, (5, 5, 3)), 3.5, 20, 20, False, TypeError, COMMON_DESC["type_err_oth"]),  

    # 4. Out-of-range `color` values
    (np.random.randint(0, 255, (5, 5)), (-1, 255, 0), 20, 20, False, ValueError, COMMON_DESC["value_err_gs"]),  
    (np.random.randint(0, 255, (5, 5)), 256, 20, 20, False, ValueError, COMMON_DESC["value_err_gs"]),  
    (np.random.randint(0, 255, (5, 5, 3)), (255, 256, 0), 20, 20, False, ValueError, COMMON_DESC["value_err_rgb"]), 
    (np.random.randint(0, 255, (5, 5, 3)), 256, 20, 20, False, ValueError, COMMON_DESC["value_err_rgb"]),  


    # Category II: Too Large Inside Border Size
    # 5. Too large inside border (grayscale and RGB) - image too small
    (np.random.randint(0, 255, (5, 5)), None, 20, 20, True, ValueError, COMMON_DESC["value_err_rel"]),   
    (np.random.randint(0, 255, (10, 10, 3)), 20, 20, True, ValueError, COMMON_DESC["value_err_rel"]),  


    # Category III: Invalid `h_border` and `w_border` Arguments
    # 6. Invalid `*_border` format
    (np.random.randint(0, 255, (5, 5)), "20", "20", False, TypeError, COMMON_DESC["type_err_border"]),  
    (np.random.randint(0, 255, (5, 5, 3)), (20), (20), False, TypeError, COMMON_DESC["type_err_border"]),  
    (np.random.randint(0, 255, (5, 5)), [20], [20], False, TypeError, COMMON_DESC["type_err_border"]),  
    (np.random.randint(0, 255, (5, 5, 3)), [], [], False, TypeError, COMMON_DESC["type_err_border"]),  

    # 7. Unsupported `*_border` data type
    (np.random.randint(0, 255, (5, 5)), 3.5, 3.5, False, TypeError, COMMON_DESC["type_err_border"]),  
    (np.random.randint(0, 255, (5, 5, 3)), 3.5, 3.5, False, TypeError, COMMON_DESC["type_err_border"]),  

    # 8. Out-of-range `*_border` values
    (np.random.randint(0, 255, (5, 5)), -1, -1, False, ValueError, COMMON_DESC["value_err_border"]),  
    (np.random.randint(0, 255, (5, 5, 3)), -1, -1, False, ValueError, COMMON_DESC["value_err_border"]),  
])
def test_error_cases(img_shape, color, h_border, w_border, inside, expected_exception, description):
    """
    Test function for various error cases in the `frame_image()` function.

    It checks the following scenarios:
    1. Invalid color input (wrong format, incorrect components, wrong type, out of range).
    2. Too large border for small image.
    3. Invalid border inputs (wrong format, wrong type, negative values).

    Each case expects the correct exception and error description to be raised.
    """
    with pytest.raises(expected_exception, match = description):
        frame_image(img_shape, h_border=h_border, w_border=w_border, inside=inside, color=color)

