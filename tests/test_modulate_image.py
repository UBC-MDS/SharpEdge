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
        "expected_rgb_to_gray": np.full((5, 5), 140, dtype=np.uint8),  # Averaging RGB values using the luminosity method
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
        ("img_rgb", 'as-is', (2, 1, 0), None, "expected_rgb_swap"),  # Swap Red and Blue channels
        ("img_rgb", 'as-is', None, [0, 1], "expected_rgb_extract"),  # Extract Red and Green channels
        ("img_rgb", 'as-is', [2, 1, 0], [0, 1], "expected_rgb_swap_extract"),  # Swap and Extract Red and Green channels
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
    warn_msg = {
        "user_warn_no_change": "Mode is 'as-is' and no channel operations are specified. Return the original image.",
        "user_warn_rgb": "Input is already RGB. No conversion needed.",
        "user_warn_gray": "Input is already grayscale. No conversion needed.",
        "user_warn_swap_extr": "Grayscale images have no channels to swap or extract.",
        "user_warn_len_0": "No channels specified for ch_extract. Return the output image with no extraction.",
        "user_warn_swap_default": "Input is in default channel order. No swap needed."
    }
    return warn_msg

@pytest.mark.parametrize(
    "test_img, mode, ch_swap, ch_extract, expected_output, expected_warning, msg_key",
    [
        # Scenario 1: No change when mode is 'as-is' with no optional arguments
        ("img_rgb", 'as-is', None, None, "img_rgb", UserWarning, "user_warn_no_change"),
        
        # Scenario 2: No change needed when input and conversion mode the same
        ("img_rgb", 'rgb', None, None, "img_rgb", UserWarning, "user_warn_rgb"),  # RGB to RGB 
        ("img_gray", 'gray', None, None, "img_gray", UserWarning, "user_warn_gray"),  # Grayscale to Grayscale 

        # Scenario 3: Grayscale not qualified for color swap or extraction
        ("img_gray", 'as-is', [2, 1, 0], None, "img_gray", UserWarning, "user_warn_swap_extr"),
        ("img_rgb", 'gray', None, [0, 1], "expected_rgb_to_gray", UserWarning, "user_warn_swap_extr"),
        
        # Scenario 4: Empty `ch_extract` tuple or list
        ("img_rgb", 'as-is', None, (), "img_rgb", UserWarning, "user_warn_len_0"),  
        ("img_rgb", 'as-is', [2, 1, 0], [], "expected_rgb_swap", UserWarning, "user_warn_len_0"), 

        # Scenario 5: Default channel order selected for ch_swap 
        ("img_rgb", 'as-is', [0, 1, 2], None, "img_rgb", UserWarning, "user_warn_swap_default") 
    ]
)
def test_edge_cases(img_dict, warn_msg, test_img, mode, ch_swap, ch_extract, expected_output, expected_warning, msg_key):
    """
    Test edge cases where no transformation should occur or where warnings should be raised.
    """
    # Retrieve the actual image and expected output from the fixture dictionary
    test_img_array = img_dict[test_img]  # Access input image array
    expected_output_array = img_dict[expected_output]  # Access expected output array
    msg = warn_msg[msg_key]  # Access expected warning msg

    # Use pytest's `warns` to check that a warning is raised during the function call
    with pytest.warns(expected_warning, match=msg):
        result = modulate_image(test_img_array, mode=mode, ch_swap=ch_swap, ch_extract=ch_extract)
    
    # Check that the result matches the expected output exactly
    np.testing.assert_array_equal(result, expected_output_array)


# Error Cases: These are the cases where the function should raise errors
@pytest.fixture
def error_desc():
    """
    Provides the common error descriptions for the error test cases.
    """
    error_desc = {
        "type_err_swap_type": "ch_swap must be a list or tuple.",
        "type_err_swap_int": "All elements in ch_swap must be integers.",
        "type_err_extr_type": "ch_extract must be a list or tuple.",
        "type_err_extr_int": "All elements in ch_extract must be integers.",
        "value_err_mode": "Invalid mode. Mode must be 'as-is', 'gray' or 'rgb'.",
        "value_err_swap_len_ind": "ch_swap must be three elements of valid RGB channel indices 0, 1, or 2.",
        "value_err_swap_no_dup": "ch_swap must include all channels 0, 1, and 2 exactly once.",
        "value_err_extr_len": "ch_extract can contain a maximum of 2 elements. Use ch_swap for 3-element extraction, swapping equivalent.",
        "value_err_extr_ind": "Invalid channel indices. Only 0, 1, or 2 are valid.",
        "value_err_extr_no_dup": "ch_extract contains duplicate channel indices."
    }

    return error_desc

@pytest.mark.parametrize(
    "test_img, mode, ch_swap, ch_extract, expected_exception, desc_key",
    [
        # Category I: Invalid `mode`
        ("img_gray", 'invalid', None, None, ValueError, "value_err_mode"), 
        ("img_rgb", 'invalid', None, None, ValueError, "value_err_mode"),
        
        # Category II: Invalid `ch_swap`
        # 1. Incorrect input data type
        ("img_gray", 'rgb', [0.0, 1.0, 2.0], None, TypeError, "type_err_swap_int"),
        ("img_gray", 'rgb', (0.1, 1.1, 2.1), None, TypeError, "type_err_swap_int"), 
        ("img_gray", 'rgb', 'wrong_type', None, TypeError, "type_err_swap_type"), 
        ("img_gray", 'rgb', np.array([0, 1, 2]), None, TypeError, "type_err_swap_type"),


        # 2. Incorrect indices or length
        ("img_gray", 'rgb', [0, 1], None, ValueError, "value_err_swap_len_ind"),  # Not enough elements
        ("img_gray", 'rgb', (0, 1, 1, 2), None, ValueError, "value_err_swap_len_ind"),  # Too many elements
        ("img_gray", 'rgb', (0, 2, 3), None, ValueError, "value_err_swap_len_ind"),  # Index out of range
        ("img_gray", 'rgb', [3], None, ValueError, "value_err_swap_len_ind"),  # Index out of range
        
        # 3. Duplicate indices
        ("img_gray", 'rgb', [0, 1, 1], None, ValueError, "value_err_swap_no_dup"),  
        ("img_gray", 'rgb', (2, 2, 2), None, ValueError, "value_err_swap_no_dup"),  

        
        # Category III: Invalid `ch_extract`
        # 4. Incorrect input data type
        ("img_gray", 'rgb', None, [0.0], TypeError, "type_err_extr_int"),
        ("img_gray", 'rgb', None, (0.1, 1.1), TypeError, "type_err_extr_int"), 
        ("img_gray", 'rgb', None, 'wrong_type', TypeError, "type_err_extr_type"), 
        ("img_gray", 'rgb', None, np.array([0, 1]), TypeError, "type_err_extr_type"),

        # 5. Out-of-range length
        ("img_gray", 'rgb', None, [0, 1, 2], ValueError, "value_err_extr_len"),  # Too many elements
        ("img_gray", 'rgb', None, (0, 1, 1, 2), ValueError, "value_err_extr_len"),  # Too many elements
        
        # 6. Incorrect indices
        ("img_gray", 'rgb', None, (0, 3), ValueError, "value_err_extr_ind"),  # Index out of range
        ("img_gray", 'rgb', None, [3], ValueError, "value_err_extr_ind"),  # Index out of range
        
        # 7. Duplicate indices
        ("img_gray", 'rgb', None, [1, 1], ValueError, "value_err_extr_no_dup"),  
        ("img_gray", 'rgb', None, (0, 0), ValueError, "value_err_extr_no_dup"),  
    ]
)
def test_error_cases(img_dict, error_desc, test_img, mode, ch_swap, ch_extract, expected_exception, desc_key):
    """
    Test error cases where expected errors should be raised.
    """
    # Retrieve the actual image and expected output from the fixture dictionary
    test_img_array = img_dict[test_img]  # Access input image array
    description = error_desc[desc_key]  # Access expected warning msg

    # Use pytest's `raises` to raise the expected error during the function call
    with pytest.raises(expected_exception, match = description):
        modulate_image(test_img_array, mode=mode, ch_extract=ch_extract, ch_swap=ch_swap)
