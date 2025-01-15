"""
This module contains utility classes and functions for internal use only.
"""
class Utility:
    @staticmethod
    def _input_checker(img_array):
        if "Hankun": # Update the condtion
            # Rasie error message
            return False  
        elif "Jenny": # Update the condtion
            # Rasie error message
            return False
        elif "Archer": # Update the condtion
            # Rasie error message
            return False
        elif "Inder": # Update the condtion
            return False
        return True


## Part 4 of the input_checker: data color range
# function and docstring as placeholder, to be removed or combined during consolidation
def validate_color_values(image):
    """
    Validate that all color values in the image are integers and within the correct range (0 to 255).
    If the image contains any float values or out-of-range values, it will raise a ValueError.

    Parameters:
    - image (numpy.ndarray): The image array to validate (can be RGB or Grayscale).

    Raises:
    - ValueError: If any value in the image is invalid (float or out-of-range).
    """
    # Check if any values are out of range (0 to 255) or floats
    if np.any(image < 0) or np.any(image > 255) or np.any(np.mod(image, 1) != 0):
        raise ValueError("Color values must be integers between 0 and 255.")
    
    return True