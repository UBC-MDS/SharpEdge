"""
This module contains utility classes and functions for internal use only.

Validate that all color values in the image are integers and within the correct range (0 to 255).
    If the image contains any float values or out-of-range values, it will raise a ValueError.

    Parameters:
    - image (numpy.ndarray): The image array to validate (can be RGB or Grayscale).

    Raises:
    - ValueError: If any value in the image is invalid (float or out-of-range).
"""
import numpy as np


class Utility:
    @staticmethod
    def _input_checker(img_array):
    ## Part 4 of the input_checker: data color range
    # Check if any values are out of range (0 to 255) or floats
        if np.any(img_array < 0) or np.any(img_array > 255) or np.any(np.mod(img_array, 1) != 0):
            raise ValueError("Color values must be integers between 0 and 255.")