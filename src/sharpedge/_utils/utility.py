"""
This module contains utility classes and functions for internal use only.
"""
import numpy as np


class Utility:
    @staticmethod
    def _input_checker(img_array):
        # Check the image format: must be a numpy array
        if not isinstance(img_array, np.ndarray):
            raise TypeError("Image format must be a numpy array.")
        return True
