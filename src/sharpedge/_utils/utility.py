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
        # Check input image array shape
        if not (img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] == 3)):
            raise ValueError(
            f"Invalid image shape: {img_array.shape}. "
            "Expected 2D (grayscale) or 3D with 3 channels (RGB)."
        )