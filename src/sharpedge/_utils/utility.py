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
        # Check if the array is empty or contains zero-sized dimensions
        if img_array.size == 0 or any(dim == 0 for dim in img_array.shape):
            raise ValueError("Image size must not be zero in any dimension.")
        # Check if the image is 2D or 3D
        if img_array.ndim not in (2, 3):
            raise ValueError("Image array must be 2D or 3D.")
