"""
This module contains utility classes and functions for internal use only.
"""
class Utility:
    @staticmethod
    def _input_checker(img_array):
        # Check if the array is empty or contains zero-sized dimensions
        if img_array.size == 0 or any(dim == 0 for dim in img_array.shape):
            raise ValueError("Image size must not be zero in any dimension.")

        # Check if the image is 2D or 3D
        if img_array.ndim not in (2, 3):
            raise ValueError("Image array must be 2D or 3D.")
        
        # If all checks pass, return None to indicate no error
        return None


        

        