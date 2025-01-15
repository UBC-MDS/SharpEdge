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
        # Check if the image size is non-zero for both 2D and 3D arrays
        if img_array.ndim not in (2, 3):
            raise ValueError("Image array must be 2D or 3D.")
        if img_array.size == 0 or any(dim == 0 for dim in img_array.shape):
            raise ValueError("Image size must not be zero in any dimension.")
        
        return True
        