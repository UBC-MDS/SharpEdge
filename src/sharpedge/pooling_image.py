import numpy as np
from sharpedge._utils.utility import Utility

def pooling_image(img, window_size, pooling_method=np.mean, normalize_rgb=True):
    """
    Perform pooling on an image using a specified window size and pooling function.
    
    This function reduces the size of an input image by dividing it into non-overlapping 
    windows of a specified size and applying a pooling function (e.g., mean, max, or min) 
    to each window.
    
    Parameters:
    img (ndarray): The input image as a 2D numpy array (grayscale) or a 3D numpy array (RGB).
    
    window_size (int): The size of the pooling window (e.g., 10 for 10x10 windows).
                                        
    pooling_method (callable, optional): The pooling function applied to each window. 
                                         Common functions: np.mean, np.median, np.max, np.min.
                                         Default is np.mean.

    normalize_rgb (bool, optional): If True, normalize RGB images to the range [0.0, 1.0].
                                    If False, return rounded uint8 images in the range [0, 255].
                                    Default is True.

    Returns:
    ndarray: The resized image after pooling, normalized or in uint8 format.
    """
    # Input validation
    #Utility._input_checker(img)

    if not isinstance(window_size, int):
        raise TypeError("window_size must be an integer.")
    
    if not callable(pooling_method):
        raise TypeError("pooling_method must be callable.")
    
    img_rows, img_cols = img.shape[:2]

    # Check if dimensions are divisible by window size
    if img_rows % window_size != 0 or img_cols % window_size != 0:
        raise ValueError("Image dimensions are not divisible by the window size.")

    # Ensure image is in float32 format for calculations
    img = img.astype(np.float32)

    # Initialize the result array with appropriate dimensions
    result_rows = img_rows // window_size
    result_cols = img_cols // window_size

    if img.ndim == 2:  # Grayscale image
        pooled_image = np.zeros((result_rows, result_cols))
        for i in range(result_rows):
            for j in range(result_cols):
                window = img[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
                pooled_image[i, j] = pooling_method(window)
    else:  # RGB image
        pooled_image = np.zeros((result_rows, result_cols, img.shape[2]))
        for i in range(result_rows):
            for j in range(result_cols):
                window = img[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size, :]
                for c in range(img.shape[2]):
                    pooled_image[i, j, c] = pooling_method(window[:, :, c])
        
        # Normalize or convert to uint8 based on the flag
        if normalize_rgb:
            pooled_image /= 255.0  # Normalize to [0.0, 1.0]
        else:
            pooled_image = np.round(pooled_image).astype(np.uint8)  # Convert to uint8

    return pooled_image
