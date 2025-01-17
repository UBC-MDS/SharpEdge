import numpy as np
from sharpedge._utils.utility import Utility

def pooling_image(img, window_size, pooling_method=np.mean):
    """
    Perform pooling on an image using a specified window size and pooling function.
    
    This function reduces the size of an input image, by dividing the image into non-overlapping 
    windows of a specified size by implementing a pooling function (e.g., mean, max, or min) to 
    each window. 
    
    Parameters:
    img (ndarray): The input image as a 2D numpy array (grayscale) or a 3D numpy 
                   array (RGB).
    
    window_size (int): The size of the pooling window (e.g., 10 for 10x10 windows). 
                                        
    pooling_method (callable, optional): The pooling function that will be used on each window. 
                                         Common functions used are np.mean, np.median, np.max, np.min.
                                         Default is np.mean.
        

    Returns:
    ndarray: The resized image that was reshaped based off of respective pooling function and
             window size.
    
    Examples:
    >>> img = np.random.rand(100, 100)
    >>> pooled_img = pooling_image(img, window_size=10, pooling_method=np.mean)
    >>> pooled_img = pooling_image(img_rgb, window_size=20, pooling_method=np.max)
    """
    # Input validation
    Utility._input_checker(img)

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

    # Recalculate the new rows and columns after cropping (if needed)
    rows, cols = img.shape[:2]

    # Initialize the result array with appropriate dimensions
    result_rows = rows // window_size
    result_cols = cols // window_size

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

    return pooled_image
