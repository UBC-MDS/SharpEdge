import numpy as np
import warnings
from sharpedge._utils.utility import Utility

def reposition_image(img, flip='none', rotate='up', shift_x=0, shift_y=0):
    """
    Flip, rotate, and shift an image based on the specified requested action.
    
    This function allows repositioning of an image by applying one or more transformations 
    i.e. (flipping, rotating, and shifting). Each transformation can be controlled by the respective 
    parameters.
    
    Parameters:
    img (ndarray): The input image as a 2D numpy array (grayscale) or a 3D numpy 
                   array (RGB).
    
    flip (str, optional): Argument used to flip image. It can be:
                         - 'none': No flipping.
                         - 'horizontal': Flip the image horizontally.
                         - 'vertical': Flip the image vertically.
                         - 'both': Flip the image both horizontally and vertically.
                         Default is 'none'
    
    rotate (str, optional): Argument used to rotate image. It can be:
                           - 'up': No rotation.
                           - 'left': Rotate the image 90 degrees counter-clockwise.
                           - 'right': Rotate the image 90 degrees clockwise.
                           - 'down': Rotate the image 180 degrees (flip upside down).
                           Default is 'up'
    
    shift_x (int, optional): Argument used to shift the image along the x-axis. Default is 0.
    
    shift_y (int, optional): Argument used to shift the image along the y-axis. Default is 0.
        
    Returns:
    ndarray: The repositioned image that has been flipped, rotated, and/or shifted based on the 
             parameter values.
    
    Examples:
    >>> img = np.random.rand(100, 100)
    >>> repositioned_img = reposition_image(img, flip='horizontal', rotate='left', shift_x=10, shift_y=20)
    >>> repositioned_img = reposition_image(img_rgb, flip='both', rotate='down', shift_x=-5, shift_y=10)
    """
    # Input validation
    Utility._input_checker(img)

    # Validate flip
    valid_flips = ["none", "horizontal", "vertical", "both"]
    if flip not in valid_flips:
        raise ValueError("flip must be one of 'none', 'horizontal', 'vertical', or 'both'.")

    # Validate rotate
    valid_rotations = ["up", "left", "right", "down"]
    if rotate not in valid_rotations:
        raise ValueError("rotate must be one of 'up', 'left', 'right', or 'down'.")

    # Validate shift_x and shift_y
    if not isinstance(shift_x, int):
        raise TypeError("shift_x must be an integer.")
    if not isinstance(shift_y, int):
        raise TypeError("shift_y must be an integer.")

    # Get image dimensions
    img_height, img_width = img.shape
    
    # Check if shift values are larger than image dimensions and issue a warning if necessary
    if shift_x >= img_width or shift_y >= img_height:
        warnings.warn(f"Shift values ({shift_x}, {shift_y}) are larger than the image dimensions.", UserWarning)

    # Perform flipping
    if flip == "horizontal":
        img = np.fliplr(img)
    elif flip == "vertical":
        img = np.flipud(img)
    elif flip == "both":
        img = np.fliplr(np.flipud(img))

    # Perform rotation
    if rotate == "left":
        img = np.rot90(img, k=1)
    elif rotate == "right":
        img = np.rot90(img, k=-1)
    elif rotate == "down":
        img = np.rot90(img, k=2)

    # Perform shifting
    img = np.roll(img, shift_x, axis=1)  # Shift along x-axis
    img = np.roll(img, shift_y, axis=0)  # Shift along y-axis

    return img


