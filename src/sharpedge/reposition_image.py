def reposition_image(img, flip='none', rotate='up', shift_x=0, shift_y=0):
    """
    Flip, rotate, and shift an image based on the specified modes.
    
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
