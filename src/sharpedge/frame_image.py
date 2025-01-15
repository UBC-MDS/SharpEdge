def frame_image(img, h_border=20, w_border=20, inside=False, color=0):
    """
    Add a decorative frame around the image with customizable color.

    This function adds a border around the input image, either inside the image 
    (preserving its original size) or outside (increasing its size). The border 
    color can be specified for both grayscale and RGB images.

    Parameters:
    img (ndarray): The input image as a 2D numpy array (grayscale) or 3D numpy 
                   array (RGB).
    h_border (int, optional): The height of the border in pixels. Default is 20.
    w_border (int, optional): The width of the border in pixels. Default is 20.
    inside (bool, optional): If True, the border is added **inside** the image 
                              (maintaining the image size). If False, the border 
                              is added **outside** the image (increasing the image size). 
                              Default is False.
    color (int or tuple/list, optional): The color of the border. Can be:
        - A single value for grayscale frames (e.g., 0 for black, 255 for white).
        - A tuple/list of 3 values for RGB frames (e.g., (0, 0, 0) for black).
        Default is 0 (black) for grayscale frames.

    Returns:
    ndarray: The framed image with the applied border.

    Examples:
    >>> img = np.random.rand(100, 100)
    >>> framed_img = frame_image(img, h_border=30, w_border=30, inside=True, color=255)
    >>> framed_img_rgb = frame_image(img_rgb, h_border=20, w_border=20, inside=False, color=(255, 0, 0))
    """
    # Check that the color input is correct for grayscale or RGB image
    if isinstance(color, (tuple, list)):
        if len(color) != 3:
            raise ValueError("For RGB frames, color must be a tuple or list of 3 integers.")
        for rgb_c in color:
            if not isinstance(rgb_c, int):
                raise TypeError("Each color component must be an integer.")
            if not (0 <= rgb_c <= 255):
                raise ValueError("Each color component must be in the range 0 to 255.")
    elif isinstance(color, int):
        if not (0 <= color <= 255):
            raise ValueError("For grayscale frames, color must be an integer in the range 0 to 255.")
    else:
        raise TypeError("Color must be either an integer for grayscale frames or a tuple/list of 3 integers for RGB frames.")
    
    
    if isinstance(color, tuple) or isinstance(color, list):
        # For RGB images, ensure the color is in the correct shape
        color = np.array(color, dtype=img.dtype)
    
    if inside:
        # Add border within the image (keeping size constant)
        framed_img = np.pad(img[h_border:-h_border, w_border:-w_border],
                            ((h_border, h_border), (w_border, w_border)),
                            'constant', constant_values=color)
    else:
        # Add border outside the image (increasing size)
        framed_img = np.pad(img,
                            ((h_border, h_border), (w_border, w_border)),
                            'constant', constant_values=color)

    return framed_img