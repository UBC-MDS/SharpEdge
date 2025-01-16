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
        - A single value for grayscale images (e.g., 0 for black, 255 for white).
        - A tuple/list of 3 values for RGB images (e.g., (0, 0, 0) for black).
        Default is 0 (black) for grayscale images.

    Returns:
    ndarray: The framed image with the applied border.

    Examples:
    >>> img = np.random.rand(100, 100)
    >>> framed_img = frame_image(img, h_border=30, w_border=30, inside=True, color=255)
    >>> framed_img_rgb = frame_image(img_rgb, h_border=20, w_border=20, inside=False, color=(255, 0, 0))
    """