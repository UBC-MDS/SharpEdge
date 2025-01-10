def modulate_image(img, mode='gray', ch_extract=None, ch_swap=None):
    """
    Convert or manipulate image color channels with flexibility for grayscale and RGB.

    This function allows you to perform various color transformations on an image, including:
    - Converting between grayscale and RGB formats.
    - Extracting specific RGB channels (e.g., Red, Green, or Blue).
    - Swapping RGB channels to rearrange the color channels.
    
    It supports both grayscale (2D) and RGB (3D) images. If a grayscale image is provided, 
    channel extraction or swapping will not be applicable, and a notification will be given.

    If the input image is already in the target mode (e.g., 'gray' or 'rgb'), the function will notify
    that no conversion is necessary and return the original image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array. This can be either a 2D numpy array (grayscale image) or a 3D numpy array 
        (RGB image). The dimensions of the image should be (height, width) for grayscale or 
        (height, width, 3) for RGB images.
    
    mode : str, optional
        The desired target color scale. This can be either:
        - `'gray'`: Convert the image to grayscale.
        - `'rgb'`: Convert the image to RGB.
        Default is `'gray'`.
        
        If the input image is already in the target mode, a notification will be printed, and the 
        function will return the input image as-is without any conversion.
        
    ch_extract : list of int, optional
        A list of integers representing the indices of the RGB channels to extract. For example:
        - `[0]`: Extract only the Red channel.
        - `[1]`: Extract only the Green channel.
        - `[2]`: Extract only the Blue channel.
        
        If `None`, no channel extraction occurs. Default is `None`.
        
        **Note**: This option is only applicable for RGB images. For grayscale images, extraction 
        is not possible, and a notification will be displayed.

    ch_swap : list of int, optional
        A list of integers representing the new order of the RGB channels. The list should contain 
        exactly three elements, each of which is an index corresponding to the RGB channels:
        - `[0, 1, 2]` will keep the channels in their original order (Red, Green, Blue).
        - `[2, 1, 0]` will swap Red and Blue channels.
        
        If `None`, no channel swapping occurs. Default is `None`.
        
        **Note**: This option is only applicable for RGB images. For grayscale images, swapping 
        channels is unnecessary, and a notification will be displayed.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the manipulated image. The output could be:
        - A grayscale image (2D array).
        - An RGB image (3D array).
        - A subset of RGB channels (e.g., only the Red channel).
        - A rearranged RGB image with swapped channels.

    Raises
    ------
    ValueError
        If the input image is not in grayscale or RGB format, or if any invalid channel indices are 
        provided for extraction or swapping.

    Notes
    ------
    - Grayscale images (2D arrays) do not have multiple color channels, so channel extraction or 
      swapping will not be possible. These operations will be skipped with a corresponding notification.
    - If no operations are specified (i.e., no conversion or channel manipulation), the function will
      return the original image.
    - If the input image is already in the target mode (e.g., 'gray' or 'rgb'), the function will
      notify that no conversion is necessary and return the image as-is.
      
    Examples
    --------
    # Convert an RGB image to grayscale
    grayscale_image = modulate_image(rgb_image, mode='gray')

    # Convert a grayscale image back to RGB
    rgb_image_again = modulate_image(grayscale_image, mode='rgb')

    # Extract the Red channel from an RGB image
    red_channel = modulate_image(rgb_image, ch_extract=[0])

    # Extract the Red and Green channels from an RGB image
    red_green_channels = modulate_image(rgb_image, ch_extract=[0, 1])

    # Swap the Red and Blue channels in an RGB image
    swapped_image = modulate_image(rgb_image, ch_swap=[2, 0, 1])
    
    """
