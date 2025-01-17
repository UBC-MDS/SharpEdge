import numpy as np
import warnings

def modulate_image(img, mode='gray', ch_swap=None, ch_extract=None):
    """
    Convert or manipulate image color channels with flexibility for grayscale and RGB.

    This function allows you to perform various color transformations on an image, including:
    - Converting between grayscale and RGB formats.
    - Swapping RGB channels to rearrange the color channels.
    - Extracting specific RGB channels (e.g., Red, Green, or Blue).
    
    It supports both grayscale (2D) and RGB (3D) images. If a grayscale image is provided, 
    channel swapping or extraction will not be applicable, and a notification will be given.

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
        
    ch_swap : list/tuple of int, optional
        A list or tuple of integers representing the new order of the RGB channels. The list should contain 
        exactly three elements, each of which is an index corresponding to the RGB channels:
        - `[0, 1, 2] or (0, 1, 2)` will keep the channels in their original order (Red, Green, Blue).
        - `[2, 1, 0] or (2, 1, 0)` will swap Red and Blue channels.
        
        If `None`, no channel swapping occurs. Default is `None`.
        
        **Note**: This option is only applicable for RGB images. For grayscale images, swapping 
        channels is unnecessary, and a notification will be displayed.
    
    ch_extract : list/tuple of int, optional
        A list or tuple of integers representing the indices of the RGB channels to extract. For example:
        - `[0] or (0)`: Extract only the Red channel.
        - `[1] or (1)`: Extract only the Green channel.
        - `[2] or (2)`: Extract only the Blue channel.
        
        If `None`, no channel extraction occurs. Default is `None`.
        
        **Note**: This option is only applicable for RGB images. For grayscale images, extraction 
        is not possible, and a notification will be displayed. 

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
    swapped_image = modulate_image(rgb_image, ch_swap=(2, 0, 1))
    
    """

    # Validate 'mode' input
    if mode not in ['gray', 'rgb']:
        raise ValueError(f"Invalid mode '{mode}'. Mode must be either 'gray' or 'rgb'.")

    # Handle grayscale mode (2D array) and RGB mode (3D array)
    if mode == 'gray' and len(img.shape) == 2:
        warnings.warn("Input is already grayscale. No conversion needed.", UserWarning)
        return img
    if mode == 'rgb' and len(img.shape) == 3:
        warnings.warn("Input is already RGB. No conversion needed.", UserWarning)
        return img

    # Convert grayscale to RGB if requested
    if mode == 'rgb' and len(img.shape) == 2:
        print("Converting grayscale to RGB...")
        img = np.stack([img] * 3, axis=-1)

    # Convert RGB to grayscale if requested
    if mode == 'gray' and len(img.shape) == 3:
        print("Converting RGB to grayscale...")
        img = np.mean(img, axis=-1)

    # Check if the image is grayscale (2D) after conversion
    if len(img.shape) == 2:
        if ch_swap is not None or ch_extract is not None:
            warnings.warn("Grayscale images have no channels to swap or extract.", UserWarning)
        return img  # Return grayscale image
    
    # Proceed with channel manipulations when image is RGB (3D) after conversion
    if len(img.shape) == 3:
    
        # Validate channel swapping when requested
        if ch_swap is not None:
            
            # Validate ch_swap: must be a list or tuple of 3 integers, with no duplicates, and must include all 0, 1, 2
            if not isinstance(ch_swap, (list, tuple)):
                raise TypeError("ch_swap must be a list or tuple.")
            
            if not all(isinstance(ch, int) for ch in ch_swap):
                raise TypeError("All elements in ch_swap must be integers.")
            
            if len(ch_swap) != 3 or not all(ch in [0, 1, 2] for ch in ch_swap):
                raise ValueError("ch_swap must be three elements of valid RGB channel indices 0, 1, or 2.")
            
            if len(set(ch_swap)) != 3:
                raise ValueError("ch_swap must include all channels 0, 1, and 2 exactly once.")
            
            if ch_swap == (0, 1, 2) or ch_swap == [0, 1, 2]:
                warnings.warn("Input is in default channel order. No swap needed.", UserWarning)

            # Perform channel swapping
            img = img[..., ch_swap]
    
        # Validate channel extraction when requested (can be potentially after ch_swap)
        if ch_extract is not None:

            # Validate ch_extract: should be a list or tuple of 0, 1, or 2, with no duplicates
            if not isinstance(ch_extract, (list, tuple)):
                raise TypeError("ch_extract must be a list or tuple.")
            
            if not all(isinstance(ch, int) for ch in ch_extract):
                raise TypeError("All elements in ch_extract must be integers.")
            
            if not all(ch in [0, 1, 2] for ch in ch_extract):
                raise ValueError("Invalid channel indices. Only 0, 1, or 2 are valid.")
            
            if len(set(ch_extract)) != len(ch_extract):
                raise ValueError("ch_extract contains duplicate channel indices.")
            
            # Perform channel extraction 
            img = img[..., ch_extract]


    # If no operation is requested, return the original image
    return img
