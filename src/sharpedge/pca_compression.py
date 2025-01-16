def pca_compression(img, preservation_rate=0.9):
    """
    Compress the input image using Principal Component Analysis (PCA).
    This function compresses the size of an image by applying the PCA method while retaining a specified 
    portion of the original data variance.
    It supports both grayscale (2D) and RGB (3D) images.
    The output image will always be in grayscale.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image array. This can be either a 2D numpy array (grayscale image) or a 3D numpy array 
        (RGB image). The dimensions of the image should be (height, width) for grayscale or 
        (height, width, 3) for RGB images.
    
    preservation_rate : float, optional
        The proportion of variance to preserve in the compressed image. Must be a value between 0 and 1.
        Higher values preserve more details from the original image, while lower values result in greater
        compression. Default is 0.9.

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
        - If the input image is not in grayscale or RGB format, or if any invalid channel indices are 
        provided for extraction or swapping.
        - If the `preservation_rate` is not between 0 and 1.
      
    Examples
    --------
    Compress a grayscale image by retaining 80% of the variance:
    >>> compressed_img = pca_compression(grayscale_img, preservation_rate=0.8)

    Compress an RGB image with the default preservation rate (90%):
    >>> compressed_img = pca_compression(rgb_img)
    """
    pass