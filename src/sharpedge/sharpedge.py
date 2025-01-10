def load_image(path):
    """
    Load an image from a specific path
    and convert it to a numpy array.

    Parameters
    ----------
    path : str
        The file path to the image.

    Returns
    -------
    numpy.ndarray
        The image data as a numpy array
        with shape (height, width, channels).

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If the file is not a valid image format.

    Examples
    --------
    >>> img = load_image('image.jpg')
    >>> print(img.shape)
    (1280, 800, 3)
    """
    pass


def display_image(img):
    """
    Display the numpy array as an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image data as a numpy array
        with shape (height, width, channels).

    Raises
    ------
    ValueError
        If the file is not a valid numpy array.

    Examples
    --------
    >>> img = load_image('image.jpg')
    >>> display_image(img)
    """
    pass
