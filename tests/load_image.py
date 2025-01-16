# This function is created solely for the purpose of unit testing.
# It is not intended to be part of the actual software package.
# Please refrain from using this function in production or in any user-facing features.
#
# It is placed in the tests folder to separate from core software functionality.
# It is included to help validate specific components of the code during testing.
# This function may be removed or modified as needed for the testing process.

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