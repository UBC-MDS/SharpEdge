# This function is created solely for the purpose of unit testing.
# It is not intended to be part of the actual software package.
# Please refrain from using this function in production or in any user-facing features.
#
# It is placed in the tests folder to separate from core software functionality.
# It is included to help validate specific components of the code during testing.
# This function may be removed or modified as needed for the testing process.

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