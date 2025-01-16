def energy(img):
    """
    Computes the energy map of an image.

    This function calculates the energy of each pixel. The
    energy map highlights areas with high contrast, which
    are less likely to be removed during seam carving.

    Parameters
    ----------
    img : numpy.ndarray
        A color image represented as a 3D numpy array of shape (height, width, 3).

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (height, width) containing the energy values
        for each pixel in the original image.

    Raises
    ------
    ValueError
        If the input image is not a 3D numpy array with 3 channels.

    Examples
    --------
    >>> img = np.random.rand(8, 5, 3)
    >>> e = energy(img)
    >>> print(e.shape)
    (8, 5)
    """
    pass


def find_vertical_seam(energy):
    """
    Find the vertical seam of lowest total energy in the image.

    Parameters
    ----------
    energy : numpy.ndarray
        A 2D array representing the energy of each pixel.

    Returns
    -------
    list
        A list indicating the seam of column indices.

    Raises
    ------
    ValueError
        If the energy map is not a 2D numpy array.

    Examples
    --------
    >>> e = np.array([[0.6625, 0.3939], [1.0069, 0.7383]])
    >>> seam = find_vertical_seam(e)
    >>> print(seam)
    [1, 1]
    """
    pass


def find_horizontal_seam(energy):
    """
    Find the horizontal seam of lowest total energy
    in the image by transposing the energy map.

    Parameters
    ----------
    energy : numpy.ndarray
        A 2D array representing the energy of each pixel.

    Returns
    -------
    list
        A list indicating the seam of row indices.

    Raises
    ------
    ValueError
        If the energy map is not a 2D numpy array.

    Examples
    --------
    >>> e = np.array([[0.6625, 0.3939], [1.0069, 0.7383]])
    >>> seam = find_horizontal_seam(e)
    >>> print(seam)
    [0, 0]
    """
    pass


def remove_vertical_seam(img, seam):
    """
    Remove a vertical seam from an image.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D array representing the original image (height, width, 3).
    seam : numpy.ndarray
        A 1D array (or list) of column indices indicating
        which pixel to remove in each row.

    Returns
    -------
    numpy.ndarray
        A new image with one less column, of shape (height, width - 1, 3).

    Raises
    ------
    ValueError
        - If the input img is not a 3D numpy array with 3 channels.
        - If the input seam is not a 1D array or a list.
        - If the length of the seam does not match the height of the image.

    Examples
    --------
    >>> img = np.random.rand(8, 5, 3)
    >>> seam = [2, 1, 3, 2, 0, 1, 4, 3]
    >>> new_img = remove_vertical_seam(img, seam)
    >>> print(new_img.shape)
    (8, 4, 3)
    """
    pass


def remove_horizontal_seam(img, seam):
    """
    Remove a horizontal seam from an image.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D array representing the original image (height, width, 3).
    seam : numpy.ndarray
        A 1D array (or list) of row indices indicating
        which pixel to remove in each column.

    Returns
    -------
    numpy.ndarray
        A new image with one less row, of shape (height - 1, width, 3).

    Raises
    ------
    ValueError
        - If the input img is not a 3D numpy array with 3 channels.
        - If the input seam is not a 1D array or a list.
        - If the length of the seam does not match the width of the image.

    Examples
    --------
    >>> img = np.random.rand(5, 8, 3)
    >>> seam = [2, 1, 3, 2, 0, 1, 4, 3]
    >>> new_img = remove_horizontal_seam(img, seam)
    >>> print(new_img.shape)
    (4, 8, 3)
    """
    pass


def seam_carve(img, target_height, target_width):
    """
    Seam carve an image to resize it to the target dimensions.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D array representing the original image (height, width, 3).
    target_height : int
        The desired height of the resized image.
    target_width : int
        The desired width of the resized image.

    Returns
    -------
    numpy.ndarray
        The resized image with dimensions (target_height, target_width, 3).

    Raises
    ------
    ValueError
        - If the input img is not a 3D numpy array with 3 channels.
        - If target_height or target_width is not an integer.
        - If target_height is greater than the original height or less than 1.
        - If target_width is greater than the original width or less than 1.

    Examples
    --------
    >>> img = np.random.rand(5, 5, 3)
    >>> resized_img = seam_carve(img, 3, 3)
    >>> print(resized_img.shape)
    (3, 3, 3)
    """
    pass
