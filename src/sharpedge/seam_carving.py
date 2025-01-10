def energy(img):
    """
    Computes the energy map of an image.

    This function calculates the energy of each pixel. The
    energy map highlights areas with high contrast, which
    are less likely to be removed during seam carving.

    Parameters
    ----------
    image : numpy.ndarray
        A color image represented as a 3D numpy array of shape (height, width, 3).

    Returns
    -------
    numpy.ndarray
        A new image where the pixels values represent the energy
        of the corresponding pixel in the original image

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
    pass


def remove_horizontal_seam(img, seam):
    pass


def seam_carve(img, target_height, target_width):
    pass
