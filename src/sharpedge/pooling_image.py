def pooling_image(image, window_size, func=np.mean, color_map='gray'):
    """
    Perform pooling on an image using a specified window size and pooling function.
    
    This function reduces the size of an input image, by dividing it into non-overlapping 
    windows of a specified size by implementing a pooling function (e.g., mean, max, or min) 
    to each window. The pooled image is then returned as displayed with a specific colormap.
    
    Parameters:
    image (ndarray): The input image as a 2D numpy array (grayscale) or a 3D numpy 
                   array (RGB).
    
    window_size (int): The size of the pooling window (e.g., 10 for 10x10 windows). 
                                        
    func (callable, optional): The pooling function that will be used on each window. 
                                Common functions used are np.mean, np.median, np.max, np.min.
                                Default is np.mean.
        
    color_map (str, optional): The colormap to use for displaying the pooled image using matplotlib. Examples include:
                               'gray', 'inferno', 'virdis', 'plasma', etc.
                                Default is `'gray'`.
    
    Returns:
    matplotlib.image: The pooled image displayed with the specified colormap.
    
    
    Examples:
    >>> img = np.random.rand(100, 100)
    >>> pooled_img = pooling_image(image=img, window_size=10, func=np.mean, color_map='gray')
    >>> pooled_img = pooling_image(image=img, window_size=20, func=np.max, color_map='viridis')
    """