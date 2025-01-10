def pooling_image(img, window_size, func=np.mean):
    """
    Perform pooling on an image using a specified window size and pooling function.
    
    This function reduces the size of an input image, by dividing the image into non-overlapping 
    windows of a specified size by implementing a pooling function (e.g., mean, max, or min) to 
    each window. 
    
    Parameters:
    img (ndarray): The input image as a 2D numpy array (grayscale) or a 3D numpy 
                   array (RGB).
    
    window_size (int): The size of the pooling window (e.g., 10 for 10x10 windows). 
                                        
    func (callable, optional): The pooling function that will be used on each window. 
                                Common functions used are np.mean, np.median, np.max, np.min.
                                Default is np.mean.
        

    Returns:
    ndarray: The resized image that was reshaped based off of respective pooling function and
             window size.
    
    Examples:
    >>> img = np.random.rand(100, 100)
    >>> pooled_img = pooling_image(img, window_size=10, func=np.mean)
    >>> pooled_img = pooling_image(img_rgb, window_size=20, func=np.max)
    """
