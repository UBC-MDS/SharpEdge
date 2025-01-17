# SharpEdge

Collection of image processing tools and techniques, including padding, flipping, colorscale conversion, seam carving, and image shrinking. Designed for efficient manipulation and transformation of images.

## Summary

This package provides a comprehensive set of image processing utilities tailored for seamless integration of image manipulation projects. It includes essential tools for image transformations, seam carving, and processing. This package enables users to efficiently prepare, manipulate and modify image based on the needs of the user. The `sharpedge` package is valuable for users seeking tools for resizing and compressing images while maintaining visual content integrity.

## Functions
- **`reposition_image`**  
  This function allows you to manipulate the position and orientation of an image by flipping, rotating, or shifting it. You can customize the actions based on your needs, such as horizontal flips, rotation to the left or right, or shifting the image along the X and Y axes.

- **`frame_image`**  
  Enhance the aesthetic of your image by adding a decorative frame around it. The frame can be customized with adjustable border sizes, placement inside or outside the image, and a specified color for the border.

- **`modulate_image`**  
  Modify the color channels of an image to achieve effects like grayscale conversion or specific channel isolation. This function is useful for color manipulation tasks, including transforming RGB images to simpler formats for analysis or artistic purposes.

- **`pooling_image`**  
  Apply pooling techniques to an image using a defined window size and a specified function, such as mean or max pooling. Pooling is commonly used to reduce image dimensions while preserving key features, making it especially relevant for preprocessing in computer vision tasks.

- **`pca_compression`**  
  Compress an image using Principal Component Analysis (PCA) by retaining only the most significant features while discarding less important data. This method is ideal for reducing file size while preserving a specified proportion of the original variance in the image.

- **`seam_carve`**  
  Resize an image intelligently using seam carving to preserve important content while adjusting dimensions. This technique minimizes distortion by removing or inserting paths of least importance, making it effective for content-aware resizing.

## Where This Fits in the Python Ecosystem

This package fits into the broader Python image ecosystem, along  with packages like [OpenCV](https://opencv.org/) and [Pillow](https://pillow.readthedocs.io/). While OpenCV and Pillow provide a wide range of general-purpose image processing tools, this package instead specializes in content-aware resizing and transformations, focusing on practical utilities for advanced image manipulations. Functions such as seam carving for resizing, PCA-based image compression, and pooling for dimensionality reduction offer unique capabilities that are not directly available in general-purpose libraries. 

## Installation

```bash
$ pip install sharpedge
```

## Usage

To harness the image processing magic of SharpEdge, follow these steps:

1. Import the required functions from the package:

    ```python
    from sharpedge.reposition_image import reposition_image
    from sharpedge.frame_image import frame_image
    from sharpedge.modulate_image import modulate_image
    from sharpedge.pooling_image import pooling_image
    from sharpedge.pca_compression import pca_compression
    from sharpedge.seam_carving import seam_carve
    ```

2. (Optional) Load your image as a NumPy array.

      ```python
      # Ensure that you have matplotlib and numpy installed in your environment
      import matplotlib.pyplot as plt
      import numpy as np

      # Load the image from the given path
      img = plt.imread(PATH_TO_IMAGE)

      # Only execute this conversion for functions other than seam_carve
      # Convert the loaded image to the range [0, 255] and cast it to np.uint8 
      # This is necessary for processing images with most functions, but 'seam_carve' handles float inputs directly
      img = np.round(np.clip(img * 255, 0, 255)).astype(np.uint8)
      ```

3. Process your images using the available functions:
   - Flip, rotate, and shift an image based on the specified requested action：

        ```python
        # Flip horizontally, rotate left, and shift the image
        repositioned_img = reposition_image(img, flip='horizontal', rotate='left', shift_x=10, shift_y=20)
        ```

   - Add a decorative frame around the image with customizable color:

        ```python
        # Add a frame around your image
        framed_img = frame_image(img, h_border=30, w_border=30, inside=True, color=255)
        ```

   - Convert or manipulate image color channels:

        ```python
        # Convert an RGB image to grayscale
        grayscale_image = modulate_image(rgb_image, mode='gray')
        ```

   - Perform pooling on an image using a specified window size and pooling function:

        ```python
        # Perform pooling on an image with mean pooling function
        pooled_img = pooling_image(img, window_size=10, func=np.mean)
        ```

   - Compress the input image using Principal Component Analysis (PCA):

        ```python
        # Compress a grayscale image by retaining 80% of the variance
        compressed_img = pca_compression(grayscale_img, preservation_rate=0.8)
        ```

   - Resize the image using seam carving to the target dimensions:

        ```python
        # Seam carve an image to resize it to the target dimensions
        resized_img = seam_carve(img, target_height=300, target_width=400)
        ```

## Contributors

Archer Liu, Hankun Xiao, Inder Khera, Jenny Zhang (ordered alphabetically)

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`sharpedge` was created by Jenny Zhang, Archer Liu, Inder Khera, Hankun Xiao. It is licensed under the terms of the MIT license.

## Credits

`sharpedge` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).