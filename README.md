# sharpedge

Collection of image processing tools and techniques, including padding, flipping, colorscale conversion, seam carving, and image shrinking. Designed for efficient manipulation and transformation of images.

## Installation

```bash
$ pip install sharpedge
```

## Usage

To harness the image processing magic of SharpEdge, follow these steps:

1. Import the required functions from the package:

    ```python
    from sharpedge.frame_image import frame_image
    from sharpedge.modulate_image import modulate_image
    from sharpedge.pca_compression import pca_compression
    from sharpedge.pooling_image import pooling_image
    from sharpedge.seam_carving import seam_carve
    ```

2. Load your image as a NumPy array.
3. Process your images using the available functions:
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

   - Compress the input image using Principal Component Analysis (PCA):

        ```python
        # Compress a grayscale image by retaining 80% of the variance
        compressed_img = pca_compression(grayscale_img, preservation_rate=0.8)
        ```

   - Perform pooling on an image using a specified window size and pooling function:

        ```python
        # Perform pooling on an image with mean pooling function
        pooled_img = pooling_image(img, window_size=10, func=np.mean)
        ```

   - Resize the image using seam carving to the target dimensions:

        ```python
        # Seam carve an image to resize it to the target dimensions
        resized_img = seam_carve(img, target_height=300, target_width=400)
        ```

## Contributors

Archer Liu, Inder Khera, Hankun Xiao, Jenny Zhang (ordered alphabetically)

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`sharpedge` was created by Jenny Zhang, Archer Liu, Inder Khera, Hankun Xiao. It is licensed under the terms of the MIT license.

## Credits

`sharpedge` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
