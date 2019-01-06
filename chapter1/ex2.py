#!/usr/bin/env python

from PIL import Image
from numpy import array, zeros
from scipy.ndimage import filters
import matplotlib.pyplot as plt

image_file = '../data/sf_view1.jpg'

def unsharp_masking(image: array, amount: float) -> array:
    """Sharpen image by subtracting the blurred version from original."""
    if image.ndim < 3:
        # grayscale image
        blur_image = filters.gaussian_filter(image, amount)
    else:
        blur_image = zeros(image.shape)
        for i in range(3):
            blur_image[:,:,i] = filters.gaussian_filter(image[:,:,i], amount)
        blur_image = array(blur_image, 'uint8')

    return (image + (image - blur_image))

if __name__ == '__main__':
    import scipy
    image = array(Image.open(image_file))
    sharp_image = unsharp_masking(image, 1.0)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(sharp_image)
    plt.show()
