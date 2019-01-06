#!/usr/bin/env python

from PIL import Image
from numpy import array
from scipy.ndimage import filters
import matplotlib.pyplot as plt

image_file = '../data/sunset_tree.jpg'

def unsharp_masking(image: array, amount: float) -> array:
    """Sharpen image by subtracting the blurred version from original."""
    if image.ndim < 3:
        # grayscale image
        blurred_image = filters.gaussian_filter(image, amount)
    else:
        blurred_image = filters.gaussian_filter(image, amount)

    return (image + (image - blurred_image))

if __name__ == '__main__':
    image = array(Image.open(image_file).convert('L'))
    sharp_image = unsharp_masking(image, 0.65)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(sharp_image, cmap="gray")
    plt.show()
