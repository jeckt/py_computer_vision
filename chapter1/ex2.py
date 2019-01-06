#!/usr/bin/env python

from PIL import Image
from numpy import array
from scipy.ndimage import filters
import matplotlib.pyplot as plt

image_file = '../data/sunset_tree.jpg'

if __name__ == '__main__':
    image = array(Image.open(image_file).convert('L'))
    blurred_image = filters.gaussian_filter(image, 0.65)
    sharp_image = image + (image - blurred_image)

    Image.fromarray(image).show()
    Image.fromarray(sharp_image).show()

    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(sharp_image, cmap="gray")
    plt.show()
    """
