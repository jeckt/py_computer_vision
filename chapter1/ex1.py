#!/usr/bin/env python

# Gaussian blur exercise

from PIL import Image
from numpy import array
from scipy.ndimage import filters
import matplotlib.pyplot as plt

sigmas = [5, 10, 15]
image_file = '../data/empire.jpg'

if __name__ == '__main__':
    image = array(Image.open(image_file).convert('L'))
    images = list(filters.gaussian_filter(image, s) for s in sigmas)
    images.insert(0, image)

    fig, axes = plt.subplots(2, len(images))
    for i, im in enumerate(images):
        axes[0, i].imshow(im, cmap="gray")
        axes[0, i].axis('off')

        axes[1, i].contour(im)
        axes[1, i].axis('off')

    plt.show()
