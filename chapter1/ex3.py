#!/usr/bin/env python
import sys
sys.path.append('..')

from PIL import Image
from numpy import array
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from image_tools import histogram_equalisation

image_file = '../data/AquaTermi_lowcontrast.jpg'

if __name__ == '__main__':
    image = array(Image.open(image_file).convert('L'))
    blur = filters.gaussian_filter(image, 3)
    quot_image = image / blur
    new_image = (image - quot_image)
    hist_image, _ = histogram_equalisation(image)

    if False:
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(image, cmap=plt.cm.gray)
        axes[1].imshow(new_image, cmap=plt.cm.gray)
        axes[2].imshow(hist_image, cmap=plt.cm.gray)
        plt.show()
