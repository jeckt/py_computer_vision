#!/usr/bin/env python

# TODO(steve): Should add some rudimentary unit tests

import os
from PIL import Image
from numpy import uint8, array, histogram, interp

def get_image_list(path: str) -> list:
    """Returns a list of filenames for all jpg images in a directory."""

    return list(os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith('.jpg'))

def image_resize(image: Image, size: (int, int)) -> array:
    """Resize an image array using PIL."""
    image = Image.fromarray(uint8(image))

    return array(image.resize(size))

def histogram_equalisation(image: Image, n_bins: int =256) -> (Image, array):
    """Histogram equalisation of a grayscale image."""

    # get image histogram
    image_hist, bins = histogram(image.flatten(), n_bins, density=True)
    cdf = image_hist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalise

    # use linear interpolation of cdf to find new pixel values
    new_image = interp(image.flatten(), bins[:-1], cdf)

    return new_image.reshape(image.shape), cdf

def compute_average(image_list: list(str)) -> array:
    """Compute the average of a list of images."""

    # open first image and make into array of type float
    average_image = array(Image.open(image_list[0]), 'f')

    for image_file in image_list[1:]:
        try:
            average_image += array(Image.open(image_file))
        except:
            print("{0}...skipped", image_file)
    average_image /= len(image_list)

    # return average as uint8
    return array(average_image, 'uint8')

if __name__ == '__main__':
    image = array(Image.open('data/AquaTermi_lowcontrast.jpg').convert('L'))
    new_image, cdf = histogram_equalisation(image)
