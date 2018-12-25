#!/usr/bin/env python

import os

def get_image_list(path):
    """Returns a list of filenames for all jpg images in a directory."""

    return list(os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith('.jpg'))

def image_resize(image, size):
    """Resize an image array using PIL."""
    image = Image.fromarray(uint8(im))

    return array(image.resize(size))

if __name__ == '__main__':
    images_files = get_image_list('data')
    for f in images_files:
        print(f)
