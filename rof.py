#!/usr/bin/env python

if __name__ == '__main__':
    from numpy import zeros, random
    from scipy.ndimage import filters
    from PIL import Image
    import matplotlib.pyplot as plt

    # create synthetic image with noise
    image = zeros((500, 500))
    image[100:400, 100:400] = 128
    image[200:300, 200:300] = 255
    image = image + 30*random.standard_normal((500, 500))

    # U, T = rof.denoise(image, image)
    G = filters.gaussian_filter(image, 10)

    # show results
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(Image.fromarray(image))
    axes[0].axis('off')

    axes[1].imshow(Image.fromarray(G))
    axes[1].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()
