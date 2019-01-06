#!/usr/bin/env python

from numpy import roll, linalg, maximum, sqrt

def denoise(image, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presented in eq (11) A. Chambolle (2005).

    Input: noisy input image (grayscale), initial guess for U, weight of
    the TV-regularizing term, step length, tolerance for stop criterion.

    Output: denoised and detextured image, texture redisual.
    """
    m, n = image.shape # size of noisy image

    # initialize
    U = U_init
    Px = image # x-component to the dual field
    Py = image # y-component to the dual field
    error = 1.0

    while (error > tolerance):
        U_old = U

        # gradient of primal variable
        grad_Ux = roll(U, -1, axis=1) - U # x-component of U's gradient
        grad_Uy = roll(U, -1, axis=0) - U # y-component of U's gradient

        # update the dual variable
        Px_new = Px + (tau / tv_weight) * grad_Ux
        Py_new = Py + (tau / tv_weight) * grad_Uy
        norm_new = maximum(1, sqrt(Px_new ** 2 + Py_new ** 2))

        Px = Px_new / norm_new # update of x-component (dual)
        Py = Py_new / norm_new # update of y-component (dual)

        # update the primal variable
        Rx_Px = roll(Px, 1, axis=1) # right x-translation of x-component
        Ry_Py = roll(Py, 1, axis=1) # right y-translation of y-component

        DivP = (Px - Rx_Px) + (Py - Ry_Py) # divergence of the dual field

        U = image + (tv_weight * DivP) # update of the primal variable

        # update of error
        error = linalg.norm(U - U_old) / sqrt(n * m)

    return U, (image - U) # denoised image and texture redisual

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

    U, T = denoise(image, image)
    G = filters.gaussian_filter(image, 10)

    # show results
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(Image.fromarray(image))
    axes[0].axis('off')

    axes[1].imshow(Image.fromarray(G))
    axes[1].axis('off')

    axes[2].imshow(Image.fromarray(U))
    axes[2].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()
