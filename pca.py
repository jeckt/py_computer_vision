#!/usr/bin/env python

from PIL import Image
from numpy import array, linalg

def pca(X: array) -> array:
    """Principal Component Analysis

    input: X, matrix with training data stored as flattened arrays, in rows
    return: projection matrix (with important dimensions first), variance
    and mean."""

    # get dimensions
    num_data, dim = X.shape

    # centre data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used
        M = dot(X, X.T) # covariance matrix
        e, EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T, EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U, S, V = linalg.svd(X)
        V = V[:num_data] # only makes sense ot return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X

if __name__ == '__main__':
    import os
    from zipfile import ZipFile
    from shutil import rmtree
    from pylab import figure, gray, subplot, imshow, show

    # unzip image files in fontimages.zip
    with ZipFile('data/fontimages.zip', 'r') as zip_file:
        zip_file.extractall('data')

    image_path = 'data/a_thumbs'
    image_files = os.listdir(image_path)

    image = array(Image.open(os.path.join(image_path, image_files[0])))
    m, n = image.shape[0:2]
    n_images = len(image_files)

    # create matrix to store all flattened images
    image_matrix = array(
        [array(Image.open(os.path.join(image_path, im))).flatten()
         for im in image_files], 'f'
    )

    # perform PCA
    V, S, image_mean = pca(image_matrix)

    # show some images (mean and first 7 modes)
    figure()
    gray()
    subplot(2, 4, 1)
    imshow(image_mean.reshape(m,n ))
    for i in range(7):
        subplot(2, 4, i + 2)
        imshow(V[i].reshape(m, n))

    show()

    # remove unzipped images
    rmtree(image_path)
