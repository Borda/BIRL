"""


Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import numpy as np


def transform_points(points, matrix):
    """ transform points according to given transformation matrix

    :param ndarray points: set of points of shape (N, 2)
    :param ndarray matrix: transformation matrix of shape (3, 3)
    :return ndarray: warped points  of shape (N, 2)
    """
    # Pad the data with ones, so that our transformation can do translations
    pts_pad = np.hstack([points, np.ones((points.shape[0], 1))])
    points_warp = np.dot(pts_pad, matrix.T)
    return points_warp[:, :-1]


def estimate_affine_transform(points_0, points_1):
    """ estimate Affine transformations and warp particular points sets
    to the other coordinate frame

    :param ndarray points_0: set of points of shape (N, 2)
    :param ndarray points_1: set of points of shape (N, 2)
    :return (ndarray, ndarray, ndarray, ndarray): transform. matrix & inverse
        and warped point sets

    >>> pts0 = np.array([[4., 116.], [4., 4.], [26., 4.], [26., 116.]], dtype=int)
    >>> pts1 = np.array([[61., 56.], [61., -56.], [39., -56.], [39., 56.]])
    >>> mx, mx_inv, pts0_w, pts1_w = estimate_affine_transform(pts0, pts1)
    >>> np.round(mx, 2)
    array([[ -1.,   0.,  65.],
           [  0.,   1., -60.],
           [  0.,   0.,   1.]])
    >>> pts0_w
    array([[ 61.,  56.],
           [ 61., -56.],
           [ 39., -56.],
           [ 39.,  56.]])
    >>> pts1_w
    array([[   4.,  116.],
           [   4.,    4.],
           [  26.,    4.],
           [  26.,  116.]])
    """
    # SEE: https://stackoverflow.com/questions/20546182
    nb = min(len(points_0), len(points_1))
    # Pad the data with ones, so that our transformation can do translations
    x = np.hstack([points_0[:nb], np.ones((nb, 1))])
    y = np.hstack([points_1[:nb], np.ones((nb, 1))])

    # Solve the least squares problem X * A = Y to find our transform. matrix A
    matrix = np.linalg.lstsq(x, y, rcond=-1)[0].T
    matrix[-1, :] = [0, 0, 1]
    # invert the transformation matrix
    matrix_inv = np.linalg.pinv(matrix.T).T
    matrix_inv[-1, :] = [0, 0, 1]

    points_0_warp = transform_points(points_0, matrix)
    points_1_warp = transform_points(points_1, matrix_inv)

    return matrix, matrix_inv, points_0_warp, points_1_warp
