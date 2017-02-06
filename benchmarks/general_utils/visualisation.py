"""
Function for drawing and visualisations

Copyright (C) 2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pylab as plt
from PIL import Image, ImageDraw

import benchmarks.general_utils.io_utils as tl_io

MAX_FIGURE_SIZE = 18


def draw_image_points(image, points, color='green', marker_size=5, shape='o'):
    """ draw marker in the image and add to each landmark its index

    :param image: np.ndarray
    :param points: np.array<nb_points, dim>
    :param str color: color of the marker
    :param int marker_size: radius of the circular marker
    :param str shape: marker shape: 'o' for circle, '.' for dot
    :return: np.ndarray

    >>> image = np.random.random((50, 50, 3))
    >>> points = np.array([[20, 30], [40, 10], [15, 25]])
    >>> image = draw_image_points(image, points)
    """
    image = tl_io.convert_ndarray_2_image(image)
    draw = ImageDraw.Draw(image)
    for i, (x, y) in enumerate(points):
        pos_marker = (x - marker_size, y - marker_size,
               x + marker_size, y + marker_size)
        pos_text = tuple(points[i] + marker_size)
        if shape == 'o':
            draw.ellipse(pos_marker, outline=color)
        elif shape == '.':
            draw.ellipse(pos_marker, fill=color)
        else:
            draw.ellipse(pos_marker, fill=color, outline=color)
        draw.text(pos_text, str(i + 1), fill=(0, 0, 0))
    image = np.array(image) / 255.
    return image


def draw_landmarks_origin_target_estim(ax, points_origin, points_target,
                                       points_estim=None, marker='o'):
    """ visualisation of transforming points, presenting 3 set of points:
    original points, targeting points, and the estimate of target points

    scenario 1:
    original - moving landmarks
    target - reference landmarks
    estimate - transformed landmarks

    scenario 2:
    original - reference landmarks
    target - moving landmarks
    estimate - transformed landmarks

    :param ax: matplotlib figure
    :param points_origin: np.array<nb_points, dim>
    :param points_target: np.array<nb_points, dim>
    :param points_estim: np.array<nb_points, dim>
    :param str marker: set the marker shape

    >>> points = np.array([[20, 30], [40, 10], [15, 25]])
    >>> draw_landmarks_origin_target_estim(plt.figure().gca(),
    ...                                points, points + 1, points - 1)
    """
    assert points_target.shape == points_origin.shape
    assert points_origin.shape == points_estim.shape
    ax.plot(points_origin[:, 0], points_origin[:, 1], marker, color='b',
            label='Original positions')
    # draw a dotted line between origin and where it should be
    for start, stop in zip(points_target, points_origin):
        x, y = zip(start, stop)
        ax.plot(x, y, '-.', color='b', linewidth=2)
    ax.plot(points_target[:, 0], points_target[:, 1], marker, color='m',
            label='Target positions')
    if points_estim is not None:
        # draw line that  should be minimal between target and estimate
        for start, stop in zip(points_target, points_estim):
            x, y = zip(start, stop)
            ax.plot(x, y, '-', color='r', linewidth=2)
        ax.plot(points_estim[:, 0], points_estim[:, 1], marker, color='g',
                label='Estimated positions')


def overlap_two_images(image1, image2, transparent=0.5):
    """ merge two images togeher with transparency level

    :param image1: np.array<height, with, dim>
    :param image2: np.array<height, with, dim>
    :param float transparent: level ot transparency in range (0, 1)
        with 1 to see only first image nad 0 to see the second one
    :return: np.array<height, with, dim>

    >>> img1 = np.ones((5, 6, 1)) * 0.2
    >>> img2 = np.ones((6, 5, 1)) * 0.8
    >>> overlap_two_images(img1, img2, transparent=0.5)[:, :, 0]
    array([[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.1],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.1],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.1],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.1],
           [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.1],
           [ 0.4,  0.4,  0.4,  0.4,  0.4,  0. ]])
    """
    assert image1.ndim == 3
    assert image1.ndim == image2.ndim
    size1, size2 = image1.shape, image2.shape
    max_size = np.max(np.array([size1, size2]), axis=0)
    image = np.zeros(max_size)
    image[0:size1[0], 0:size1[1], 0:size1[2]] += image1 * transparent
    image[0:size2[0], 0:size2[1], 0:size2[2]] += image2 * (1. - transparent)
    return image


def draw_images_warped_landmarks(image_target, image_source,
                                 points_init, points_target, points_estim,
                                 fig_max_size=MAX_FIGURE_SIZE):
    """ composed form several functions - images overlap + landmarks + legend

    :param image_target: np.array<height, with, dim>
    :param image_source: np.array<height, with, dim>
    :param points_target: np.array<nb_points, dim>
    :param points_init: np.array<nb_points, dim>
    :param points_estim: np.array<nb_points, dim>
    :param int fig_max_size: maximal figure size for major image dimension
    :return: object

    >>> image = np.random.random((50, 50, 3))
    >>> points = np.array([[20, 30], [40, 10], [15, 25]])
    >>> fig = draw_images_warped_landmarks(image, 1 - image,
    ...                                    points, points + 1, points - 1)
    """
    image = overlap_two_images(image_target, image_source, transparent=0.3)
    size = np.array(image.shape[:2])
    fig_size = size[::-1] / float(size.max()) * fig_max_size
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image)
    draw_landmarks_origin_target_estim(ax, points_init, points_target, points_estim)
    ax.legend(loc='lower right', title='Legend')
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    return fig


def export_figure(path_fig, fig):
    """ export the figure and close it afterwords

    :param str path_fig: path to the new figure image
    :param fig: object

    >>> export_figure('./sample_figure.jpg', plt.figure())
    >>> os.remove('./sample_figure.jpg')
    """
    assert os.path.exists(os.path.dirname(path_fig))
    fig.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    fig.savefig(path_fig)
    plt.close(fig)
