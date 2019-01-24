"""
Function for drawing and visualisations

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os

# import matplotlib
# if os.environ.get('DISPLAY', '') == '':
#     print('No display found. Using non-interactive Agg backend')
#     matplotlib.use('Agg')

import numpy as np
import matplotlib.pylab as plt
from PIL import ImageDraw

import benchmark.utilities.data_io as tl_io

MAX_FIGURE_SIZE = 18


def draw_image_points(image, points, color='green', marker_size=5, shape='o'):
    """ draw marker in the image and add to each landmark its index

    :param ndarray image: input image
    :param ndarray points: np.array<nb_points, dim>
    :param str color: color of the marker
    :param int marker_size: radius of the circular marker
    :param str shape: marker shape: 'o' for circle, '.' for dot
    :return: np.ndarray

    >>> image = np.zeros((10, 10, 3))
    >>> points = np.array([[7, 9], [2, 2], [5, 5]])
    >>> img = draw_image_points(image, points, marker_size=1)
    >>> img.shape == (10, 10, 3)  # Windows x64 returns (10L, 10L, 3L)
    True
    >>> np.round(img[:, :, 1], 2)
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0.5,  0.5,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0.5,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0.5,  0.5,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.5,  0.5,  0.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.5,  0.5,  0.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0. ]])
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


def draw_landmarks_origin_target_warped(ax, points_origin, points_target,
                                        points_warped=None, marker='o'):
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
    :param ndarray points_origin: np.array<nb_points, dim>
    :param ndarray points_target: np.array<nb_points, dim>
    :param ndarray points_warped: np.array<nb_points, dim>
    :param str marker: set the marker shape

    >>> points = np.array([[20, 30], [40, 10], [15, 25]])
    >>> draw_landmarks_origin_target_warped(plt.figure().gca(),
    ...                                    points, points + 1, points - 1)
    """
    assert points_target.shape == points_origin.shape, \
        'image dimension has to match %s != %s' \
        % (repr(points_target.shape), repr(points_origin.shape))
    assert points_origin.shape == points_warped.shape, \
        'image dimension has to match %s != %s' \
        % (repr(points_origin.shape), repr(points_warped.shape))
    ax.plot(points_origin[:, 0], points_origin[:, 1], marker, color='b',
            label='Original positions')
    # draw a dotted line between origin and where it should be
    for start, stop in zip(points_target, points_origin):
        x, y = zip(start, stop)
        ax.plot(x, y, '-.', color='b', linewidth=2)
    ax.plot([0, 0], [0, 0], '-.', color='b', linewidth=2, label='registration shift')
    ax.plot(points_target[:, 0], points_target[:, 1], marker, color='m',
            label='Target positions')
    if points_warped is not None:
        # draw line that  should be minimal between target and estimate
        for start, stop in zip(points_target, points_warped):
            x, y = zip(start, stop)
            ax.plot(x, y, '-', color='r', linewidth=2)
        ax.plot([0, 0], [0, 0], '-', color='r', linewidth=2., label='regist. error (TRE)')
        ax.plot(points_warped[:, 0], points_warped[:, 1], marker, color='g',
                label='Estimated positions')


def overlap_two_images(image1, image2, transparent=0.5):
    """ merge two images together with transparency level

    :param ndarray image1: np.array<height, with, dim>
    :param ndarray image2: np.array<height, with, dim>
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
    assert image1.ndim == 3, 'required RGB images, got %s' % repr(image1.ndim)
    assert image1.ndim == image2.ndim, 'image dimension has to match, %s != %s' \
                                       % (repr(image1.ndim), repr(image2.ndim))
    size1, size2 = image1.shape, image2.shape
    max_size = np.max(np.array([size1, size2]), axis=0)
    image = np.zeros(max_size)
    image[0:size1[0], 0:size1[1], 0:size1[2]] += image1 * transparent
    image[0:size2[0], 0:size2[1], 0:size2[2]] += image2 * (1. - transparent)
    return image


def draw_images_warped_landmarks(image_target, image_source,
                                 points_init, points_target, points_warped,
                                 figsize_max=MAX_FIGURE_SIZE):
    """ composed form several functions - images overlap + landmarks + legend

    :param ndarray image_target: np.array<height, with, dim>
    :param ndarray image_source: np.array<height, with, dim>
    :param ndarray points_target: np.array<nb_points, dim>
    :param ndarray points_init: np.array<nb_points, dim>
    :param ndarray points_warped: np.array<nb_points, dim>
    :param int figsize_max: maximal figure size for major image dimension
    :return: object

    >>> image = np.random.random((50, 50, 3))
    >>> points = np.array([[20, 30], [40, 10], [15, 25]])
    >>> fig = draw_images_warped_landmarks(image, 1 - image,
    ...                                    points, points + 1, points - 1)  # doctest: +ELLIPSIS
    >>> isinstance(fig, plt.Figure)
    True
    """
    image = overlap_two_images(image_target, image_source, transparent=0.3)
    fig, ax = create_figure(image.shape, figsize_max)
    ax.imshow(image)
    draw_landmarks_origin_target_warped(ax, points_init, points_target, points_warped)
    ax.legend(loc='lower right', title='Legend')
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    return fig


def create_figure(im_size, figsize_max=MAX_FIGURE_SIZE):
    """ create an empty figure of image size maximise maximal size

    :param (int, int) im_size:
    :param float figsize_max:
    :return:

    >>> fig, ax = create_figure((100, 150))
    >>> isinstance(fig, plt.Figure)
    True
    """
    assert len(im_size) >= 2, 'not valid image size - %s' % repr(im_size)
    size = np.array(im_size[:2])
    fig_size = size[::-1] / float(size.max()) * figsize_max
    fig, ax = plt.subplots(figsize=fig_size)
    return fig, ax


def export_figure(path_fig, fig):
    """ export the figure and close it afterwords

    :param str path_fig: path to the new figure image
    :param fig: object

    >>> path_fig = './sample_figure.jpg'
    >>> export_figure(path_fig, plt.figure())
    >>> os.remove(path_fig)
    """
    assert os.path.isdir(os.path.dirname(path_fig)), \
        'missing folder "%s"' % os.path.dirname(path_fig)
    fig.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    fig.savefig(path_fig)
    plt.close(fig)
