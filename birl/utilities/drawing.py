"""
Function for drawing and visualisations

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging
import collections

import numpy as np
import matplotlib.pylab as plt
from PIL import ImageDraw
from matplotlib import colors as plt_colors, ticker as plt_ticker

from birl.utilities.data_io import convert_ndarray2image
from birl.utilities.dataset import scale_large_images_landmarks
from birl.utilities.evaluate import compute_matrix_user_ranking

#: default figure size for visualisations
MAX_FIGURE_SIZE = 18  # inches


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
    >>> img = draw_image_points(None, points, marker_size=1)
    """
    assert list(points), 'missing points'
    if image is None:
        # landmark range plus minimal offset to avoid zero image
        lnds_range = np.max(points, axis=0) - np.min(points, axis=0) + 1
        image = np.zeros(lnds_range.astype(int).tolist() + [3])
    image = convert_ndarray2image(image)
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
    ...                                     points, points + 1, points - 1)
    """
    pts_sizes = [len(pts) for pts in [points_origin, points_target, points_warped]
                 if pts is not None]
    assert pts_sizes, 'no landmarks points given'
    min_pts = min(pts_sizes)
    assert min(pts_sizes) > 0, 'no points given for sizes: %r' % pts_sizes
    points_origin = points_origin[:min_pts] if points_origin is not None else None
    points_target = points_target[:min_pts] if points_target is not None else None

    def _draw_lines(points1, points2, style, color, label):
        if points1 is None or points2 is None:
            return
        for start, stop in zip(points1, points2):
            x, y = zip(start, stop)
            ax.plot(x, y, style, color=color, linewidth=2)
        ax.plot([0, 0], [0, 0], style, color=color, linewidth=2, label=label)

    if points_origin is not None:
        ax.plot(points_origin[:, 0], points_origin[:, 1], marker, color='g',
                label='Original positions')
    # draw a dotted line between origin and target
    _draw_lines(points_target, points_origin, '-.', 'g', 'true shift')
    if points_target is not None:
        ax.plot(points_target[:, 0], points_target[:, 1], marker, color='m',
                label='Target positions')

    if points_warped is not None:
        points_warped = points_warped[:min_pts]
        # draw a dotted line between origin and warped
        _draw_lines(points_origin, points_warped, '-.', 'b', 'warped shift')
        # draw line that  should be minimal between target and estimate

        _draw_lines(points_target, points_warped, '-', 'r', 'regist. error (TRE)')
        ax.plot(points_warped[:, 0], points_warped[:, 1], marker, color='b',
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
    assert image1.ndim == 3, 'required RGB images, got %i' % image1.ndim
    assert image1.ndim == image2.ndim, 'image dimension has to match, %r != %r' \
                                       % (image1.ndim, image2.ndim)
    size1, size2 = image1.shape, image2.shape
    max_size = np.max(np.array([size1, size2]), axis=0)
    image = np.zeros(max_size)
    image[0:size1[0], 0:size1[1], 0:size1[2]] += image1 * transparent
    image[0:size2[0], 0:size2[1], 0:size2[2]] += image2 * (1. - transparent)
    # np.clip(image, a_min=0., a_max=1., out=image)
    return image


def draw_images_warped_landmarks(image_target, image_source,
                                 points_init, points_target, points_warped,
                                 fig_size_max=MAX_FIGURE_SIZE):
    """ composed form several functions - images overlap + landmarks + legend

    :param ndarray image_target: np.array<height, with, dim>
    :param ndarray image_source: np.array<height, with, dim>
    :param ndarray points_target: np.array<nb_points, dim>
    :param ndarray points_init: np.array<nb_points, dim>
    :param ndarray points_warped: np.array<nb_points, dim>
    :param float fig_size_max: maximal figure size for major image dimension
    :return: object

    >>> image = np.random.random((50, 50, 3))
    >>> points = np.array([[20, 30], [40, 10], [15, 25], [5, 50], [10, 60]])
    >>> fig = draw_images_warped_landmarks(image, 1 - image, points, points + 1, points - 1)
    >>> isinstance(fig, plt.Figure)
    True
    >>> fig = draw_images_warped_landmarks(None, None, points, points + 1, points - 1)
    >>> isinstance(fig, plt.Figure)
    True
    >>> draw_images_warped_landmarks(image, None, points, points + 1, points - 1)  # doctest: +ELLIPSIS
    <...>
    >>> draw_images_warped_landmarks(None, image, points, points + 1, points - 1)  # doctest: +ELLIPSIS
    <...>
    """
    # down-scale images and landmarks if they are too large
    (image_target, image_source), (points_init, points_target, points_warped) = \
        scale_large_images_landmarks([image_target, image_source],
                                     [points_init, points_target, points_warped])

    if image_target is not None and image_source is not None:
        image = overlap_two_images(image_target, image_source, transparent=0.5)
    elif image_target is not None:
        image = image_target
    elif image_source is not None:
        image = image_source
    else:
        image = None

    if image is not None:
        im_size = image.shape
        fig, ax = create_figure(im_size, fig_size_max)
        ax.imshow(image)
    else:
        lnds_size = [np.max(pts, axis=0) + np.min(pts, axis=0)
                     for pts in [points_init, points_target, points_warped] if pts is not None]
        im_size = np.max(lnds_size, axis=0).tolist() if lnds_size else (1, 1)
        fig, ax = create_figure(im_size, fig_size_max)

    draw_landmarks_origin_target_warped(ax, points_init, points_target, points_warped)
    ax.legend(loc='lower right', title='Legend')
    ax.set(xlim=[0, im_size[1]], ylim=[im_size[0], 0])
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    return fig


def create_figure(im_size, figsize_max=MAX_FIGURE_SIZE):
    """ create an empty figure of image size maximise maximal size

    :param tuple(int,int) im_size:
    :param float figsize_max:
    :return:

    >>> fig, ax = create_figure((100, 150))
    >>> isinstance(fig, plt.Figure)
    True
    """
    assert len(im_size) >= 2, 'not valid image size - %r' % im_size
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
    logging.debug('exporting Figure: %s', path_fig)
    fig.savefig(path_fig)
    plt.close(fig)


def effective_decimals(num):
    """ find the first effective decimal

    :param float num: number
    :return int: number of the first effective decimals
    """
    dec = 0
    while 0. < num < 1.:
        dec += 1
        num *= 10
    return dec


class RadarChart(object):
    """
    * https://stackoverflow.com/questions/24659005
    * https://datascience.stackexchange.com/questions/6084

    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.random((5, 3)), columns=list('abc'))
    >>> RadarChart(df)  # doctest: +ELLIPSIS
    <...>
    """

    def __init__(self, df, steps=5, fig=None, rect=None, fill_alpha=0.05, colors='nipy_spectral',
                 *args, **kwargs):
        """ draw a dataFrame with scaled axis

        :param df: data
        :param int steps: number of steps per axis
        :param obj|None fig: Figure or None for a new one
        :param tuple(float,float,float,float) rect: rectangle inside figure
        :param float fill_alpha: transparency of filled region
        :param str cmap: used color map
        :param args: optional arguments
        :param kwargs: optional key arguments
        """
        if fig is None:
            fig = plt.figure()
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.titles = list(df.columns)
        self.nb_steps = steps
        self.data = df.copy()
        self.angles = np.linspace(0, 360, len(self.titles), endpoint=False)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                     for i in range(len(self.titles))]
        self.fig = fig

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=self.titles, wrap=True)  # , fontsize=14

        for ax in self.axes[1:]:
            self.__ax_set_invisible(ax)

        for ax, angle, title in zip(self.axes, self.angles, self.titles):
            self.__draw_labels(ax, angle, title)

        self.maxs = np.array([self.data[title].max() for title in self.titles])

        # uf just color space is given, sample colors
        colors = _list_colors(colors, len(self.data))

        for i, (idx, row) in enumerate(self.data.iterrows()):
            self.__draw_curve(idx, row, fill_alpha, color=colors[i], *args, **kwargs)

        self._labels = []
        for ax in self.axes:
            for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
                self.__realign_polar_xtick(ax, theta, label)
                self._labels.append(label)

        self._legend = self.ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.7))

    @classmethod
    def __ax_set_invisible(self, ax):
        ax.patch.set_visible(False)
        ax.grid(False)
        ax.xaxis.set_visible(False)

    def __draw_labels(self, ax, angle, title):
        """ draw some labels

        :param ax:
        :param float angle: angle in degree
        :param str title: name
        """
        vals = np.linspace(self.data[title].min(), self.data[title].max(), self.nb_steps + 1)
        dec = effective_decimals(self.data[title].max()) + 1
        ax.set_rgrids(range(1, self.nb_steps), angle=angle, labels=np.around(vals, dec))
        ax.spines["polar"].set_visible(False)
        # ax.set_ylim(0, 5)

    def __draw_curve(self, idx, row, fill_alpha=0.05, *args, **kw):
        """ draw particular curve

        :param str idx: name
        :param row: data with values
        :param fill_alpha: transparency of filled region
        :param args: optional arguments
        :param kw: optional key arguments
        """
        vals = (row.values / self.maxs * self.nb_steps + 1).tolist()
        vals.append(vals[0])
        angs = self.angles.tolist() + [self.angles[0]]
        self.ax.plot(np.deg2rad(angs), vals, label=idx, *args, **kw)
        self.ax.fill(np.deg2rad(angs), vals, alpha=fill_alpha)

    @classmethod
    def __realign_polar_xtick(self, ax, theta, label):
        """ shift label for particular axis

        :param ax: axis
        :param obj theta:
        :param obj label:
        """
        # https://stackoverflow.com/questions/20222436
        theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
        theta = np.pi / 2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        elif x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        elif y <= -0.5:
            label.set_verticalalignment('top')


def _list_colors(colors, nb):
    """ sample color space

    :param str|list colors:
    :param int nb:
    :return list:

    >>> _list_colors('jet', 2)
    [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.0, 1.0)]
    >>> _list_colors(plt.cm.jet, 3)  # doctest: +ELLIPSIS
    [(0.0, 0.0, 0.5, 1.0), (0.0, 0.0, 0.5..., 1.0), (0.0, 0.0, 0.5..., 1.0)]
    >>> _list_colors([(255, 0, 0), (0, 255, 0)], 1)
    [(255, 0, 0), (0, 255, 0)]
    """
    # uf just color space is given, sample colors
    if isinstance(colors, str):
        colors = plt.get_cmap(colors, nb)
    # assume case that the color is callable plt.cm.jet
    if isinstance(colors, collections.Callable):
        colors = [colors(i) for i in range(nb)]
    return colors


def draw_heatmap(data, row_labels=None, col_labels=None, ax=None,
                 cbar_kw=None, cbar_label="", **kwargs):
    """
    Create a draw_heatmap from a numpy array and two lists of labels.

    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the draw_heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbar_label  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """
    cbar_kw = {} if cbar_kw is None else cbar_kw
    ax = plt.figure(figsize=data.shape[::-1]).gca() if ax is None else ax
    # Plot the draw_heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va='bottom')

    # We want to show all ticks and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(col_labels, va='center')
    else:
        ax.set_xticks([])

    if row_labels is not None:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(row_labels, va='center')
    else:
        ax.set_yticks([])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha='left', rotation_mode='anchor')

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.grid(False)  # for the general grid
    # grid splitting particular color-box, kind of padding
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im, cbar


def draw_matrix_user_ranking(df_stat, higher_better=False, fig=None, cmap='tab20'):
    """ show matrix as image, sorted per column and unique colour per user

    :param DF df_stat: table where index are users and columns are scoring
    :param bool higher_better: ranking such that larger value is better
    :param fig: optional figure
    :param str cmap: color map
    :return Figure:

    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.random((5, 3)), columns=list('abc'))
    >>> draw_matrix_user_ranking(df)  # doctest: +ELLIPSIS
    <...>
    """
    ranking = compute_matrix_user_ranking(df_stat, higher_better)

    if fig is None:
        fig, _ = plt.subplots(figsize=np.array(df_stat.as_matrix().shape[::-1]) * 0.35)
    ax = fig.gca()
    arange = np.linspace(-0.5, len(df_stat) - 0.5, len(df_stat) + 1)
    norm = plt_colors.BoundaryNorm(arange, len(df_stat))
    fmt = plt_ticker.FuncFormatter(lambda x, pos: df_stat.index[x])

    draw_heatmap(ranking, np.arange(1, len(df_stat) + 1), df_stat.columns, ax=ax,
                 cmap=plt.get_cmap(cmap, len(df_stat)), norm=norm,
                 cbar_kw=dict(ticks=range(len(df_stat)), format=fmt),
                 cbar_label='Methods')
    ax.set_ylabel('Ranking')

    fig.tight_layout()
    return fig


def draw_scatter_double_scale(df, colors='nipy_spectral',
                              ax_decs=None,
                              idx_markers=('o', 'd'),
                              xlabel='',
                              figsize=None,
                              legend_style=None,
                              plot_style=None,
                              x_spread=(0.4, 5)):
    """Draw a scatter with double scales on left and right

    :param DF df: dataframe
    :param func cmap: color mapping
    :param dict ax_decs: dictionary with names of left and right axis
    :param tuple idx_markers:
    :param str xlabel: title of x axis
    :param tuple(float,float) figsize:
    :param dict legend_style: legend configuration
    :param dict plot_style: extra plot configuration
    :param tuple(float,int) x_spread: range of spreads and number of samples
    :return tuple: figure and both axis

    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.random((10, 3)), columns=['col1', 'col2', 'col3'])
    >>> fig, axs = draw_scatter_double_scale(df, ax_decs={'name': None}, xlabel='X')
    >>> axs  # doctest: +ELLIPSIS
    {...}
    >>> # just the selected columns
    >>> fig, axs = draw_scatter_double_scale(df, ax_decs={'name1': ['col1', 'col2'],
    ...                                                   'name2': ['col3']})
    >>> fig  # doctest: +ELLIPSIS
    <...>
    >>> # for the "name2" use all remaining columns
    >>> fig, axs = draw_scatter_double_scale(df, ax_decs={'name1': ['col1', 'col2'],
    ...                                                   'name2': None})
    >>> fig  # doctest: +ELLIPSIS
    <...>
    """
    # https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots(figsize=figsize)
    assert isinstance(ax_decs, dict)
    ax_names = list(ax_decs.keys())
    idx_names = list(df.index)

    # uf just color space is given, sample colors
    colors = _list_colors(colors, len(idx_names))

    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
    # https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html
    tab_colors = ('tab:brown', 'tab:gray') if len(ax_names) > 1 else ('black', )

    ax1.set_ylabel(ax_names[0], color=tab_colors[0])
    ax1.grid(True, linestyle='dashed', color=tab_colors[0])
    ax1.tick_params(axis='y', labelcolor=tab_colors[0])
    # automatically fill missing names in the other collections
    if not ax_decs[ax_names[0]]:
        # if it is just one add all columns else just supplement
        ax_decs[ax_names[0]] = [c for c in df.columns if c not in ax_decs[ax_names[1]]] \
            if len(ax_names) > 1 and ax_decs.get(ax_names[1]) else list(df.columns)

    # add second y-axes
    if len(ax_names) == 2:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(ax_names[1], color=tab_colors[1])  # we already handled the x-label with ax1
        ax2.grid(True, linestyle='dotted', color=tab_colors[1])
        ax2.tick_params(axis='y', labelcolor=tab_colors[1])
        # automatically fill missing names in the other collections
        if not ax_decs[ax_names[1]] and ax_decs[ax_names[0]]:
            ax_decs[ax_names[1]] = [c for c in df.columns if c not in ax_decs[ax_names[0]]]
    else:
        ax2 = None

    plot_style = plot_style if plot_style else {}
    # is some spread over x around zero is define
    if x_spread:
        x_offsets = np.linspace(-x_spread[0] / 2., x_spread[0] / 2., x_spread[1])
    else:
        x_offsets = [0]

    for i, col in enumerate(df.columns):
        ax = ax1 if col in ax_decs[ax_names[0]] else ax2
        for j, idx in enumerate(idx_names):
            # print (idx, col, i, df.loc[idx, col])
            mkr = j % len(idx_markers)
            x_off = x_offsets[j % len(x_offsets)]
            ax.plot(i + x_off, df.loc[idx, col], idx_markers[mkr],
                    color=colors[j], label=idx, **plot_style)

    if xlabel:
        ax1.set_xlabel(xlabel)
    # X label ticks - https://stackoverflow.com/questions/43152502
    ax1.set_xticks(range(len(df.columns)))
    ax1.set_xticklabels(df.columns, rotation=45, ha="right")

    # legend - https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    if legend_style is None:
        legend_style = dict(loc='upper center', bbox_to_anchor=(1.25, 1.0), ncol=1)
    lgd = ax1.legend(idx_names, **legend_style)

    extras = {'ax1': ax1, 'ax2': ax2, 'legend': lgd}
    return fig, extras
