"""
Script for generating synthetic datasets from a single image and landmarks.
The output is set of geometrical deformed images with also change color space
and related computed new landmarks.

Example run:
>> python create_dataset_real_image_synthetic_deformation.py \
    -img ../data/images/Rat_Kidney_HE.jpg \
    -lnd ../data/landmarks/Rat_Kidney_HE.csv \
    -out ../output/synth_dataset

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import argparse
import logging
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
from scipy import ndimage, stats, interpolate
import matplotlib.pyplot as plt

COLUMNS_COORD = ['Y', 'X']
NB_THREADS = int(mproc.cpu_count() * .8)
NB_DEFORMATIONS = 5
HUE_SHIFT_MIN = 20
HUE_SHIFT_MAX = 120
FIG_MAX_SIZE = 16
DEFORMATION_MAX = 50
DEFORMATION_SMOOTH = 25
DEFORMATION_BOUNDARY_COEF = 3


def arg_parse_params():
    """ parse the input parameters
    SEE: https://docs.python.org/3/library/argparse.html
    :return: {str: str}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--path_image', type=str, required=True,
                        help='path to the input image')
    parser.add_argument('-lnd', '--path_landmarks', type=str, required=True,
                        help='path to the input landmarks')
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('-nb', '--nb_deforms', type=int, required=False,
                        help='number of deromed images',
                        default=NB_DEFORMATIONS)
    parser.add_argument('--visu', type=bool, required=False, default=False,
                        help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMS: \n %s', repr(args))
    for k in (k for k in args if 'path' in k):
        args[k] = os.path.abspath(os.path.expanduser(args[k]))
        assert os.path.exists(args[k]), '%s' % args[k]
    args['visu'] = bool(args['visu'])
    return args


def generate_deformation_field_gauss(shape, points, max_deform=DEFORMATION_MAX,
                                     deform_smooth=DEFORMATION_SMOOTH):
    """ generate deformation field as combination of positive and
    negative Galatians densities scaled in range +/- max_deform

    :param shape: tuple of size 2
    :param points: np.array<nb_points, 2> list of landmarks
    :param max_deform: float, maximal deformation distance in any direction
    :param deform_smooth: float, smoothing the deformation by Gaussian filter
    :return: np.array<shape>
    """
    ndim = len(shape)
    x, y = np.mgrid[0:shape[0], 0:shape[1]]
    pos_grid = np.rollaxis(np.array([x, y]), 0, 3)
    # initialise the deformation
    deform = np.zeros(shape)
    for point in points:
        sign = np.random.choice([-1, 1])
        cov = np.random.random((ndim, ndim))
        cov[np.eye(ndim, dtype=bool)] = 100 * np.random.random(ndim)
        # obtain a positive semi-definite matrix
        cov = np.dot(cov, cov.T) * (0.1 * np.mean(shape))
        gauss = stats.multivariate_normal(point, cov)
        deform += sign * gauss.pdf(pos_grid)
    # normalise the deformation and multiply by the amplitude
    deform *= max_deform / np.abs(deform).max()
    # set boundary region to zeros
    fix_deform_bounds = DEFORMATION_BOUNDARY_COEF * deform_smooth
    deform[:fix_deform_bounds, :] = 0
    deform[-fix_deform_bounds:, :] = 0
    deform[:, :fix_deform_bounds] = 0
    deform[:, -fix_deform_bounds:] = 0
    # smooth the deformation field
    deform = ndimage.gaussian_filter(deform, sigma=deform_smooth, order=0)
    return deform


def generate_deformation_field_rbf(shape, points, max_deform=DEFORMATION_MAX,
                                   nb_bound_points=25):
    """ generate deformation field as thin plate spline  deformation
    in range +/- max_deform

    :param shape: tuple of size 2
    :param points: np.array<nb_points, 2> list of landmarks
    :param max_deform: float, maximal deformation distance in any direction
    :param nb_bound_points: int, number of fix boundary points
    :return: np.array<shape>
    """
    x_point = points[:, 0]
    y_point = points[:, 1]
    # generate random shifting
    move = (np.random.random(points.shape[0]) - 0.5) * max_deform

    # fix boundary points
    # set the boundary points
    bound = np.ones(nb_bound_points - 1)
    x_bound = np.linspace(0, shape[0] - 1, nb_bound_points)
    y_bound = np.linspace(0, shape[1] - 1, nb_bound_points)
    x_point = np.hstack((x_point, 0 * bound, x_bound[:-1],
                         (shape[0] - 1) * bound, x_bound[::-1][:-1]))
    y_point = np.hstack((y_point, y_bound[:-1], (shape[1] - 1) * bound,
                         y_bound[::-1][:-1], 0 * bound))
    # the boundary points sex as 0 shift
    move = np.hstack((move, np.zeros(4 * nb_bound_points - 4)))
    # create the interpolation function
    smooth = 0.5 * max_deform
    rbf = interpolate.Rbf(x_point, y_point, move, function='thin-plate',
                          epsilon=1, smooth=smooth)
    # interpolate in regular grid
    x_grid, y_grid = np.mgrid[0:shape[0], 0:shape[1]]
    deform = rbf(x_grid, y_grid)
    return deform


def deform_image_landmarks(image, points, max_deform=DEFORMATION_MAX):
    """ deform the image by randomly generated deformation field
    and compute new positions for all landmarks

    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param max_deform: float, maximal deformation distance in any direction
    :return: np.array<height, width, 3>, np.array<nb_points, 2>
    """
    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    # generate the deformation field
    nb_fix_points = int(np.max(image.shape) / max_deform * 2.)
    x_deform = generate_deformation_field_rbf(image.shape[:2], points,
                                              max_deform, nb_fix_points)
    y_deform = generate_deformation_field_rbf(image.shape[:2], points,
                                              max_deform, nb_fix_points)
    # interpolate the image
    img_warped = interpolate.griddata(zip(x.ravel(), y.ravel()),
                                      image.reshape(-1, 3),
                                      (x + x_deform, y + y_deform),
                                      method='linear', fill_value=1.)
    # compute new positions of landmarks
    x_new = x - x_deform
    y_new = y - y_deform
    pts_warped = np.array([[x_new[pt[0], pt[1]], y_new[pt[0], pt[1]]]
                           for pt in points])
    return img_warped, pts_warped


def image_color_shift_hue(image, sat_change=True):
    """ take the original image and shift the colour space in HUE

    :param image: np.array<height, width, 3>
    :param sat_change: bool whether change also the saturation
    :return: np.array<height, width, 3>

    """
    # generate hue shift
    h_shift = np.random.randint(HUE_SHIFT_MIN, HUE_SHIFT_MAX)
    h_shift *= -1 if np.random.random() < 0.5 else 1
    # generate saturation power
    s_power = 0.3 + np.random.random()
    logging.debug('image color change with Hue shift %d and Sat power %f',
                  h_shift, s_power)
    # convert image into range (0, 1)
    if image.max() > 1.:
        image = (image / 255.)

    img_hsv = matplotlib.colors.rgb_to_hsv(image)
    # color transformation
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + (h_shift / 360.0)) % 1.0
    if sat_change:
        img_hsv[:, :, 1] = img_hsv[:, :, 1] ** s_power

    image = matplotlib.colors.hsv_to_rgb(img_hsv)
    return image


def draw_image_landmarks(image, points):
    """ draw landmarks over the image and return the figure

    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :return: object
    """
    shape = np.array(image.shape[:2])
    fig_size = shape / float(max(shape)) * FIG_MAX_SIZE
    fig_size = fig_size.tolist()[-1::-1]
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()
    ax.imshow(image)
    ax.plot(points[:, 1], points[:, 0], 'o', color='k')
    ax.plot(points[:, 1], points[:, 0], '.', color='w')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    fig.tight_layout()
    return fig


def export_image_landmarks(image, points, idx, path_out, name_img, name_points,
                           visual=False):
    """ export the image, landmarks as csv file and if the 'visual' is set,
    draw also landmarks in the image (in separate image)

    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param idx: int
    :param path_out: str, path to the output directory
    :param name_img: str, image file name
    :param name_points: str, landmarks file name
    :param visual: bool
    """
    if image.max() <= 1.:
        image = (image * 255).astype(np.uint8)
    # export the image
    path_image = os.path.join(path_out, name_img + '_%i.jpg' % idx)
    logging.debug('exporting image #%i: %s', idx, path_image)
    Image.fromarray(image).save(path_image)
    # export landmarks
    path_csv = os.path.join(path_out, name_points + '_%i.csv' % idx)
    logging.debug('exporting points #%i: %s', idx, path_csv)
    pd.DataFrame(points, columns=COLUMNS_COORD).to_csv(path_csv)
    if visual:  # visualisation
        fig = draw_image_landmarks(image, points)
        path_fig = os.path.join(path_out, name_img + '_%i_landmarks.png' % idx)
        fig.savefig(path_fig)
        plt.close(fig)


def perform_deform_export(idx, image, points, path_out, name_img, name_points,
                          visual=False):
    """ perform complete image colour change, and deformation on image
    and landmarks and if required draw a visualisation

    :param idx: int
    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param path_out: str
    :param name_img: str
    :param name_points: str
    :param visual: bool
    """
    image_out = image_color_shift_hue(image)
    max_deform = int(0.03 * np.mean(image.shape[:2]))
    image_out, points_out = deform_image_landmarks(image_out, points,
                                                   max_deform)
    export_image_landmarks(image_out, points_out, idx + 1, path_out,
                           name_img, name_points, visual)


def get_name(path):
    """ parse the name without extension from complete path

    :param path: str
    :return: str
    """
    return os.path.splitext(os.path.basename(path))[0]


def main(params):
    """ main entry point

    :param params: {str: str}
    """
    logging.info('running...')

    image = np.array(Image.open(params['path_image']))
    logging.debug('loaded image, shape: %s', image.shape)
    df_points = pd.DataFrame.from_csv(params['path_landmarks'])
    points = df_points[COLUMNS_COORD].values
    logging.debug('loaded landmarks, dim: %s', points.shape)

    name_img = get_name(params['path_image'])
    name_points = get_name(params['path_landmarks'])

    export_image_landmarks(image, points, 0, params['path_out'],
                           name_img, name_points, visual=params['visu'])

    # create the wrapper for parallel usage
    wrapper_deform_export = partial(perform_deform_export, image=image,
                                    points=points, path_out=params['path_out'],
                                    name_img=name_img, name_points=name_points,
                                    visual=params['visu'])

    tqdm_bar = tqdm.tqdm(total=params['nb_deforms'])
    if params['nb_jobs'] > 1:
        mproc_pool = mproc.Pool(params['nb_jobs'])
        for r in mproc_pool.imap_unordered(wrapper_deform_export,
                                           range(params['nb_deforms'])):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for i in range(params['nb_deforms']):
            wrapper_deform_export(i)
            tqdm_bar.update()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    params = arg_parse_params()
    main(params)
