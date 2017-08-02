"""
Script for generating synthetic datasets from a single image and landmarks.
The output is set of geometrical deformed images with also change color space
and related computed new landmarks.

Example run:
>> python create_synth_dataset_real_image.py \
    -img ../data_images/images/Rat_Kidney_HE.jpg \
    -lnd ../data_images/landmarks/Rat_Kidney_HE.csv \
    -out ../output/synth_dataset

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import argparse
import logging
import multiprocessing as mproc
from functools import partial

import matplotlib
# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import tqdm
import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
from scipy import ndimage, stats, interpolate
import matplotlib.pyplot as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import benchmark.utils.experiments as tl_expt
import benchmark.utils.data_io as tl_io

COLUMNS_COORD = tl_io.LANDMARK_COORDS
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
    :return dict: {str: str}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--path_image', type=str, required=True,
                        help='path to the input image')
    parser.add_argument('-lnd', '--path_landmarks', type=str, required=True,
                        help='path to the input landmarks')
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('-nb', '--nb_samples', type=int, required=False,
                        help='number of deromed images',
                        default=NB_DEFORMATIONS)
    parser.add_argument('--visual', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    logging.info(tl_expt.string_dict(args, 'ARGUMENTS:'))
    assert tl_expt.check_paths(args, ['path_out'])
    args['visual'] = bool(args['visual'])
    return args


def generate_deformation_field_gauss(shape, points, max_deform=DEFORMATION_MAX,
                                     deform_smooth=DEFORMATION_SMOOTH):
    """ generate deformation field as combination of positive and
    negative Galatians densities scaled in range +/- max_deform

    :param (int, int) shape: tuple of size 2
    :param points: <nb_points, 2> list of landmarks
    :param float max_deform: maximal deformation distance in any direction
    :param float deform_smooth: smoothing the deformation by Gaussian filter
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

    :param (int, int) shape: tuple of size 2
    :param points: np.array<nb_points, 2> list of landmarks
    :param float max_deform: maximal deformation distance in any direction
    :param int nb_bound_points: number of fix boundary points
    :return: np.array<shape>
    """
    # x_point = points[:, 0]
    # y_point = points[:, 1]
    # generate random shifting
    move = (np.random.random(points.shape[0]) - 0.5) * max_deform

    # fix boundary points
    # set the boundary points
    bound = np.ones(nb_bound_points - 1)
    x_bound = np.linspace(0, shape[0] - 1, nb_bound_points)
    y_bound = np.linspace(0, shape[1] - 1, nb_bound_points)
    x_point = np.hstack((points[:, 0], 0 * bound, x_bound[:-1],
                         (shape[0] - 1) * bound, x_bound[::-1][:-1]))
    y_point = np.hstack((points[:, 1], y_bound[:-1], (shape[1] - 1) * bound,
                         y_bound[::-1][:-1], 0 * bound))
    # the boundary points sex as 0 shift
    move = np.hstack((move, np.zeros(4 * nb_bound_points - 4)))
    # create the interpolation function
    smooth = 0.2 * max_deform
    rbf = interpolate.Rbf(x_point, y_point, move, function='thin-plate',
                          epsilon=1, smooth=smooth)
    # interpolate in regular grid
    x_grid, y_grid = np.mgrid[0:shape[0], 0:shape[1]].astype(np.int32)
    # FIXME: it takes to much of RAM memory, for sample image more that 8GM !
    deform = rbf(x_grid, y_grid)
    return deform


def deform_image_landmarks(image, points, max_deform=DEFORMATION_MAX):
    """ deform the image by randomly generated deformation field
    and compute new positions for all landmarks

    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param float max_deform: maximal deformation distance in any direction
    :return: np.array<height, width, 3>, np.array<nb_points, 2>
    """
    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    # generate the deformation field
    nb_fix_points = int(np.max(image.shape) / max_deform * 2.)
    x_deform = generate_deformation_field_rbf(image.shape[:2], points,
                                              max_deform, nb_fix_points)
    # TODO: look for another elastic deformation which is friendly to Memory usage
    # -> generate random elastic deformation and using this field get new landmarks
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


def image_color_shift_hue(image, change_satur=True):
    """ take the original image and shift the colour space in HUE

    :param image: np.array<height, width, 3>
    :param bool change_satur: whether change also the saturation
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
    if change_satur:
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
    :param int idx:
    :param str path_out: path to the output directory
    :param str name_img: image file name
    :param str name_points: landmarks file name
    :param bool visual:
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

    :param int idx:
    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param str path_out:
    :param str name_img:
    :param str name_points:
    :param bool visual:
    """
    image_out = image_color_shift_hue(image)
    max_deform = int(0.03 * np.mean(image.shape[:2]))
    image_out, points_out = deform_image_landmarks(image_out, points,
                                                   max_deform)
    export_image_landmarks(image_out, points_out, idx + 1, path_out,
                           name_img, name_points, visual)


def get_name(path):
    """ parse the name without extension from complete path

    :param str path:
    :return str:
    """
    return os.path.splitext(os.path.basename(path))[0]


def main(params):
    """ main entry point

    :param dict params: {str: str}
    """
    logging.info('running...')

    if not os.path.isdir(params['path_out']):
        logging.info('creating folder: %s', params['path_out'])
        os.mkdir(params['path_out'])
    else:
        logging.warning('using existing folder: %s', params['path_out'])

    image = np.array(Image.open(params['path_image']))
    logging.debug('loaded image, shape: %s', image.shape)
    df_points = pd.DataFrame.from_csv(params['path_landmarks'])
    points = df_points[COLUMNS_COORD].values
    logging.debug('loaded landmarks, dim: %s', points.shape)

    name_img = get_name(params['path_image'])
    name_points = get_name(params['path_landmarks'])

    export_image_landmarks(image, points, 0, params['path_out'],
                           name_img, name_points, visual=params['visual'])

    # create the wrapper for parallel usage
    wrapper_deform_export = partial(perform_deform_export, image=image,
                                    points=points, path_out=params['path_out'],
                                    name_img=name_img, name_points=name_points,
                                    visual=params['visual'])

    tqdm_bar = tqdm.tqdm(total=params['nb_samples'])
    if params['nb_jobs'] > 1:
        mproc_pool = mproc.Pool(params['nb_jobs'])
        for _ in mproc_pool.imap_unordered(wrapper_deform_export,
                                           range(params['nb_samples'])):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for i in range(params['nb_samples']):
            wrapper_deform_export(i)
            tqdm_bar.update()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    params = arg_parse_params()
    main(params)
