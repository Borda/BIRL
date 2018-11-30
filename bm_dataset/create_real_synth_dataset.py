"""
Script for generating synthetic datasets from a single image and landmarks.
The output is set of geometrical deformed images with also change color space
and related computed new landmarks.

Example run:
>> python create_real_synth_dataset.py \
    -i ../data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.jpg \
    -l ../data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.csv \
    -o ../output/synth_dataset --visual

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
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
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import tqdm
import numpy as np
import pandas as pd
import cv2 as cv
from scipy import ndimage, stats, interpolate
import matplotlib.pyplot as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import parse_arg_params
from birl.utilities.data_io import LANDMARK_COORDS

COLUMNS_COORD = LANDMARK_COORDS
NB_THREADS = max(1, int(mproc.cpu_count() * .8))
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
    parser.add_argument('-i', '--path_image', type=str, required=True,
                        help='path to the input image')
    parser.add_argument('-l', '--path_landmarks', type=str, required=True,
                        help='path to the input landmarks')
    parser.add_argument('-o', '--path_out', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('-n', '--nb_samples', type=int, required=False,
                        help='number of deformed images', default=NB_DEFORMATIONS)
    parser.add_argument('--visual', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_workers', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    args = parse_arg_params(parser, upper_dirs=['path_out'])
    args['visual'] = bool(args['visual'])
    return args


def estimate_transformation_tps(shape, points, max_deform=DEFORMATION_MAX,
                                nb_bound_points=5):
    """ generate deformation as thin plate spline deformation

    SEE" http://answers.opencv.org/question/186368

    :param (int, int) shape: tuple of size 2
    :param points: np.array<nb_points, 2> list of landmarks
    :param float max_deform: maximal deformation distance in any direction
    :param int nb_bound_points: number of fix boundary points
    :return: np.array<shape>
    """
    points_source = np.array(points)
    # generate random shifting
    # todo: normalise the random shift by distance to nearest landmarks
    shift = (np.random.random(points.shape) - 0.5) * max_deform
    points_target = np.array(points) + shift
    # fix boundary points
    if nb_bound_points > 1:
        bound_one = np.ones(nb_bound_points - 1)
        # set the boundary points
        x_range = np.round(np.linspace(0, shape[0] - 1, nb_bound_points), 0)
        y_range = np.round(np.linspace(0, shape[1] - 1, nb_bound_points), 0)
        x_bound = np.hstack((0 * bound_one, x_range[:-1],
                             (shape[0] - 1) * bound_one, x_range[::-1][:-1]))
        y_bound = np.hstack((y_range[:-1], (shape[1] - 1) * bound_one,
                             y_range[::-1][:-1], 0 * bound_one))
        boundary = np.vstack((x_bound, y_bound)).T
        # extend the points
        points_source = np.vstack((points_source, boundary))
        points_target = np.vstack((points_target, boundary))

    assert len(points_source) == len(points_target)
    matches = [cv.DMatch(i, i, 0) for i in range(len(points_source))]

    tps = cv.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(points_source.astype(np.float32).reshape(1, -1, 2),
                               points_target.astype(np.float32).reshape(1, -1, 2),
                               matches)

    return tps


def deform_image_landmarks(image, points, max_deform=DEFORMATION_MAX):
    """ deform the image by randomly generated deformation field
    and compute new positions for all landmarks

    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param float max_deform: maximal deformation distance in any direction
    :return: np.array<height, width, 3>, np.array<nb_points, 2>
    """
    nb_fix_points = int(np.max(image.shape) / max_deform * 0.5)
    tps = estimate_transformation_tps(image.shape[:2], points,
                                      max_deform, nb_fix_points)

    img_warped = tps.warpImage(image)
    ret, out = tps.applyTransformation(np.array(points).astype(np.float32).reshape(1, -1, 2))
    pts_warped = out[0]

    return img_warped, pts_warped


def image_color_hsv_shift(image, change_satur=True):
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
    if image.max() > 1.5:
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
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image)
    ax.plot(points[:, 1], points[:, 0], 'o', color='k')
    ax.plot(points[:, 1], points[:, 0], '.', color='w')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    fig.tight_layout()
    return fig


def export_image_landmarks(image, points, idx, path_out, name_img,
                           visual=False):
    """ export the image, landmarks as csv file and if the 'visual' is set,
    draw also landmarks in the image (in separate image)

    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param int idx:
    :param str path_out: path to the output directory
    :param str name_img: image file name
    :param bool visual:
    """
    if image.max() <= 1.:
        image = (image * 255).astype(np.uint8)
    # export the image
    path_image = os.path.join(path_out, name_img + '_%i.jpg' % idx)
    logging.debug('exporting image #%i: %s', idx, path_image)
    cv.imwrite(path_image, image)
    # export landmarks
    path_csv = os.path.join(path_out, name_img + '_%i.csv' % idx)
    logging.debug('exporting points #%i: %s', idx, path_csv)
    pd.DataFrame(points, columns=COLUMNS_COORD).to_csv(path_csv)
    if visual:  # visualisation
        fig = draw_image_landmarks(image, points)
        path_fig = os.path.join(path_out, name_img + '_%i_landmarks.pdf' % idx)
        fig.savefig(path_fig)
        plt.close(fig)


def perform_deform_export(idx, image, points, path_out, name_img, visual=False):
    """ perform complete image colour change, and deformation on image
    and landmarks and if required draw a visualisation

    :param int idx:
    :param image: np.array<height, width, 3>
    :param points: np.array<nb_points, 2>
    :param str path_out:
    :param str name_img:
    :param bool visual:
    """
    image_ = image_color_hsv_shift(image)
    max_deform = int(0.03 * np.mean(image_.shape[:2]))
    image_, points_ = deform_image_landmarks(image_, points, max_deform)
    export_image_landmarks(image_, points_, idx + 1, path_out, name_img,
                           visual)


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

    image = np.array(plt.imread(params['path_image']))
    logging.debug('loaded image, shape: %s', image.shape)
    df_points = pd.read_csv(params['path_landmarks'], index_col=0)
    points = df_points[COLUMNS_COORD].values
    logging.debug('loaded landmarks, dim: %s', points.shape)

    name_img = get_name(params['path_image'])
    # name_points = get_name(params['path_landmarks'])

    export_image_landmarks(image, points, 0, params['path_out'],
                           name_img, visual=params['visual'])

    # create the wrapper for parallel usage
    wrapper_deform_export = partial(perform_deform_export, image=image,
                                    points=points, path_out=params['path_out'],
                                    name_img=name_img,
                                    visual=params.get('visual', False))

    tqdm_bar = tqdm.tqdm(total=params['nb_samples'])
    if params['nb_workers'] > 1:
        mproc_pool = mproc.Pool(params['nb_workers'])
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

    arg_params = arg_parse_params()
    main(arg_params)
