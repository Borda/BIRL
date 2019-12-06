"""
Converting images to particular scales

With given path pattern to images crete particular scales within the same set

.. note:: Using these scripts for 1+GB images take several tens of GB RAM

Sample usage::

    python rescale_tissue_images.py \
        -i "/datagrid/Medical/dataset_ANHIR/images_private/COAD_*/scale-100pc/*.png" \
        --scales 5 10 25 50 -ext .jpg --nb_workers 4

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import time
import gc
import logging
import argparse
from functools import partial

import cv2 as cv

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import iterate_mproc_map, is_iterable, nb_workers
from birl.utilities.dataset import (load_large_image, save_large_image,
                                    parse_path_scale, args_expand_parse_images)
from birl.utilities.data_io import create_folder

NB_WORKERS = nb_workers(0.5)
DEFAULT_SCALES = (5, 10, 15, 20, 25, 50)
IMAGE_EXTENSION = '.jpg'
# IMWRITE_PARAMS = (cv.IMWRITE_JPEG_QUALITY, 100)
FOLDER_TEMPLATE = 'scale-%ipc'


def arg_parse_params():
    """ parse the input parameters
    :return dict: parameters
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', type=int, required=False, nargs='+',
                        help='list of output scales', default=DEFAULT_SCALES)
    parser.add_argument('-ext', '--image_extension', type=str, required=False,
                        help='output image extension', default=IMAGE_EXTENSION)
    args = args_expand_parse_images(parser, NB_WORKERS)
    if not is_iterable(args['scales']):
        args['scales'] = [args['scales']]
    logging.info('ARGUMENTS: \n%r' % args)
    return args


def scale_image(img_path, scale, image_ext=IMAGE_EXTENSION, overwrite=False):
    """ scaling images by given scale factor

    :param img_path: input image path
    :param int scale: selected scaling in percents
    :param str image_ext: image extension used on output
    :param bool overwrite: whether overwrite existing image on output
    """
    base = os.path.dirname(os.path.dirname(img_path))
    name, _ = os.path.splitext(os.path.basename(img_path))
    base_scale = parse_path_scale(os.path.dirname(img_path))

    path_dir = os.path.join(base, FOLDER_TEMPLATE % scale)
    create_folder(path_dir)

    path_img_scale = os.path.join(path_dir, name + image_ext)
    if os.path.isfile(path_img_scale) and not overwrite:
        logging.debug('existing "%s"', path_img_scale)
        return

    img = load_large_image(img_path)
    sc = scale / float(base_scale)
    # for down-scaling use just linear
    if sc == 1.:
        img_sc = img
    else:
        interp = cv.INTER_CUBIC if sc > 1 else cv.INTER_LINEAR
        img_sc = cv.resize(img, None, fx=sc, fy=sc, interpolation=interp)
        del img
    gc.collect()
    time.sleep(1)

    logging.debug('creating >> %s', path_img_scale)
    save_large_image(path_img_scale, img_sc)


def wrap_scale_image(img_path_scale, image_ext=IMAGE_EXTENSION, overwrite=False):
    img_path, scale = img_path_scale
    try:
        return scale_image(img_path, scale, image_ext, overwrite)
    except Exception:
        logging.exception('scaling %i of image: %s', scale, img_path)


def main(path_images, scales, image_extension, overwrite, nb_workers):
    """ main entry point

    :param str path_images: path to input images
    :param list(float) scales: define scales in percentage, range (0, 100)
    :param str image_extension: image extension used on output
    :param bool overwrite: whether overwrite existing image on output
    :param int nb_workers: nb jobs running in parallel
    :return:
    """
    image_paths = sorted(glob.glob(path_images))
    image_path_scales = [(im_path, sc) for im_path in image_paths
                         for sc in scales]

    if not image_paths:
        logging.info('No images found on "%s"', path_images)
        return

    _wrap_scale = partial(wrap_scale_image, image_ext=image_extension,
                          overwrite=overwrite)
    list(iterate_mproc_map(_wrap_scale, image_path_scales, desc='Scaling images',
                           nb_workers=nb_workers))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
