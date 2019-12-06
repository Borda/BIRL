"""
Splitting image containing two samples

.. note:: Using these scripts for 1+GB images take several tens of GB RAM

Sample usage::

    python split_images_two_tissues.py \
        -i "/datagrid/Medical/dataset_ANHIR/images/COAD_*/scale-100pc/*_*.png" \
        --nb_workers 3

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
from birl.utilities.dataset import (find_split_objects, project_object_edge, load_large_image,
                                    save_large_image, args_expand_parse_images)
from birl.utilities.experiments import iterate_mproc_map, nb_workers

NB_WORKERS = nb_workers(0.5)
#: use following image size for estimating cutting line
SCALE_SIZE = 512
#: cut image in one dimension/axis
CUT_DIMENSION = 0


def arg_parse_params():
    """ parse the input parameters
    :return dict: parameters
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, required=False, choices=[0, 1],
                        help='cutting dimension', default=CUT_DIMENSION)
    args = args_expand_parse_images(parser, NB_WORKERS)
    logging.info('ARGUMENTS: \n%r' % args)
    return args


def split_image(img_path, overwrite=False, cut_dim=CUT_DIMENSION):
    """ split two images in single dimension

    the input images assume to contain two names in the image name separated by "_"

    :param str img_path: path to the input / output image
    :param bool overwrite: allow overwrite exiting output images
    :param int cut_dim: define splitting dimension
    """
    name, ext = os.path.splitext(os.path.basename(img_path))
    folder = os.path.dirname(img_path)
    obj_names = name.split('_')
    paths_img = [os.path.join(folder, obj_name + ext) for obj_name in obj_names]

    if all(os.path.isfile(p) for p in paths_img) and not overwrite:
        logging.debug('existing all splits of %r', paths_img)
        return

    img = load_large_image(img_path)
    # work with just a scaled version
    scale_factor = max(1, img.shape[cut_dim] / float(SCALE_SIZE))
    sc = 1. / scale_factor
    order = cv.INTER_AREA if scale_factor > 1 else cv.INTER_LINEAR
    img_small = 255 - cv.resize(img, None, fx=sc, fy=sc, interpolation=order)
    img_edge = project_object_edge(img_small, cut_dim)
    del img_small

    # prepare all cut edges and scale them to original image size
    splits = find_split_objects(img_edge, nb_objects=len(obj_names))
    if not splits:
        logging.error('no splits found for %s', img_path)
        return

    edges = [int(round(i * scale_factor))
             for i in [0] + splits + [len(img_edge)]]

    # cutting images
    for i, path_img_cut in enumerate(paths_img):
        if os.path.isfile(path_img_cut) and not overwrite:
            logging.debug('existing "%s"', path_img_cut)
            continue
        if cut_dim == 0:
            img_cut = img[edges[i]:edges[i + 1], ...]
        elif cut_dim == 1:
            img_cut = img[:, edges[i]:edges[i + 1], ...]
        else:
            raise Exception('unsupported dimension: %i' % cut_dim)
        save_large_image(path_img_cut, img_cut)
        gc.collect()
        time.sleep(1)


def main(path_images, dimension, overwrite, nb_workers):
    """ main entry point

    :param path_images: path to images
    :param int dimension: for 2D inages it is 0 or 1
    :param bool overwrite: whether overwrite existing image on output
    :param int nb_workers: nb jobs running in parallel
    """
    image_paths = sorted(glob.glob(path_images))

    if not image_paths:
        logging.info('No images found on "%s"', path_images)
        return

    _wrap_split = partial(split_image, cut_dim=dimension, overwrite=overwrite)
    list(iterate_mproc_map(_wrap_split, image_paths, desc='Cut image tissues',
                           nb_workers=nb_workers))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
