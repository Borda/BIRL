"""
Crop images around major object with set padding

Note, for the loading we have to use matplotlib while ImageMagic nor other lib
(opencv, skimage, Pillow) is able to load larger images then 32k.

EXAMPLE
-------
>> python dataset_crop_images.py \
    -i "/datagrid/Medical/dataset_ANHIR/images/COAD_*/scale-100pc/*.png" \
    --padding 0.1 --nb_jobs 3

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import time
import gc
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import cv2 as cv
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.dataset import find_largest_object, project_object_edge
from benchmark.utilities.dataset import load_large_image, save_large_image
from benchmark.utilities.experiments import wrap_execute_sequence

NB_THREADS = int(mproc.cpu_count() * .5)
SCALE_SIZE = 512
CUT_DIMENSION = 0
TISSUE_CONTENT = 0.05


def arg_parse_params():
    """ parse the input parameters
    :return dict: {str: any}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_images', type=str, required=True,
                        help='path (pattern) to the input image')
    parser.add_argument('--padding', type=float, required=False, default=0.5,
                        help='padding around the object in image percents')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_images'] = os.path.expanduser(args['path_images'])
    logging.info('ARGUMENTS: \n%s' % repr(args))
    return args


def crop_image(img_path, crop_dim, padding=0.15):
    img = load_large_image(img_path)
    scale_factor = max(1, img.shape[crop_dim] / float(SCALE_SIZE))
    # work with just a scaled version
    sc = 1. / scale_factor
    order = cv.INTER_LINEAR if scale_factor > 1 else cv.INTER_CUBIC
    img_small = 255 - cv.resize(img, None, fx=sc, fy=sc, interpolation=order)

    img_edge = project_object_edge(img_small, crop_dim)
    del img_small

    begin, end = find_largest_object(img_edge, threshold=TISSUE_CONTENT)
    img_diag = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
    pad_px = padding * img_diag
    begin_px = max(0, int((begin * scale_factor) - pad_px))
    end_px = min(img.shape[crop_dim], int((end * scale_factor) + pad_px))

    if crop_dim == 0:
        img = img[begin_px:end_px, ...]
    elif crop_dim == 1:
        img = img[:, begin_px:end_px, ...]
    else:
        raise Exception('unsupported dimension %i' % crop_dim)

    save_large_image(img_path, img)
    gc.collect(), time.sleep(1)


def wrap_img_crop(img_path_dim, padding=0.15):
    img_path, dim = img_path_dim
    return crop_image(img_path, crop_dim=dim, padding=padding)


def main(path_images, padding, nb_jobs):
    image_paths = sorted(glob.glob(path_images))

    image_paths_dims = [(p_img, d) for p_img in image_paths for d in range(2)]
    _wrap_crop = partial(wrap_img_crop, padding=padding)
    list(wrap_execute_sequence(_wrap_crop, image_paths_dims,
                               desc='Cut image objects', nb_jobs=nb_jobs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    main(arg_params['path_images'], arg_params['padding'],
         arg_params['nb_jobs'])

    logging.info('DONE')
