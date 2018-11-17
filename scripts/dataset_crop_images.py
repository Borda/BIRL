"""
Crop images around major object with set padding

Note, that using these scripts for 1+GB images take several tens of GB RAM

EXAMPLE
-------
>> python dataset_crop_images.py \
    -i "/datagrid/Medical/dataset_ANHIR/images/COAD_*/scale-100pc/*.png" \
    --padding 0.1 --nb_jobs 2

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
TISSUE_CONTENT = 0.01


def arg_parse_params():
    """ parse the input parameters
    :return dict: {str: any}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_images', type=str, required=True,
                        help='path (pattern) to the input image')
    parser.add_argument('--padding', type=float, required=False, default=0.1,
                        help='padding around the object in image percents')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_images'] = os.path.expanduser(args['path_images'])
    logging.info('ARGUMENTS: \n%s' % repr(args))
    return args


def crop_image(img_path, crop_dims=(0, 1), padding=0.15):
    img = load_large_image(img_path)
    scale_factor = max(1, np.mean(img.shape[:2]) / float(SCALE_SIZE))
    # work with just a scaled version
    sc = 1. / scale_factor
    order = cv.INTER_AREA if scale_factor > 1 else cv.INTER_LINEAR
    img_small = 255 - cv.resize(img, None, fx=sc, fy=sc, interpolation=order)

    crops = {}
    for crop_dim in crop_dims:
        assert crop_dim in (0, 1), 'not supported dimension'
        img_edge = project_object_edge(img_small, crop_dim)

        begin, end = find_largest_object(img_edge, threshold=TISSUE_CONTENT)
        # img_diag = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
        pad_px = padding * (end - begin) * scale_factor
        begin_px = max(0, int((begin * scale_factor) - pad_px))
        end_px = min(img.shape[crop_dim], int((end * scale_factor) + pad_px))
        crops[crop_dim] = (begin_px, end_px)
    del img_small

    for dim in range(2):
        if 0 not in crops:
            crops[0] = (0, img.shape[0])

    img = img[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], ...]

    save_large_image(img_path, img)
    gc.collect(), time.sleep(1)


def wrap_img_crop(img_path, padding=0.1):
    try:
        crop_image(img_path, crop_dims=(0, 1), padding=padding)
    except Exception:
        logging.exception('crop image: %s', img_path)


def main(path_images, padding, nb_jobs):
    image_paths = sorted(glob.glob(path_images))

    if not image_paths:
        logging.info('No images found on "%s"', path_images)
        return

    _wrap_crop = partial(wrap_img_crop, padding=padding)
    list(wrap_execute_sequence(_wrap_crop, image_paths,
                               desc='Crop image tissue', nb_jobs=nb_jobs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    main(arg_params['path_images'], arg_params['padding'],
         arg_params['nb_jobs'])

    logging.info('DONE')
