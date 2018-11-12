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

import numpy as np
from skimage.transform import rescale

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.dataset import find_largest_object, project_object_edge
from benchmark.utilities.dataset import load_large_image, save_large_image
from benchmark.utilities.experiments import wrap_execute_sequence

NB_THREADS = int(mproc.cpu_count() * .5)
SCALE_FACTOR = 100.
CUT_DIMENSION = 0


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


def crop_image(img_path, crop_dime, padding=0.15):
    img = load_large_image(img_path)
    # work with just a scaled version
    img_small = 1. - rescale(img, 1. / SCALE_FACTOR, order=0, multichannel=True,
                             mode='constant', anti_aliasing=True,
                             preserve_range=True)
    img_edge = project_object_edge(img_small, crop_dime)
    del img_small

    begin, end = find_largest_object(img_edge, threshold=0.05)
    img_diag = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
    pad_px = padding * img_diag
    begin_px = max(0, int((begin * SCALE_FACTOR) - pad_px))
    end_px = min(img.shape[crop_dime], int((end * SCALE_FACTOR) + pad_px))

    if crop_dime == 0:
        img = img[begin_px:end_px, ...]
    elif crop_dime == 1:
        img = img[:, begin_px:end_px, ...]
    else:
        raise Exception('unsupported dimension %i' % crop_dime)

    save_large_image(img_path, img)
    gc.collect(), time.sleep(1)


def wrap_img_crop(img_path_dim, padding=0.15):
    img_path, dim = img_path_dim
    return crop_image(img_path, crop_dime=dim, padding=padding)


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
