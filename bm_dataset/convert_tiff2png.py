"""
Converting TIFF, SVS images to PNG

Convert the original TIFF image containing a pyramid to single image
of a particular level (regarding the pyramid) using mosaic techniques

Be aware that this conversion takes lots of memory, for example
image of size 50k x 60k takes about 10GB in RAM

Sample usage::

    python convert_tiff2png.py -l 0 --nb_workers 2 \
        -i "/datagrid/Medical/dataset_ANHIR/images_raw/*/*.tiff"
    python convert_tiff2png.py -l 1 --nb_workers 5 --overwrite \
        -i "/datagrid/Medical/dataset_ANHIR/images_raw/*/*.svs"


Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import time
import gc
import multiprocessing as mproc
from functools import partial

import tqdm
import cv2 as cv
import numpy as np
try:
    from openslide import OpenSlide
except Exception:
    print('It seems that you do not have installed OpenSlides on your computer.'
          ' To do so, please follow instructions - https://openslides.org')

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import iterate_mproc_map
from birl.utilities.dataset import args_expand_parse_images

DEFAULT_LEVEL = 1
MAX_LOAD_IMAGE_SIZE = 16000
IMAGE_EXTENSION = '.png'
NB_THREADS = max(1, int(mproc.cpu_count() * .5))


def arg_parse_params():
    """ parse the input parameters
    :return dict: parameters
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', type=int, required=False,
                        help='list of output scales', default=DEFAULT_LEVEL)
    args = args_expand_parse_images(parser, NB_THREADS)
    logging.info('ARGUMENTS: \n%r' % args)
    return args


def convert_image(path_img, level=DEFAULT_LEVEL, overwrite=False):
    """ convert TIFF/SVS image to standard format
    The output image has the same name and it is exported in the same folder

    :param str path_img: path to the input image
    :param int level: selected level of the internal pyramid representation
        the level 0 means full scale and higher number is small image in pyramid scaling
    :param bool overwrite: whether overwrite existing image on output
    """
    slide_img = OpenSlide(path_img)
    assert level < len(slide_img.level_dimensions), \
        'unsupported level %i of %i' % (level, slide_img.level_count)

    path_img_new = os.path.splitext(path_img)[0] + IMAGE_EXTENSION
    if os.path.isfile(path_img_new) and not overwrite:
        logging.warning('existing "%s"', path_img_new)
        return

    level_size = slide_img.level_dimensions[level]
    level_scale = slide_img.level_downsamples[level]

    level_downsample = 1
    while max(np.array(level_size) / level_downsample) > MAX_LOAD_IMAGE_SIZE:
        level_downsample *= 2
    logging.debug('using down-sample: %i', level_downsample)

    tile_size = (np.array(level_size) / level_downsample).astype(int)
    locations = [(i * tile_size[0], j * tile_size[1])
                 for i in range(level_downsample)
                 for j in range(level_downsample)]
    im = np.array(slide_img.read_region((0, 0), 0, size=(10, 10)))
    nb_channels = min(3, im.shape[2]) if im.ndim == 3 else 1
    img_size = list(tile_size * level_downsample)[::-1] + [nb_channels]
    image = np.zeros(img_size, dtype=np.uint8)
    for loc_i, loc_j in tqdm.tqdm(locations, desc=os.path.basename(path_img)):
        loc_img = int(loc_i * level_scale), int(loc_j * level_scale)
        img = np.array(slide_img.read_region(loc_img, level, size=tile_size))
        image[loc_j:loc_j + img.shape[0],
              loc_i:loc_i + img.shape[1], ...] = img[:, :, :nb_channels]
        del img

    if nb_channels == 2:
        image = image[:, :, 0]

    logging.debug('save image: "%s"', path_img_new)
    cv.imwrite(path_img_new, image, params=(cv.IMWRITE_PNG_COMPRESSION, 9))
    gc.collect()
    time.sleep(1)


def main(path_images, level=DEFAULT_LEVEL, overwrite=False, nb_workers=1):
    """ main entry point

    :param str path_images: path to images
    :param int level: selected level of the internal pyramid representation
        the level 0 means full scale and higher number is small image in pyramid scaling
    :param bool overwrite: whether overwrite existing image on output
    :param int nb_workers: nb jobs running in parallel
    """
    paths_img = sorted(glob.glob(path_images))

    _wrap_convert = partial(convert_image,
                            level=level,
                            overwrite=overwrite)

    list(iterate_mproc_map(_wrap_convert, paths_img, desc='Converting images',
                           nb_workers=nb_workers))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
