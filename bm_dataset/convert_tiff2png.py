"""
Converting TIFF, SVS images to PNG

Convert the original TIFF image containing a pyramid to single image
of a particular level (regarding the pyramid) using mosaic techniques

Be aware that this conversion takes lots of memory, for example
image of size 50k x 60k takes about 10GB in RAM

EXAMPLE
-------
>> python convert_tiff2png.py -l 0 --nb_jobs 2 \
    -i "/datagrid/Medical/dataset_ANHIR/images_raw/*/*.tiff"
>> python convert_tiff2png.py -l 1 --nb_jobs 5 --overwrite \
    -i "/datagrid/Medical/dataset_ANHIR/images_raw/*/*.svs"


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

import tqdm
import cv2 as cv
import numpy as np
from openslide import OpenSlide

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.experiments import wrap_execute_sequence

DEFAULT_LEVEL = 1
MAX_LOAD_IMAGE_SIZE = 16000
IMAGE_EXTENSION = '.png'
NB_THREADS = int(mproc.cpu_count() * .5)


def arg_parse_params():
    """ parse the input parameters
    :return dict: {str: any}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_images', type=str, required=True,
                        help='path (pattern) to the input image')
    parser.add_argument('-l', '--level', type=int, required=False,
                        help='list of output scales', default=DEFAULT_LEVEL)
    parser.add_argument('--overwrite', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_images'] = os.path.expanduser(args['path_images'])
    logging.info('ARGUMENTS: \n%s' % repr(args))
    return args


def convert_image(path_img, level=DEFAULT_LEVEL, overwrite=False):
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


def main(path_images, level=DEFAULT_LEVEL, overwrite=False, nb_jobs=1):
    paths_img = sorted(glob.glob(path_images))

    _wrap_convert = partial(convert_image,
                            level=level,
                            overwrite=overwrite)

    list(wrap_execute_sequence(_wrap_convert, paths_img,
                               desc='converting images', nb_jobs=nb_jobs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    main(arg_params['path_images'], arg_params['level'],
         arg_params['overwrite'], arg_params['nb_jobs'])

    logging.info('DONE')
