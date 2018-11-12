"""
Converting TIFF images to PNG

Convert the original TIFF image containing a pyramid to single image
of a particular level (regarding the pyramid) using mosaic techniques

Be aware that this conversion takes lots of memory, for example
image of size 50k x 100k takes about 80GB in RAM

EXAMPLE
-------
>> python convert_tiff2png.py \
    -i "/datagrid/Medical/microscopy/histology/*/*.tiff" \
    -l 0 --level_offset 2 --nb_jobs 2

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
import cv2
import numpy as np
from openslide import OpenSlide

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.experiments import wrap_execute_sequence

DEFAULT_LEVEL = 1
DEFAULT_LEVEL_OFFSET = 2
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
    parser.add_argument('--level_offset', type=int, required=False,
                        help='list of output scales', default=DEFAULT_LEVEL_OFFSET)
    parser.add_argument('--overwrite', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_images'] = os.path.expanduser(args['path_images'])
    logging.info('ARGUMENTS: \n%s' % repr(args))
    return args


def convert_image(path_img, level=DEFAULT_LEVEL, level_shift=DEFAULT_LEVEL_OFFSET, overwrite=False):
    slide_img = OpenSlide(path_img)
    assert level < len(slide_img.level_dimensions), \
        'unsupported level %i' % level
    assert (level + level_shift) < len(slide_img.level_dimensions), \
        'unsupported shift %i for level %i' % (level_shift, level)

    path_img_new = os.path.splitext(path_img)[0] + IMAGE_EXTENSION
    if os.path.isfile(path_img_new) and not overwrite:
        logging.debug('existing "%s"', path_img_new)
        return

    tile_size = slide_img.level_dimensions[level + level_shift]
    img_tiles_d0 = []
    tqdm_bar = tqdm.tqdm(total=(2 ** level_shift) ** 2,
                         desc=os.path.basename(path_img))
    for i in range(2 ** level_shift):
        img_tiles_d1 = []
        for j in range(2 ** level_shift):
            loc = (i * tile_size[0] * 2 ** level,
                   j * tile_size[1] * 2 ** level)
            img = slide_img.read_region(loc, level, size=tile_size)
            img_tiles_d1.append(img)
            tqdm_bar.update()
        img_tiles_d0.append(np.vstack(img_tiles_d1))
    image = np.hstack(img_tiles_d0)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    logging.debug('save image: "%s"', path_img_new)
    cv2.imwrite(path_img_new, image)
    gc.collect(), time.sleep(1)


def main(path_images, level=DEFAULT_LEVEL, level_shift=DEFAULT_LEVEL_OFFSET,
         overwrite=False, nb_jobs=1):
    paths_img = sorted(glob.glob(path_images))

    _wrap_convert = partial(convert_image,
                            level=level,
                            level_shift=level_shift,
                            overwrite=overwrite)

    list(wrap_execute_sequence(_wrap_convert, paths_img,
                               desc='converting images', nb_jobs=nb_jobs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    arg_params = arg_parse_params()
    main(arg_params['path_images'], arg_params['level'], arg_params['level_offset'],
         arg_params['overwrite'], arg_params['nb_jobs'])

    logging.info('DONE')
