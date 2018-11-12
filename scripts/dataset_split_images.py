"""
Spliting image containing two samples

Note, for the loading we have to use matplotlib while ImageMagic nor other lib
(opencv, skimage, Pillow) is able to load larger images then 32k.
On the other hand matplotlib is not able to save such large image.

EXAMPLE
-------
>> python dataset_split_images.py \
    -i "/datagrid/Medical/dataset_ANHIR/images/COAD_*/scale-100pc/*_*.png" \
    --nb_jobs 3

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

from skimage.transform import rescale

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.dataset import find_split_objects, project_object_edge
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
    parser.add_argument('--overwrite', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    args['path_images'] = os.path.expanduser(args['path_images'])
    logging.info('ARGUMENTS: \n%s' % repr(args))
    return args


def split_image(img_path, overwrite=False, cut_dimension=CUT_DIMENSION):
    name = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(os.path.basename(img_path))[-1]
    dir = os.path.dirname(img_path)
    obj_names = name.split('_')
    paths_img = [os.path.join(dir, obj_name + ext) for obj_name in obj_names]

    if all(os.path.isfile(p) for p in paths_img) and not overwrite:
        logging.debug('existing all splits of %s', repr(paths_img))
        return

    img = load_large_image(img_path)
    # work with just a scaled version
    img_small = 1. - rescale(img, 1. / SCALE_FACTOR, order=0, multichannel=True,
                             mode='constant', anti_aliasing=True,
                             preserve_range=True)
    img_edge = project_object_edge(img_small, cut_dimension)
    del img_small

    # prepare all cut edges and scale them to original image size
    splits = find_split_objects(img_edge, nb_objects=len(obj_names))
    if not splits:
        logging.error('no splits found for %s', img_path)
        return
    edges = [int(round(i * SCALE_FACTOR))
             for i in [0] + splits + [len(img_edge)]]

    # cutting images
    for i, path_img_cut in enumerate(paths_img):
        if os.path.isfile(path_img_cut) and not overwrite:
            logging.debug('existing "%s"', repr(paths_img))
            continue
        if cut_dimension == 0:
            img_cut = img[edges[i]:edges[i + 1], ...]
        elif cut_dimension == 1:
            img_cut = img[:, edges[i]:edges[i + 1], ...]
        else:
            raise Exception('unsuposted dimensio: %i' % cut_dimension)
        save_large_image(path_img_cut, img_cut)
        gc.collect(), time.sleep(1)



def main(path_images, overwrite, nb_jobs):
    image_paths = sorted(glob.glob(path_images))

    _wrap_split = partial(split_image, overwrite=overwrite)
    list(wrap_execute_sequence(_wrap_split, image_paths,
                               desc='Cut image objects', nb_jobs=nb_jobs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    main(arg_params['path_images'],
         arg_params['overwrite'], arg_params['nb_jobs'])

    logging.info('DONE')
