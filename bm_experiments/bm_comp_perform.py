"""
Simple benchmarks measuring basic computer performances

We run image registration in single thread and then in all available thread
in parallel and measure the execution time.

The tested image registration scenario is as following

 1. load both images
 2. perform som simple denoising
 3. extract ORB features
 4. estimate affine transform via RANSAC
 5. warp and export image

Example run::

    pip install --user tqdm numpy scikit-image https://github.com/Borda/BIRL/archive/master.zip
    python bm_comp_perform.py -o ../output -n 3

Copyright (C) 2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import json
import argparse
import datetime
import logging
import platform
import hashlib
import multiprocessing as mproc
import traceback
import warnings
from functools import partial

import tqdm
import matplotlib
matplotlib.use('Agg')
import numpy as np
from skimage import data, io, __version__ as skimage__version
from skimage.transform import resize, warp, AffineTransform
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.util import random_noise
from skimage.restoration import denoise_bilateral, denoise_wavelet
from skimage.feature import ORB, match_descriptors

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import computer_info

IMAGE_SIZE = (2000, 2000)
IMAGE_NOISE = 0.01
SKIMAGE_VERSION = (0, 14, 0)

CPU_COUNT = int(mproc.cpu_count())
NAME_REPORT = 'computer-performances.json'
NAME_IMAGE_TARGET = 'temp_regist-image_target.png'
NAME_IMAGE_SOURCE = 'temp_regist-image_source.png'
NAME_IMAGE_WARPED = 'temp_regist-image_warped-%i.jpg'


def arg_parse_params():
    """ parse the input parameters
    :return dict: parameters
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--path_out', type=str, required=False,
                        help='path to the output folder', default='')
    parser.add_argument('-n', '--nb_runs', type=int, required=False,
                        help='number of run experiments', default=5)
    args = vars(parser.parse_args())
    logging.info('ARGUMENTS: \n%r' % args)
    return args


def _prepare_images(path_out, im_size=IMAGE_SIZE):
    """ generate and prepare synth. images for registration

    :param str path_out: path to the folder
    :param tuple(int,int) im_size: desired image size
    :return tuple(str,str): paths to target and source image
    """
    image = resize(data.astronaut(), output_shape=im_size, mode='constant')
    img_target = random_noise(image, var=IMAGE_NOISE)
    path_img_target = os.path.join(path_out, NAME_IMAGE_TARGET)
    io.imsave(path_img_target, img_target)

    # warp synthetic image
    tform = AffineTransform(scale=(0.9, 0.9),
                            rotation=0.2,
                            translation=(200, -50))
    img_source = warp(image, tform.inverse, output_shape=im_size)
    img_source = random_noise(img_source, var=IMAGE_NOISE)
    path_img_source = os.path.join(path_out, NAME_IMAGE_SOURCE)
    io.imsave(path_img_source, img_source)
    return path_img_target, path_img_source


def _clean_images(image_paths):
    """ remove temporary images

    :param str image_paths: path to images
    """
    for p_img in image_paths:
        os.remove(p_img)


def register_image_pair(idx, path_img_target, path_img_source, path_out):
    """ register two images together

    :param int idx: empty parameter for using the function in parallel
    :param str path_img_target: path to the target image
    :param str path_img_source: path to the source image
    :param str path_out: path for exporting the output
    :return tuple(str,float):
    """
    start = time.time()
    # load and denoise reference image
    img_target = io.imread(path_img_target)[..., :3]
    img_target = denoise_wavelet(img_target, wavelet_levels=7, multichannel=True)
    img_target_gray = rgb2gray(img_target)

    # load and denoise moving image
    img_source = io.imread(path_img_source)[..., :3]
    img_source = denoise_bilateral(img_source, sigma_color=0.05,
                                   sigma_spatial=2, multichannel=True)
    img_source_gray = rgb2gray(img_source)

    # detect ORB features on both images
    detector_target = ORB(n_keypoints=150)
    detector_source = ORB(n_keypoints=150)
    detector_target.detect_and_extract(img_target_gray)
    detector_source.detect_and_extract(img_source_gray)
    matches = match_descriptors(detector_target.descriptors,
                                detector_source.descriptors)
    # robustly estimate affine transform model with RANSAC
    model, _ = ransac((detector_target.keypoints[matches[:, 0]],
                       detector_source.keypoints[matches[:, 1]]),
                      AffineTransform, min_samples=25, max_trials=500,
                      residual_threshold=0.95)

    # warping source image with estimated transformations
    img_warped = warp(img_target, model.inverse, output_shape=img_target.shape[:2])
    path_img_warped = os.path.join(path_out, NAME_IMAGE_WARPED % idx)
    try:
        io.imsave(path_img_warped, img_warped)
    except Exception:
        traceback.print_exc()
    # summarise experiment
    execution_time = time.time() - start
    return path_img_warped, execution_time


def measure_registration_single(path_out, nb_iter=5):
    """ measure mean execration time for image registration running in 1 thread

    :param str path_out: path to the temporary output space
    :param int nb_iter: number of experiments to be averaged
    :return dict: dictionary of float values results
    """
    path_img_target, path_img_source = _prepare_images(path_out, IMAGE_SIZE)
    paths = [path_img_target, path_img_source]

    execution_times = []
    for i in tqdm.tqdm(range(nb_iter), desc='using single-thread'):
        path_img_warped, t = register_image_pair(i, path_img_target,
                                                 path_img_source,
                                                 path_out)
        paths.append(path_img_warped)
        execution_times.append(t)

    _clean_images(set(paths))
    logging.info('registration @1-thread: %f +/- %f',
                 np.mean(execution_times), np.std(execution_times))
    res = {'registration @1-thread': np.mean(execution_times)}
    return res


def measure_registration_parallel(path_out, nb_iter=3, nb_workers=CPU_COUNT):
    """ measure mean execration time for image registration running in N thread

    :param str path_out: path to the temporary output space
    :param int nb_iter: number of experiments to be averaged
    :param int nb_workers: number of thread available on the computer
    :return dict: dictionary of float values results
    """
    path_img_target, path_img_source = _prepare_images(path_out, IMAGE_SIZE)
    paths = [path_img_target, path_img_source]
    execution_times = []

    _regist = partial(register_image_pair, path_img_target=path_img_target,
                      path_img_source=path_img_source, path_out=path_out)
    nb_tasks = int(nb_workers * nb_iter)
    logging.info('>> running %i tasks in %i threads', nb_tasks, nb_workers)
    tqdm_bar = tqdm.tqdm(total=nb_tasks, desc='parallel @ %i threads' % nb_workers)

    pool = mproc.Pool(nb_workers)
    for path_img_warped, t in pool.map(_regist, (range(nb_tasks))):
        paths.append(path_img_warped)
        execution_times.append(t)
        tqdm_bar.update()
    pool.close()
    pool.join()
    tqdm_bar.close()

    _clean_images(set(paths))
    logging.info('registration @%i-thread: %f +/- %f', nb_workers,
                 np.mean(execution_times), np.std(execution_times))
    res = {'registration @n-thread': np.mean(execution_times)}
    return res


def main(path_out='', nb_runs=5):
    """ the main entry point

    :param str path_out: path to export the report and save temporal images
    :param int nb_runs: number of trails to have an robust average value
    """
    logging.info('Running the computer benchmark.')
    skimage_ver = skimage__version.split('.')
    skimage_ver = tuple(map(int, skimage_ver))
    if skimage_ver < SKIMAGE_VERSION:
        logging.warning('You are using older version of scikit-image then we expect.'
                        ' Please upadte by `pip install -U --user scikit-image>=%s`',
                        '.'.join(map(str, SKIMAGE_VERSION)))

    hasher = hashlib.sha256()
    hasher.update(open(__file__, 'rb').read())
    report = {
        'computer': computer_info(),
        'created': str(datetime.datetime.now()),
        'file': hasher.hexdigest(),
        'number runs': nb_runs,
        'python-version': platform.python_version(),
        'skimage-version': skimage__version,
    }
    report.update(measure_registration_single(path_out, nb_iter=nb_runs))
    nb_runs_ = max(1, int(nb_runs / 2.))
    report.update(measure_registration_parallel(path_out, nb_iter=nb_runs_))

    path_json = os.path.join(path_out, NAME_REPORT)
    logging.info('exporting report: %s', path_json)
    with open(path_json, 'w') as fp:
        json.dump(report, fp)
    logging.info('\n\t '.join('%s: \t %r' % (k, report[k]) for k in report))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = arg_parse_params()
    logging.info('running...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(**arg_params)
    logging.info('Done :]')
