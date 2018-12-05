"""
Simple benchmarks measuring basic computer performances

"""

import os
import time
import random
import json
import shutil
import multiprocessing as mproc

import matplotlib
# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')

import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt

NB_THREADS = mproc.cpu_count()
NAME_REPORT = 'computer-performances.json'


def wrap(func, nb_iter, *args, **kwargs):
    start = time.time()
    for _ in tqdm.tqdm(range(int(nb_iter)), desc=func.__name__):
        func(*args, **kwargs)
    res = {func.__name__: (time.time() - start) / nb_iter}
    return res


def timeit_long(func):
    """ costume decorator to measure execution time
    :param func:
    :return float:
    """
    return wrap(func, nb_iter=1e5)


def timeit_short(func):
    """ costume decorator to measure execution time
    :param func:
    :return float:
    """
    return wrap(func, nb_iter=25)


@timeit_long
def numeric_operations(num=None):
    num = random.random() if num is None else num
    num = 1. if num == 0 else num
    num = float(num)
    math.sqrt(abs(num + num * num - num / num))
    num = int(num * 1000)
    num = 1 if num == 0 else num
    return math.sqrt(abs(num + num * num - num / num))


@timeit_long
def matrix_operations():
    mx = np.random.random((100, 200)).astype(np.float32)
    np.sqrt(np.abs(mx + mx * mx - mx / mx))
    mx = (mx * 1000).astype(np.int32)
    mx[mx == 0] = 1
    return np.sqrt(np.abs(mx + mx * mx - mx / mx))


@timeit_long
def list_index(length=1000):
    lst = list(range(length)) + list(range(length, length * 2))
    return lst.index(length)


def wrap_fn(num):
    return num + num * num


@timeit_short
def parallel_comp(nb_jobs=NB_THREADS):
    pool = mproc.Pool(nb_jobs)
    list(pool.map(wrap_fn, range(int(nb_jobs * 10))))
    pool.close()
    pool.join()


def io_image(im_shape=(100, 150)):
    img = np.random.random(list(im_shape) + [3])
    img_path = 'testing-image-BM.png'
    plt.imsave(img_path, img)
    plt.imread(img_path)
    os.remove(img_path)


@timeit_short
def io_images(scales=2):
    sc = np.array([600, 800])
    for _ in range(scales):
        sc = sc * 2
        io_image(sc)


def main(path_out=''):
    report = {'computer': os.uname(),
              'nb. threads': mproc.cpu_count()}
    report.update(list_index)
    report.update(numeric_operations)
    report.update(matrix_operations)
    # report.update(parallel_comp)
    # report.update(io_images)

    path_json = os.path.join(path_out, NAME_REPORT)
    with open(path_json, 'w') as fp:
        json.dump(report, fp)


if __name__ == '__main__':
    main()
