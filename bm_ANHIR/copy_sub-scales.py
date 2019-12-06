"""
Copy files as all sub-scales

.. note:: all paths are hard-coded to be used in CMP grid

"""

import os
import sys
import shutil
import logging

import tqdm
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.benchmark import ImRegBenchmark
from birl.utilities.dataset import parse_path_scale
from bm_ANHIR.generate_regist_pairs import VAL_STATUS_TRAIN

PATH_CSV = '/datagrid/Medical/dataset_ANHIR/images/dataset_medium.csv'
PATH_IMAGES = '/datagrid/Medical/dataset_ANHIR/images'
PATH_IMAGES_ALL = '/datagrid/Medical/dataset_ANHIR/images_private'
PATH_LANDMARKS = '/datagrid/Medical/dataset_ANHIR/landmarks'
PATH_LANDMARKS_ALL = '/datagrid/Medical/dataset_ANHIR/landmarks_user'
SCALES = (2, 5, 10, 15, 20, 25, 50, 100)
FOLDER_NAME = 'scale-%ipc'
FORCE_COPY = False


def main(path_csv, path_in, path_out, col_name, train_only=True):
    """ main entry point

    :param str path_csv: path to dataset cover
    :param str path_in: path to input images
    :param str path_out: path to output images
    :param str col_name: column from the cover table
    :param bool train_only: use only training cases
    """
    df = pd.read_csv(path_csv)
    if train_only:
        df = df[df[ImRegBenchmark.COL_STATUS] == VAL_STATUS_TRAIN]
    files = df[col_name]

    for p_file in tqdm.tqdm(files, desc=col_name):
        scale = parse_path_scale(os.path.dirname(p_file))
        # print(scale, SCALES[:SCALES.index(scale)])
        tissue_name = p_file.split(os.path.sep)[0]
        case_name = os.path.basename(p_file)
        for sc in SCALES[:SCALES.index(scale) + 1]:
            path_file = os.path.join(tissue_name, FOLDER_NAME % sc, case_name)
            path_dir = os.path.join(path_out, tissue_name, FOLDER_NAME % sc)
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir)
            path_src = os.path.join(path_in, path_file)
            path_dst = os.path.join(path_out, path_file)
            if not os.path.isfile(path_src):
                logging.debug('missing source file: %s', path_src)
                continue
            # print(path_src, path_dst)
            if not os.path.isfile(path_dst) or FORCE_COPY:
                shutil.copy(path_src, path_dst)
            elif os.path.isfile(path_dst) and not FORCE_COPY:
                logging.debug('existing target file: %s', path_dst)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')
    main(PATH_CSV, PATH_LANDMARKS_ALL, PATH_LANDMARKS,
         ImRegBenchmark.COL_POINTS_REF, train_only=True)
    main(PATH_CSV, PATH_LANDMARKS_ALL, PATH_LANDMARKS,
         ImRegBenchmark.COL_POINTS_MOVE, train_only=False)
    main(PATH_CSV, PATH_IMAGES_ALL, PATH_IMAGES,
         ImRegBenchmark.COL_IMAGE_REF, train_only=False)
    main(PATH_CSV, PATH_IMAGES_ALL, PATH_IMAGES,
         ImRegBenchmark.COL_IMAGE_MOVE, train_only=False)
    logging.info('Done >]')
