"""
Copy files as all sub-scales

"""

import os
import sys
import shutil
import logging

import tqdm
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.cls_benchmark import COL_IMAGE_REF, COL_IMAGE_MOVE, COL_POINTS_MOVE
from birl.utilities.dataset import parse_path_scale

PATH_CSV = '/datagrid/Medical/dataset_ANHIR/images/dataset_medium.csv'
PATH_IMAGES = '/datagrid/Medical/dataset_ANHIR/images'
PATH_IMAGES_ALL = '/datagrid/Medical/dataset_ANHIR/images_private'
PATH_LANDMARKS = '/datagrid/Medical/dataset_ANHIR/landmarks'
PATH_LANDMARKS_ALL = '/datagrid/Medical/dataset_ANHIR/landmarks_user'
SCALES = (2, 5, 10, 15, 20, 25, 50, 100)
FOLDER_NAME = 'scale-%ipc'
FORCE_COPY = False


def main(path_csv, path_in, path_out, col_name):
    df = pd.read_csv(path_csv)
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

    main(PATH_CSV, PATH_LANDMARKS_ALL, PATH_LANDMARKS, COL_POINTS_MOVE)
    main(PATH_CSV, PATH_IMAGES_ALL, PATH_IMAGES, COL_IMAGE_REF)
    main(PATH_CSV, PATH_IMAGES_ALL, PATH_IMAGES, COL_IMAGE_MOVE)
