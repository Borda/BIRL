""" Zip scaled datasets

The paths and all other constants are set to run on CMP grid for ANHIR dataset
zip only images mentioned in cover file and landmarks from source

>> python bm_ANHIR/zip_dataset_by_cover.py \
    -i /datagrid/Medical/dataset_ANHIR/images_private \
    -l /datagrid/Medical/dataset_ANHIR/landmarks \
    -la /datagrid/Medical/dataset_ANHIR/landmarks_all \
    -csv /datagrid/Medical/dataset_ANHIR/images/dataset_medium.csv

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import subprocess

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.cls_benchmark import COL_IMAGE_REF, COL_IMAGE_MOVE, COL_POINTS_MOVE

ZIP_COMMAND = 'cd %s && zip --split-size 1g %s.zip -r %s'


def arg_parse_params():
    """ parse the input parameters

    :return dict: {str: str}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_dataset', type=str,
                        help='path to the input image', required=True)
    parser.add_argument('-l', '--path_landmarks', type=str,
                        help='path to the input landmarks', required=True)
    parser.add_argument('-la', '--path_landmarks_all', type=str,
                        help='path to the all landmarks', required=True)
    parser.add_argument('-csv', '--path_csv', type=str, required=True,
                        help='path to coordinate csv file')
    args = vars(parser.parse_args())
    return args


def _process_cmd(command):
    logging.info(command)
    subprocess.call(command, shell=True)


def main(path_dataset, path_landmarks, path_landmarks_all, path_csv):
    name_csv = os.path.splitext(os.path.basename(path_csv))[0]
    df_cover = pd.read_csv(path_csv)

    images = df_cover[COL_IMAGE_REF].tolist() + df_cover[COL_IMAGE_MOVE].tolist()
    folders = set(os.path.dirname(p) for p in images
                  if os.path.isdir(os.path.join(path_dataset, os.path.dirname(p))))
    # Remove previous compressed images
    cmd_remove = 'rm -f %s' % os.path.join(path_dataset, name_csv + '.z*')
    _process_cmd(cmd_remove)
    # compress the images
    cmd_zip_imgs = ZIP_COMMAND % (path_dataset, name_csv, ' '.join(folders))
    _process_cmd(cmd_zip_imgs)

    landmarks = set(df_cover[COL_POINTS_MOVE].tolist())
    landmarks = [p for p in landmarks
                 if os.path.isfile(os.path.join(path_landmarks_all, p))]
    # compress the landmarks
    cmd_zip_lnds = ZIP_COMMAND % (path_landmarks_all, name_csv, ' '.join(landmarks))
    _process_cmd(cmd_zip_lnds)
    # Move the compressed landmarks
    cmd_move = 'mv %s %s' % (os.path.join(path_landmarks_all, name_csv + '.zip'),
                             os.path.join(path_landmarks, name_csv + '.zip'))
    _process_cmd(cmd_move)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_params = arg_parse_params()
    main(**arg_params)
