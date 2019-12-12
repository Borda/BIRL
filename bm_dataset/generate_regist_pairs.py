"""
Script for generating registration pairs in two schemas

Sample run::

    python generate_regist_pairs.py \
        -i "../output/synth_dataset/*.jpg" \
        -l "../output/synth_dataset/*.csv" \
        -csv ../output/cover.csv --mode each2all

    python bm_dataset/generate_regist_pairs.py \
        -i "~/Medical-data/dataset_CIMA/lung-lesion_1/scale-100pc/*.png" \
        -l "~/Medical-data/dataset_CIMA/lung-lesion_1/scale-100pc/*.csv" \
        -csv ~/Medical-data/dataset_CIMA/dataset_CIMA_100pc.csv --mode each2all

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import argparse
import logging

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import image_sizes
from birl.utilities.experiments import parse_arg_params
from birl.benchmark import ImRegBenchmark

# list of combination options
OPTIONS_COMBINE = ('first2all', 'each2all')


def arg_parse_params():
    """ parse the input parameters

    :return dict: parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_pattern_images', type=str,
                        help='path to the input image', required=True)
    parser.add_argument('-l', '--path_pattern_landmarks', type=str,
                        help='path to the input landmarks', required=True)
    parser.add_argument('-csv', '--path_csv', type=str, required=True,
                        help='path to coordinate csv file')
    parser.add_argument('--mode', type=str, required=False,
                        help='type of combination of registration pairs',
                        default=OPTIONS_COMBINE[0], choices=OPTIONS_COMBINE)
    args = parse_arg_params(parser, upper_dirs=['path_csv'])
    return args


def generate_pairs(path_pattern_imgs, path_pattern_lnds, mode):
    """ generate the registration pairs as reference and moving images

    :param str path_pattern_imgs: path to the images and image name pattern
    :param str path_pattern_lnds: path to the landmarks and its name pattern
    :param str mode: one of OPTIONS_COMBINE
    :return: DF
    """
    list_imgs = sorted(glob.glob(path_pattern_imgs))
    list_lnds = sorted(glob.glob(path_pattern_lnds))
    assert len(list_imgs) == len(list_lnds), \
        'the list of loaded images (%i) and landmarks (%i) is different length' \
        % (len(list_imgs), len(list_lnds))
    assert len(list_imgs) >= 2, 'the minimum is 2 elements'
    logging.info('combining list %i files with "%s"', len(list_imgs), mode)

    pairs = [(0, i) for i in range(1, len(list_imgs))]
    if mode == 'each2all':
        pairs += [(i, j) for i in range(1, len(list_imgs))
                  for j in range(i + 1, len(list_imgs))]

    reg_pairs = []
    for i, j in pairs:
        rec = dict(zip(ImRegBenchmark.COVER_COLUMNS,
                       (list_imgs[i], list_imgs[j], list_lnds[i], list_lnds[j])))
        img_size, img_diag = image_sizes(rec[ImRegBenchmark.COL_IMAGE_REF])
        rec.update({
            ImRegBenchmark.COL_IMAGE_SIZE: img_size,
            ImRegBenchmark.COL_IMAGE_DIAGONAL: img_diag,
        })
        reg_pairs.append(rec)

    df_overview = pd.DataFrame(reg_pairs)
    return df_overview


def main(path_pattern_images, path_pattern_landmarks, path_csv, mode='all2all'):
    """ main entry point

    :param str path_pattern_images: path to images
    :param str path_pattern_landmarks: path to landmarks
    :param str path_csv: path output cover table, add new rows if it exists
    :param str mode: option first2all or all2all
    """
    # if the cover file exist continue in it, otherwise create new
    if os.path.isfile(path_csv):
        logging.info('loading existing csv file: %s', path_csv)
        df_overview = pd.read_csv(path_csv, index_col=0)
    else:
        logging.info('creating new cover file')
        df_overview = pd.DataFrame()

    df_ = generate_pairs(path_pattern_images, path_pattern_landmarks, mode)
    df_overview = pd.concat((df_overview, df_), axis=0)  # , sort=True
    df_overview = df_overview[list(ImRegBenchmark.COVER_COLUMNS_EXT)].reset_index(drop=True)

    logging.info('saving csv file with %i records \n %s', len(df_overview), path_csv)
    df_overview.to_csv(path_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
