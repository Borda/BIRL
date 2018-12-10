"""
Creating cover file for configuring registration image pairs
The paths and all other constants are set to run on CMP grid for ANHIR dataset

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.dataset import IMAGE_EXTENSIONS, generate_pairing
from benchmark.cls_benchmark import (COL_IMAGE_REF, COL_IMAGE_MOVE,
                                     COL_POINTS_REF, COL_POINTS_MOVE,
                                     COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP)

DATASET_IMAGES = '/datagrid/Medical/dataset_ANHIR/images'
DATASET_LANDMARKS = '/datagrid/Medical/dataset_ANHIR/landmarks_all'
DATASET_COVERS = '/datagrid/Medical/dataset_ANHIR/images'
NAME_COVER_FILE = 'dataset_%s.csv'
GENERATED_SCALES = (5, 10, 25, 50, 100)
NAME_DIR_SCALE = 'scale-%ipc'
# define datasets scale size names and the shift in GENERATED_SCALES
SCALE_NAMES = (
    'small',  # just thumbnail about 2500 image edge
    'medium',  # the image edge should have abound 10k
)
# define tissues with all landmarks presented
DATASET_TISSUE_SCALE_COMPLETE = {
    'lung-lesion_[1,3]': (5, 50),
    'lung-lesion_2': (5, 25),
    'lung-lobes_*': (5, 50),
    'mammary-gland_*': (5, 25),
}
# define tissues which hide some samples as test
DATASET_TISSUE_SCALE_PARTIAL = {
    'kidney_*': (5, 25),
    'COAD_*': (2, 25),
    'gastric_*': (2, 20),
}
# define tissues to be part of the dataset
DATASET_TISSUE_SCALE = DATASET_TISSUE_SCALE_COMPLETE.copy()
DATASET_TISSUE_SCALE.update(DATASET_TISSUE_SCALE_PARTIAL)
# each N sample in test will be considers as test case
HIDE_TEST_TISSUE_STEP = 3
# requires empty columns in the dataset cover
COLUMNS_EMPTY = (
    COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP,
    'Execution time [minutes]'
)


def get_relative_paths(paths, path_base):
    paths_r = [p.replace(path_base, '')[1:] for p in sorted(paths)]
    return paths_r


def list_landmarks_images(path_tissue, sc, path_landmarks, path_images):
    path_ = os.path.join(path_tissue, NAME_DIR_SCALE % sc, '*.csv')
    rp_lnds = get_relative_paths(glob.glob(path_), path_landmarks)
    if not rp_lnds:
        logging.debug('found no landmarks for: %s', path_)
        return [], []
    paths_imgs = []
    for rp_lnd in rp_lnds:
        p_imgs = glob.glob(os.path.join(path_images,
                                        os.path.splitext(rp_lnd)[0] + '.*'))
        p_imgs = [p for p in p_imgs
                  if os.path.splitext(p)[-1] in IMAGE_EXTENSIONS]
        if not len(p_imgs):
            logging.error('missing image for "%s"', rp_lnd)
            return [], []
        paths_imgs.append(sorted(p_imgs)[0])
    rp_imgs = get_relative_paths(paths_imgs, path_images)
    return rp_lnds, rp_imgs


def generate_reg_pairs(rp_imgs, rp_lnds, pairs):
    reg_pairs = []
    for i, j in pairs:
        reg_pairs.append({
            COL_IMAGE_REF: rp_imgs[i],
            COL_IMAGE_MOVE: rp_imgs[j],
            COL_POINTS_REF: rp_lnds[i],
            COL_POINTS_MOVE: rp_lnds[j],
        })
    return reg_pairs


def create_dataset_cover(name, dataset, path_images, path_landmarks, path_out,
                         step_hide_landmarks, tissue_partial):
    """ geberate cover CSV file for particular dataset size/scale

    :param str name:
    :param {} dataset:
    :param str path_images:
    :param str path_landmarks:
    :param str path_out:
    :param int step_hide_landmarks:
    :param [str] tissue_partial:
    :return:
    """
    # name, scale_step = dataset

    reg_pairs = []
    for tissue in sorted(dataset):
        sc = dataset[tissue][SCALE_NAMES.index(name)]
        paths_tissue = [p for p in glob.glob(os.path.join(path_landmarks, tissue))
                        if os.path.isdir(p)]
        for p_tissue in sorted(paths_tissue):
            rp_lnds, rp_imgs = list_landmarks_images(p_tissue, sc,
                                                     path_landmarks, path_images)
            assert len(rp_lnds) == len(rp_imgs), \
                'the list of landmarks and images does not match'
            step_hide_landmarks = step_hide_landmarks if tissue in tissue_partial else None
            pairs = generate_pairing(len(rp_lnds), step_hide_landmarks)
            reg_pairs += generate_reg_pairs(rp_imgs, rp_lnds, pairs)

    df_cover = pd.DataFrame(reg_pairs)
    for col in COLUMNS_EMPTY:
        df_cover[col] = None
    path_csv = os.path.join(path_out, NAME_COVER_FILE % name)
    logging.info('exporting CSV: %s', path_csv)
    df_cover.to_csv(path_csv)


def main(path_images, path_landmarks, path_out, step_lnds, dataset,
         tissue_partial, scale_names):

    _create_cover = partial(create_dataset_cover,
                            dataset=dataset,
                            path_images=path_images,
                            path_landmarks=path_landmarks,
                            path_out=path_out,
                            step_hide_landmarks=step_lnds,
                            tissue_partial=tissue_partial)

    for name in scale_names:
        _create_cover(name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')
    main(path_images=DATASET_IMAGES, path_landmarks=DATASET_LANDMARKS,
         path_out=DATASET_COVERS, step_lnds=HIDE_TEST_TISSUE_STEP,
         dataset=DATASET_TISSUE_SCALE, scale_names=SCALE_NAMES,
         tissue_partial=DATASET_TISSUE_SCALE_PARTIAL.keys())
    logging.info('Done :]')
