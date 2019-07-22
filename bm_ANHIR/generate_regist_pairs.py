"""
Creating cover file for configuring registration image pairs for ANHIR dataset.
The paths and all other constants are set to run on CMP grid.

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
from functools import partial

import tqdm
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import image_sizes, update_path
from birl.utilities.dataset import IMAGE_EXTENSIONS, generate_pairing
from birl.benchmark import ImRegBenchmark

DATASET_IMAGES = '/datagrid/Medical/dataset_ANHIR/images_private'
DATASET_LANDMARKS = '/datagrid/Medical/dataset_ANHIR/landmarks_all'
DATASET_COVERS = '/datagrid/Medical/dataset_ANHIR/images'
NAME_COVER_FILE = 'dataset_%s.csv'
GENERATED_SCALES = (5, 10, 15, 20, 25, 50, 100)
NAME_DIR_SCALE = 'scale-%ipc'
# define datasets scale size names and the shift in GENERATED_SCALES
SCALE_NAMES = (
    'small',  # just thumbnail about 2500 image edge
    'medium',  # the image edge should have abound 10k
)
# define tissues with all landmarks presented
DATASET_TISSUE_SCALE_COMPLETE = {
    'lung-lesion_[1,3]': {'small': 5, 'medium': 50},
    'lung-lesion_2': {'small': 5, 'medium': 25},
    'lung-lobes_*': {'small': 5, 'medium': 100},
    'mammary-gland_*': {'small': 5, 'medium': 25},
}
# define tissues which hide some samples as test
DATASET_TISSUE_SCALE_PARTIAL = {
    'mice-kidney_*': {'small': 5, 'medium': 25},
    'COAD_*': {'small': 5, 'medium': 25},
    'gastric_*': {'small': 2, 'medium': 15},
    'breast_*': {'small': 2, 'medium': 20},
    'kidney_*': {'small': 5, 'medium': 25},
}
# define tissues to be part of the dataset
DATASET_TISSUE_SCALE = DATASET_TISSUE_SCALE_COMPLETE.copy()
DATASET_TISSUE_SCALE.update(DATASET_TISSUE_SCALE_PARTIAL)
# each N sample in test will be considers as test case
HIDE_TEST_TISSUE_STEP = 3
# requires empty columns in the dataset cover
COLUMNS_EMPTY = (ImRegBenchmark.COL_POINTS_REF_WARP,
                 ImRegBenchmark.COL_POINTS_MOVE_WARP,
                 ImRegBenchmark.COL_TIME)
# define train / test status
VAL_STATUS_TRAIN = 'training'
VAL_STATUS_TEST = 'evaluation'


def get_relative_paths(paths, path_base):
    """ transform paths to relati according given base path

    :param list(str) paths: collection of paths
    :param str path_base: past that can be removed from the input paths
    :return str:
    """
    paths_r = [p.replace(path_base, '')[1:] for p in sorted(paths)]
    return paths_r


def list_landmarks_images(path_tissue, sc, path_landmarks, path_images):
    """ list image and landmarks paths

    :param str path_tissue: path to a tissue - image set
    :param int sc: used scale
    :param str path_landmarks:
    :param str path_images:
    :return tuple(list(str),list(str)):
    """
    path_ = os.path.join(path_tissue, NAME_DIR_SCALE % sc, '*.csv')
    rp_lnds = get_relative_paths(glob.glob(path_), path_landmarks)
    if not rp_lnds:
        logging.debug('found no landmarks for: %s', path_)
        return [], []
    paths_imgs, rp_lnds_filter = [], []
    for rp_lnd in rp_lnds:
        p_imgs = glob.glob(os.path.join(path_images,
                                        os.path.splitext(rp_lnd)[0] + '.*'))
        p_imgs = [p for p in p_imgs if os.path.splitext(p)[-1] in IMAGE_EXTENSIONS]
        if not p_imgs:
            logging.warning('missing image for "%s"', rp_lnd)
        else:
            rp_lnds_filter.append(rp_lnd)
            paths_imgs.append(sorted(p_imgs)[0])
    rp_imgs = get_relative_paths(paths_imgs, path_images)
    return rp_lnds_filter, rp_imgs


def generate_reg_pairs(rp_imgs, rp_lnds, pairs, public, path_images=DATASET_IMAGES):
    """ format a registration pair as dictionaries/rows in cover table for a set

    :param list(str) rp_imgs: relative paths to images
    :param rp_lnds: relative paths to related landmarks
    :param list(tuple(int,int)) pairs: pairing among images/landmarks
    :param list(bool) public: marks whether the particular pair is training or evaluation
    :param str path_images: path to the dataset folder
    :return list(dict): registration pairs
    """
    reg_pairs = []
    for k, (i, j) in enumerate(pairs):
        img_size, img_diag = image_sizes(update_path(rp_imgs[i], pre_path=path_images))
        reg_pairs.append({
            ImRegBenchmark.COL_IMAGE_REF: rp_imgs[i],
            ImRegBenchmark.COL_IMAGE_MOVE: rp_imgs[j],
            ImRegBenchmark.COL_POINTS_REF: rp_lnds[i],
            ImRegBenchmark.COL_POINTS_MOVE: rp_lnds[j],
            ImRegBenchmark.COL_STATUS: VAL_STATUS_TRAIN if public[k] else VAL_STATUS_TEST,
            ImRegBenchmark.COL_IMAGE_SIZE: img_size,
            ImRegBenchmark.COL_IMAGE_DIAGONAL: img_diag,
        })
    return reg_pairs


def create_dataset_cover(name, dataset, path_images, path_landmarks, path_out,
                         step_hide_landmarks, tissue_partial):
    """ generate cover CSV file for particular dataset size/scale

    :param str name: name of selected scale
    :param dict({scale_name: float}) dataset: definition of dataset
        fist level key is name of the tissue,
        next dictionary is composed from scale name and used scale in percents
    :param str path_images: path to folder with images
    :param str path_landmarks: path to folder with landmarks
    :param str path_out: path to output directory
    :param int step_hide_landmarks: take each N-th image/landmark out as a test case
    :param list(str) tissue_partial:
    """
    # name, scale_step = dataset
    tissues = [(tissue, p) for tissue in sorted(dataset)
               for p in glob.glob(os.path.join(path_landmarks, tissue))
               if os.path.isdir(p)]

    reg_pairs = []
    logging.debug('found: %r', sorted(set([os.path.basename(tp[1]) for tp in tissues])))
    for tissue, p_tissue in tqdm.tqdm(sorted(tissues)):
        sc = dataset[tissue][name]
        rp_lnds, rp_imgs = list_landmarks_images(p_tissue, sc, path_landmarks,
                                                 path_images)
        assert len(rp_lnds) == len(rp_imgs), \
            'the list of landmarks and images does not match'
        step_hide_lnds = step_hide_landmarks if tissue in tissue_partial else None
        pairs, pub = generate_pairing(len(rp_lnds), step_hide_lnds)
        reg_pairs += generate_reg_pairs(rp_imgs, rp_lnds, pairs, pub)

    df_overview = pd.DataFrame(reg_pairs)
    for col in COLUMNS_EMPTY:
        df_overview[col] = None
    path_csv = os.path.join(path_out, NAME_COVER_FILE % name)
    logging.info('exporting CSV: %s', path_csv)
    df_overview.to_csv(path_csv)


def main(path_images, path_landmarks, path_out, step_lnds, dataset,
         tissue_partial, scale_names):
    """ the main entry point

    :param str path_images: path to folder with images
    :param str path_landmarks: path to folder with landmarks
    :param str path_out: path to output directory
    :param int step_lnds: take each N-th image/landmark out as a test case
    :param dict({scale_name: float}) dataset: definition of dataset
        fist level key is name of the tissue,
        next dictionary is composed from scale name and used scale in percents
    :param list(str) tissue_partial: names of tissues which will have partially hidden cases
        also consider a testing tissues
    :param list(str) scale_names: name of chosen scales
    """

    _create_cover = partial(create_dataset_cover,
                            dataset=dataset,
                            path_images=path_images,
                            path_landmarks=path_landmarks,
                            path_out=path_out,
                            step_hide_landmarks=step_lnds,
                            tissue_partial=tissue_partial)

    for sc_name in scale_names:
        _create_cover(sc_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')
    main(path_images=DATASET_IMAGES, path_landmarks=DATASET_LANDMARKS,
         path_out=DATASET_COVERS, step_lnds=HIDE_TEST_TISSUE_STEP,
         dataset=DATASET_TISSUE_SCALE, scale_names=SCALE_NAMES,
         tissue_partial=DATASET_TISSUE_SCALE_PARTIAL.keys())
    logging.info('Done :]')
