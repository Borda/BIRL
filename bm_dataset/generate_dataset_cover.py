"""
Creating cover file for configuring registration image pairs
The paths and all other constants are set to run on CMP grid for ANHIR dataset

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import glob
import logging
from functools import partial

import pandas as pd

DATASET_IMAGES = '/datagrid/Medical/dataset_ANHIR/images'
DATASET_LANDMARKS = '/datagrid/Medical/dataset_ANHIR/landmarks_all'
DATASET_COVERS = '/datagrid/Medical/dataset_ANHIR/images'
NAME_COVER_FILE = 'dataset_%s.csv'
GENERATED_SCALES = (5, 10, 25, 50, 100)
NAME_DIR_SCALE = 'scale-%ipc'
DATASET_TISSUE_SCALE = {
    'kidney_*': 5,
    'lesions_[1,3]': 10,
    'lesions_2': 5,
    'lung-lobes_*': 10,
    'mammary-gland_*': 5,
    'COAD_*': 5,
}
DATASET_TISSUE_PARTIAL = ('kidney_*', 'COAD_*')
STEP_PARTIAL = 3
DATASET_SCALES = {
    'small': 0,
    'medium': 1,
    'big': 2,
}
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
COLUMNS_EMPTY = (
    'Warped target landmarks',
    'Warped source landmarks',
    'Execution time [minutes]'
)


def get_relative_paths(paths, path_base):
    paths_r = [p.replace(path_base, '')[1:] for p in sorted(paths)]
    return paths_r


def list_landmarks_images(path_tissue, sc, path_landmarks, path_images):
    path_ = os.path.join(path_tissue, NAME_DIR_SCALE % sc, '*.csv')
    rp_lnds = get_relative_paths(glob.glob(path_), path_landmarks)
    paths_imgs = []
    for rp_lnd in rp_lnds:
        p_imgs = glob.glob(os.path.join(path_images, os.path.splitext(rp_lnd)[0] + '.*'))
        p_imgs = [p for p in p_imgs if os.path.splitext(p)[-1] in IMAGE_EXTENSIONS]
        if not len(p_imgs):
            logging.error('missing image for "%s"', rp_lnd)
            return [], []
        paths_imgs.append(sorted(p_imgs)[0])
    rp_imgs = get_relative_paths(paths_imgs, path_images)
    return rp_lnds, rp_imgs


def get_pairing(count, step=None):
    idxs = list(range(count))
    priv = idxs[::step]
    # prune image on diagonal and missing both landmarks (targer and source)
    pairs = [(i, j) for i in idxs for j in idxs
             if i != j and (i not in priv or j in priv)]
    # prune symmetric image pairs
    pairs = [(i, j) for k, (i, j) in enumerate(pairs)
             if (j, i) not in pairs[:k]]
    return pairs


def generate_reg_pairs(rp_imgs, rp_lnds, pairs):
    reg_pairs = []
    for i, j in pairs:
        reg_pairs.append({
            'Target image': rp_imgs[i],
            'Source image': rp_imgs[j],
            'Target landmarks': rp_lnds[i],
            'Source landmarks': rp_lnds[j],
        })
    return reg_pairs


def create_dataset_cover(dataset, path_images, path_landmarks, path_out,
                         step_landmarks, tissue_partial):
    name, scale_step = dataset

    reg_pairs = []
    for tissue in sorted(DATASET_TISSUE_SCALE):
        sc = DATASET_TISSUE_SCALE[tissue]
        sc = GENERATED_SCALES[min(GENERATED_SCALES.index(sc) + scale_step,
                                  len(GENERATED_SCALES))]
        paths_tissue = [p for p in glob.glob(os.path.join(path_landmarks, tissue))
                        if os.path.isdir(p)]
        for p_tissue in sorted(paths_tissue):
            rp_lnds, rp_imgs = list_landmarks_images(p_tissue, sc,
                                                     path_landmarks, path_images)
            assert len(rp_lnds) == len(rp_imgs), \
                'the list of landmarks and images does not match'
            step_landmarks = step_landmarks if tissue in tissue_partial else None
            pairs = get_pairing(len(rp_lnds), step_landmarks)
            reg_pairs += generate_reg_pairs(rp_imgs, rp_lnds, pairs)

    df_cover = pd.DataFrame(reg_pairs)
    for col in COLUMNS_EMPTY:
        df_cover[col] = None
    path_csv = os.path.join(path_out, NAME_COVER_FILE % name)
    logging.info('exporting CSV: %s', path_csv)
    df_cover.to_csv(path_csv)


def main(path_images, path_landmarks, path_out, step_lnds, datasets,
         tissue_partial):

    _create_cover = partial(create_dataset_cover, path_images=path_images,
                            path_landmarks=path_landmarks, path_out=path_out,
                            step_lnds=step_lnds, tissue_partial=tissue_partial)

    for name in datasets:
        scale_step = datasets[name]
        _create_cover((name, scale_step))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')
    main(path_images=DATASET_IMAGES, path_landmarks=DATASET_LANDMARKS,
         path_out=DATASET_COVERS, step_lnds=STEP_PARTIAL,
         datasets=DATASET_SCALES, tissue_partial=DATASET_TISSUE_PARTIAL)
    logging.info('Done :]')
