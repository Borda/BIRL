"""
According given annotations select a subset and add synthetic points
and scale it into particular scales used in dataset

The expected structure of annotations is as follows
ANNOTATIONS/<tissue>/<user>_scale-<number>pc/<csv-file>
The expected structure of dataset is
DATASET/<tissue>/scale-<number>pc/<image-file>

EXAMPLE
-------
>> python rescale_tissue_landmarks.py -a data_images -d results

>> python bm_dataset/rescale_tissue_landmarks.py \
    -a /datagrid/Medical/dataset_ANHIR/landmarks_all \
    -d /datagrid/Medical/dataset_ANHIR/landmarks_user \
    --scales 2 5 10 15 20 25 50 100 --nb_selected 0.2

In case, you are working with the user annotation you need to generate consensus
landmark annotation first, using https://borda.github.io/dataset-histology-landmarks/
>> python handlers/run_generate_landmarks.py \
    -a /datagrid/Medical/dataset_ANHIR/landmarks_annot \
    -d /datagrid/Medical/dataset_ANHIR/landmarks_all \
    --scales 2 5 10 15 20 25 50 100

Copyright (C) 2014-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.experiments import (wrap_execute_sequence,
                                             parse_arg_params, is_iterable)
from benchmark.utilities.data_io import create_folder, load_landmarks_csv, save_landmarks_csv
from benchmark.utilities.dataset import (list_sub_folders, parse_path_scale,
                                         compute_bounding_polygon, inside_polygon)
from benchmark.utilities.registration import estimate_affine_transform, transform_points
from bm_dataset.rescale_tissue_images import NB_THREADS, DEFAULT_SCALES, FOLDER_TEMPLATE


def arg_parse_params():
    """ argument parser from cmd

    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: ...}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--path_annots', type=str, required=False,
                        help='path to folder with annotations')
    parser.add_argument('-d', '--path_dataset', type=str, required=False,
                        help='path to the output directory - dataset')
    parser.add_argument('--scales', type=int, required=False, nargs='*',
                        help='generated scales for the dataset', default=DEFAULT_SCALES)
    parser.add_argument('--nb_selected', type=float, required=False, default=None,
                        help='number ot ration of selected landmarks')
    parser.add_argument('--nb_total', type=int, required=False, default=None,
                        help='total number of generated landmarks')
    parser.add_argument('--nb_workers', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    args = parse_arg_params(parser)
    if not is_iterable(args['scales']):
        args['scales'] = [args['scales']]
    return args


def load_largest_scale(path_set):
    """ in given set find the largest scale and load all landmarks in full size

    :param str path_set: path to image/landmark set
    :return {str: ndarray}: dictionary with loaded landmarks in full scale
    """
    scales_folders = [(parse_path_scale(p), os.path.basename(p))
                      for p in list_sub_folders(path_set)]
    if not scales_folders:
        return None
    scale, folder = sorted(scales_folders, reverse=True)[0]

    paths_csv = glob.glob(os.path.join(path_set, folder, '*.csv'))
    scaling = 100. / scale
    names_lnds = {os.path.basename(p): load_landmarks_csv(p) * scaling
                  for p in paths_csv}
    return names_lnds


def generate_random_points_inside(ref_points, nb_extras):
    """ generate some extra points inside the tissue boundary polygon

    :param ref_points: point of the tissue
    :param int nb_extras: number of point to be added
    :return [(int, int)]: extra points
    """
    # tighter approximation, not all tissue is really convex
    convex_polygon = compute_bounding_polygon(ref_points)
    poly_mins = np.min(convex_polygon, axis=0)
    poly_size = np.max(convex_polygon, axis=0) - poly_mins
    # generate sample points inside polygon
    points_extra = []
    for i in range(int(1e5)):
        point = (np.random.random(2) * poly_size + poly_mins).astype(int)
        if inside_polygon(convex_polygon, point):
            points_extra.append(point)
            if len(points_extra) > nb_extras:
                logging.debug('particular polygon generated %f inside',
                              nb_extras / float(i))
                break
    else:  # in case the loop ended regularly without break
        logging.warning('something went wrong with ')
    return points_extra


def expand_random_warped_landmarks(names_lnds, names_lnds_new, nb_total):
    """ add some extra point which are randomly sampled in the first sample
    and warped to the other images using estimated affine transform

    :param {str: ndarray} names_lnds: the original landmarks
    :param {str: ndarray} names_lnds_new: the generated landmarks
    :param int nb_total: total number of point - landmarks
    :return {str: ndarray}:
    """
    # estimate then number of required points
    nb_min_new = min(map(len, names_lnds_new.values()))
    nb_extras = nb_total - nb_min_new
    if nb_extras <= 0:
        return names_lnds_new

    ref_name = sorted(names_lnds)[0]
    ref_points = names_lnds[ref_name]
    points_extra = generate_random_points_inside(ref_points, nb_extras)

    for name in filter(lambda n: n != ref_name, names_lnds):
        # prepare the points
        nb_common = min([len(names_lnds[ref_name]), len(names_lnds[name])])
        pts1 = names_lnds[ref_name][:nb_common]
        pts2 = names_lnds[name][:nb_common]
        # estimate the internal affine transformation
        matrix, _, _, _ = estimate_affine_transform(pts1, pts2)
        points_warp = transform_points(points_extra, matrix)
        # insert the warped points
        names_lnds_new[name] = np.vstack([names_lnds_new[name][:nb_min_new],
                                          points_warp])
    # insert also the reference sample
    names_lnds_new[ref_name] = np.vstack([names_lnds_new[ref_name][:nb_min_new],
                                          points_extra])

    # reorder landmarks but equally in all sets
    reorder = list(range(nb_total))
    np.random.shuffle(reorder)
    names_lnds_new = {n: names_lnds_new[n][reorder] for n in names_lnds_new}
    return names_lnds_new


def extend_landmarks(path_set, path_dataset, nb_selected=None, nb_total=None):
    """ select and extend the original set of landmarks

    :param str path_set: path to the particular set if images/landmarks
    :param str path_dataset: root path to generated dataset
    :param float|int|None nb_selected: portion of selected points,
        if None use all original landmarks
    :param int|None nb_total: add extra points up to total number,
        if None, no adding extra points
    :return:
    """
    logging.debug('> processing: %s', path_set)

    # search form mas scale in set and load all related landmarks
    names_lnds = load_largest_scale(path_set)
    if not names_lnds:
        logging.warning('no landmarks was loaded for "%s"', path_set)
        return

    # select subset of selected landmarks
    names_lnds_new = {}
    if nb_selected is not None:
        assert nb_selected >= 0, 'number of selected has to be positive'
        lens = list(map(len, names_lnds.values()))
        # transform the relative count to absolute number
        if nb_selected < 1:
            nb_selected = np.ceil(nb_selected * max(lens)).astype(int)
        # perform the selection
        indexes = list(range(min(lens)))
        np.random.shuffle(indexes)
        # just a required subset
        indexes = indexes[:nb_selected]
        for name in names_lnds:
            names_lnds_new[name] = names_lnds[name][indexes]
    else:
        names_lnds_new = names_lnds

    if nb_total is not None:
        names_lnds_new = expand_random_warped_landmarks(
            names_lnds, names_lnds_new, nb_total)

    # export the landmarks
    path_set_scale = os.path.join(path_dataset, os.path.basename(path_set),
                                  FOLDER_TEMPLATE % 100)
    create_folder(path_set_scale)
    for name in names_lnds_new:
        save_landmarks_csv(os.path.join(path_set_scale, name), names_lnds_new[name])


def dataset_expand_landmarks(path_annots, path_dataset, nb_selected=None,
                             nb_total=None, nb_workers=NB_THREADS):
    """ select and expand over whole dataset

    :param str path_annots: root path to original dataset
    :param str path_dataset: root path to generated dataset
    :param float|int|None nb_selected: portion of selected points
    :param int|None nb_total: add extra points up to total number
    :param int nb_workers: number of jobs running in parallel
    :return [int]:
    """
    list_sets = list_sub_folders(path_annots)
    logging.info('Found sets: %i', len(list_sets))

    _wrap_extend = partial(extend_landmarks, path_dataset=path_dataset,
                           nb_selected=nb_selected, nb_total=nb_total)
    counts = list(wrap_execute_sequence(_wrap_extend, sorted(list_sets),
                                        nb_workers=nb_workers, desc='expand landmarks'))
    return counts


def scale_set_landmarks(path_set, scales=DEFAULT_SCALES):
    """ scale the updated (generated) landmarks

    the scales are created within the same path set

    :param str path_set: path to the image/landmark set
    :param [int] scales: created scales
    :return:
    """
    logging.debug('> processing: %s', path_set)
    path_scale100 = os.path.join(path_set, FOLDER_TEMPLATE % 100)
    if not os.path.isdir(path_scale100):
        logging.error('missing base scale 100pc in "%s"', path_scale100)
        return
    list_csv = glob.glob(os.path.join(path_scale100, '*.csv'))
    logging.debug('>> found landmarks: %i', len(list_csv))
    dict_lnds = {os.path.basename(p): pd.read_csv(p, index_col=0)
                 for p in list_csv}
    set_scales = {}
    for sc in (sc for sc in scales if sc not in [100]):  # drop the base scale
        folder_name = FOLDER_TEMPLATE % sc
        path_scale = create_folder(os.path.join(path_set, folder_name))
        for name in dict_lnds:
            df_scale = dict_lnds[name] * (sc / 100.)
            df_scale.to_csv(os.path.join(path_scale, name))
        set_scales[sc] = len(dict_lnds)
    dict_lens = {os.path.basename(path_set): set_scales}
    return dict_lens


def dataset_scale_landmarks(path_dataset, scales=DEFAULT_SCALES, nb_workers=NB_THREADS):
    """ generate several scales within the same dataset

    :param str path_dataset: path to the souorce/generated dataset
    :param [inr] scales: created scales
    :param int nb_workers: number of jobs running in parallel
    :return:
    """
    list_sets = list_sub_folders(path_dataset)
    logging.info('Found sets: %i', len(list_sets))

    _wrap_scale = partial(scale_set_landmarks, scales=scales)
    counts = list(wrap_execute_sequence(_wrap_scale, sorted(list_sets),
                                        nb_workers=nb_workers, desc='scaling sets'))
    return counts


def main(path_annots, path_dataset, scales, nb_selected=None, nb_total=None,
         nb_workers=NB_THREADS):
    """ main entry point

    :param str path_annots: root path to original dataset
    :param str path_dataset: root path to generated dataset
    :param [int] scales: generated scales
    :param float|int|None nb_selected: portion of selected points
    :param int|None nb_total: add extra points up to total number
    :param int nb_workers: number of jobs running in parallel
    :return:
    """
    count_gene = dataset_expand_landmarks(path_annots, path_dataset,
                                          nb_selected, nb_total, nb_workers=nb_workers)
    count_scale = dataset_scale_landmarks(path_dataset, scales=scales,
                                          nb_workers=nb_workers)
    return count_gene, count_scale


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    main(**arg_params)

    logging.info('DONE')
