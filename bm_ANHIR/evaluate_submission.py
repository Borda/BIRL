"""
Evaluating passed experiments, for instance if metric was changed

The expected submission structure and required files:
 * `registration-results.csv` - cover file with experimental results
 * `computer-performances.json` - computer performance evaluation
 * landmarks in CSV files with relative path described
    in `registration-results.csv` in column 'Warped source landmarks'

The required files in the reference (ground truth):
 * `dataset_cover.csv` - cover file with planed registrations
 * `computer-performances.json` - reference performance evaluation
 * `provided/` provided landmarks in CSV files with relative path described
    in `dataset_cover.csv` in column 'Source landmarks'
 * `reference/` reference (ground truth) landmarks in CSV files with relative
    path described in `dataset_cover.csv` in both columns 'Target landmarks'
    and 'Source landmarks'


EXAMPLE
-------
>> python evaluate_experiment.py \
    -e ./results/BmUnwarpJ \
    -c ./data_images/pairs-imgs-lnds_anhir.csv \
    -d ./data_images \
    -p ./bm_experiments/computer-performances_cmpgrid-71.json \
    -o ./output

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import re
import json
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.data_io import create_folder, load_landmarks, save_landmarks
from benchmark.utilities.dataset import common_landmarks
from benchmark.utilities.experiments import wrap_execute_sequence, parse_arg_params
from benchmark.cls_benchmark import (
    NAME_CSV_REGISTRATION_PAIRS, COVER_COLUMNS, COL_IMAGE_REF_WARP, COL_POINTS_REF_WARP,
    COL_POINTS_REF, COL_POINTS_MOVE, COL_POINTS_MOVE_WARP, COL_TIME, COL_ROBUSTNESS,
    COL_IMAGE_DIAGONAL, COL_IMAGE_SIZE, compute_landmarks_statistic, update_path_)
from bm_experiments.bm_comp_perform import NAME_REPORT

NB_THREADS = max(1, int(mproc.cpu_count() * .9))
NAME_CSV_RESULTS = 'registration-results.csv'
NAME_JSON_COMPUTER = NAME_REPORT
NAME_JSON_RESULTS = 'metrics.json'
COL_NORM_TIME = 'Norm. execution time [minutes]'
COL_FOUND_LNDS = 'Ration matched landmarks'


def create_parser():
    """ parse the input parameters
    :return dict: {str: any}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--path_experiment', type=str, required=True,
                        help='path to the experiments')
    parser.add_argument('-c', '--path_cover', type=str, required=True,
                        help='path to cover table (csv file)')
    parser.add_argument('-d', '--path_dataset', type=str, required=True,
                        help='path to dataset with provided landmarks')
    parser.add_argument('-r', '--path_reference', type=str, required=False,
                        help='path to complete ground truth landmarks')
    parser.add_argument('-p', '--path_comp_bm', type=str, required=False,
                        help='path to reference computer performance JSON')
    parser.add_argument('-o', '--path_output', type=str, required=False,
                        help='path to output results')
    return parser


def filter_landmarks(idx_row, df_experiments, path_experiments, path_dataset, path_reference):
    """ filter all relevant landmarks which were used and copy them to experiment

    :param DF idx_row: experiment DataFrame
    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_dataset: path to provided landmarks
    :param str path_reference: path to the complete landmark collection
    """
    idx, row = idx_row
    path_ref = update_path_(row[COL_POINTS_MOVE], path_reference)
    path_load = update_path_(row[COL_POINTS_MOVE], path_dataset)
    pairs = common_landmarks(load_landmarks(path_ref), load_landmarks(path_load),
                             threshold=1)
    ind_ref, ind_load = pairs[:, 0], pairs[:, 1]

    # moving and reference landmarks
    for col in [COL_POINTS_REF, COL_POINTS_MOVE]:
        path_in = update_path_(row[col], path_reference)
        path_out = update_path_(row[col], path_experiments)
        create_folder(os.path.dirname(path_out), ok_existing=True)
        save_landmarks(path_out, load_landmarks(path_in)[ind_ref])

    if COL_POINTS_MOVE_WARP not in row or not isinstance(row[COL_POINTS_MOVE_WARP], str):
        logging.debug('missing warped landmarks for: %r', dict(row))
        return
    # warped landmarks
    path_lnds = update_path_(row[COL_POINTS_MOVE_WARP], path_experiments)
    lnds_warp = load_landmarks(path_lnds)
    save_landmarks(path_lnds, lnds_warp[ind_load])
    # save ratio of found landmarks
    df_experiments.loc[idx, COL_FOUND_LNDS] = len(pairs) / float(len(lnds_warp))


def normalize_exec_time(df_experiments, path_experiments, path_comp_bm=None):
    """ normalize execution times if reference and experiment computer is given

    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_comp_bm: path to reference comp. benchmark
    """
    path_comp_bm_expt = os.path.join(path_experiments, NAME_JSON_COMPUTER)
    if not path_comp_bm:
        logging.warning('Reference comp. perform. not specified.')
        return
    elif not all(os.path.isfile(p) for p in [path_comp_bm, path_comp_bm_expt]):
        logging.warning('Missing one of the JSON files: \n %s (%s)\n %s (%s)',
                        path_comp_bm, os.path.isfile(path_comp_bm),
                        path_comp_bm_expt, os.path.isfile(path_comp_bm_expt))
        return

    with open(path_comp_bm, 'r') as fp:
        comp_ref = json.load(fp)
    with open(path_comp_bm_expt, 'r') as fp:
        comp_exp = json.load(fp)

    time_ref = np.mean([comp_ref['registration @%s-thread' % i] for i in ['1', 'n']])
    time_exp = np.mean([comp_exp['registration @%s-thread' % i] for i in ['1', 'n']])
    coef = time_ref / time_exp
    df_experiments[COL_NORM_TIME] = df_experiments[COL_TIME] * coef


def parse_landmarks(row, path_experiments):
    lnds_ref = load_landmarks(
        update_path_(row[COL_POINTS_REF], path_experiments))
    if isinstance(row[COL_POINTS_MOVE_WARP], str):
        lnds_warp = load_landmarks(update_path_(row[COL_POINTS_MOVE_WARP], path_experiments))
    else:
        lnds_warp = np.array([[]])
    path_dir = os.path.dirname(row[COL_POINTS_MOVE])
    record = {
        'tissue': os.path.basename(os.path.dirname(path_dir)),
        'scale': os.path.basename(path_dir),
        'reference name': os.path.splitext(os.path.basename(row[COL_POINTS_REF]))[0],
        'source name': os.path.splitext(os.path.basename(row[COL_POINTS_MOVE]))[0],
        'reference landmarks': np.round(lnds_ref, 1).tolist(),
        'warped landmarks': np.round(lnds_warp, 1).tolist(),
        'matched landmarks': np.nan_to_num(row[COL_FOUND_LNDS])
    }
    return record


def compute_scores(df_experiments):
    # compute summary
    df_summary = df_experiments.describe()
    df_robust = df_experiments[df_experiments[COL_ROBUSTNESS] > 0.5]
    df_summary_robust = df_robust.describe()
    pd.set_option('expand_frame_repr', False)

    # parse final metrics
    scores = {
        'Avg. Robustness': df_summary[COL_ROBUSTNESS]['mean'],
        'Avg. median rTRE': df_summary['rTRE Median (final)']['mean'],
        'Avg. median rTRE robust.': df_summary_robust['rTRE Median (final)']['mean'],
        'Avg. rank median rTRE': None,
        'Avg. max rTRE': df_summary['rTRE Max (final)']['mean'],
        'Avg. max rTRE robust.': df_summary_robust['rTRE Max (final)']['mean'],
        'Avg. rank max rTRE': None,
    }
    if COL_NORM_TIME in df_experiments.columns:
        scores.update({
            'Avg. time': df_summary[COL_NORM_TIME]['mean'],
            'Avg. time robust.': df_summary_robust[COL_NORM_TIME]['mean'],
        })
    return scores


def export_summary_json(df_experiments, path_experiments, path_output):
    """ summarize results in particular JSON format

    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_output: path to generated results
    """
    # copy the initial to final for missing
    cols = [col for col in df_experiments.columns
            if re.match(r'(r)?TRE \w+ .final.', col)]
    for col in cols:
        mask = df_experiments[col].isnull()
        col2 = col.replace('final', 'init')
        df_experiments.loc[mask, col] = df_experiments.loc[mask, col2]

    # parse final metrics
    scores = compute_scores(df_experiments)

    # export partial results
    cases = []
    for idx, row in df_experiments.iterrows():
        record = parse_landmarks(row, path_experiments)
        cases.append(record)

    results = {'aggregates': scores, 'cases': cases}
    path_json = os.path.join(path_output, NAME_JSON_RESULTS)
    logging.info('exporting JSON results: %s', path_json)
    with open(path_json, 'w') as fp:
        json.dump(results, fp)


def main(path_experiment, path_cover, path_dataset, path_output,
         path_reference=None, path_comp_bm=None, nb_jobs=NB_THREADS):
    """ main entry point

    :param str path_experiment: path to experiment folder
    :param str path_cover: path to assignment file (requested registration pairs)
    :param str path_dataset: path to provided landmarks
    :param str path_output: path to generated results
    :param str|None path_reference: path to the complete landmark collection,
        if None use dataset folder
    :param str|None path_comp_bm: path to reference comp. benchmark
    :param int nb_jobs: number of parallel processes
    """

    path_results = os.path.join(path_experiment, NAME_CSV_REGISTRATION_PAIRS)
    assert os.path.isfile(path_results)
    path_reference = path_dataset if not path_reference else path_reference

    df_cover = pd.read_csv(path_cover)
    df_experiments = pd.read_csv(path_results)
    df_experiments.drop([COL_IMAGE_DIAGONAL, COL_IMAGE_SIZE],
                        axis=1, errors='ignore', inplace=True)
    df_experiments = pd.merge(df_cover, df_experiments, how='left', on=COVER_COLUMNS)
    df_experiments.drop([COL_IMAGE_REF_WARP, COL_POINTS_REF_WARP],
                        axis=1, errors='ignore', inplace=True)

    normalize_exec_time(df_experiments, path_experiment, path_comp_bm)

    # filter used landmarks
    _filter_lnds = partial(filter_landmarks, df_experiments=df_experiments,
                           path_experiments=path_experiment, path_dataset=path_dataset,
                           path_reference=path_reference)
    list(wrap_execute_sequence(_filter_lnds, df_experiments.iterrows(),
                               desc='Filtering', nb_jobs=nb_jobs))

    # compute landmarks statistic
    _compute_lnds_stat = partial(compute_landmarks_statistic, df_experiments=df_experiments,
                                 path_dataset=path_experiment, path_experiment=path_experiment)
    list(wrap_execute_sequence(_compute_lnds_stat, df_experiments.iterrows(),
                               desc='Statistic', nb_jobs=nb_jobs))

    path_results = os.path.join(path_output, os.path.basename(path_results))
    logging.debug('exporting CSV results: %s', path_results)
    df_experiments.to_csv(path_results)

    export_summary_json(df_experiments, path_experiment, path_output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = parse_arg_params(create_parser())
    main(**arg_params)

    logging.info('DONE')
