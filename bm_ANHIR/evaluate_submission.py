"""
Evaluating passed experiments, for instance if metric was changed

The expected submission structure and required files:
 * `registration-results.csv` - cover file with experimental results
 * `computer-performances.json` - computer performance evaluation
 * landmarks in CSV files with relative path described
    in `registration-results.csv` in column 'Warped source landmarks'

The required files in the reference (ground truth):
 * `dataset.csv` - cover file with planed registrations
 * `computer-performances.json` - reference performance evaluation
 * `lnds_provided/` provided landmarks in CSV files with relative path described
    in `dataset.csv` in column 'Source landmarks'
 * `lnds_reference/` reference (ground truth) landmarks in CSV files with relative
    path described in `dataset_cover.csv` in both columns 'Target landmarks'
    and 'Source landmarks'


EXAMPLE
-------
>> python evaluate_submission.py \
    -e ./results/BmUnwarpJ \
    -c ./data_images/pairs-imgs-lnds_anhir.csv \
    -d ./data_images \
    -r ./data_images \
    -p ./bm_experiments/computer-performances_cmpgrid-71.json \
    -o ./output \
    --min_landmarks 0.20

DOCKER
------
>> python evaluate_submission.py \
    -e /input \
    -c /opt/evaluation/dataset.csv \
    -d /opt/evaluation/lnds_provided \
    -r /opt/evaluation/lnds_reference \
    -p /opt/evaluation/computer-performances.json \
    -o /output \
    --min_landmarks 0.20
or run locally:
>> python evaluate_submission.py \
    -e bm_ANHIR/submission \
    -c bm_ANHIR/dataset_ANHIR/dataset_medium.csv \
    -d bm_ANHIR/dataset_ANHIR/landmarks_user \
    -r bm_ANHIR/dataset_ANHIR/landmarks_all \
    -p bm_ANHIR/dataset_ANHIR/computer-performances_cmpgrid-71.json \
    -o output \
    --min_landmarks 0.20

References:
* https://grand-challengeorg.readthedocs.io/en/latest/evaluation.html

Copyright (C) 2018-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
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
from benchmark.utilities.dataset import common_landmarks, parse_path_scale
from benchmark.utilities.experiments import wrap_execute_sequence, parse_arg_params
from benchmark.cls_benchmark import (
    NAME_CSV_REGISTRATION_PAIRS, COVER_COLUMNS, COVER_COLUMNS_WRAP,
    COL_IMAGE_REF_WARP, COL_POINTS_REF_WARP, COL_POINTS_REF, COL_POINTS_MOVE,
    COL_TIME, COL_ROBUSTNESS, compute_landmarks_statistic, update_path_)
# from bm_experiments.bm_comp_perform import NAME_REPORT

NB_THREADS = max(1, int(mproc.cpu_count() * .9))
NAME_CSV_RESULTS = 'registration-results.csv'
NAME_JSON_COMPUTER = 'computer-performances.json'
NAME_JSON_RESULTS = 'metrics.json'
COL_NORM_TIME = 'Norm. execution time [minutes]'
COL_FOUND_LNDS = 'Ration matched landmarks'
CMP_THREADS = ('1', 'n')


def create_parser():
    """ parse the input parameters
    :return dict: {str: any}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--path_experiment', type=str, required=True,
                        help='path to the experiments', default='/input/')
    parser.add_argument('-c', '--path_cover', type=str, required=True,
                        help='path to cover table (csv file)',
                        default='/opt/evaluation/dataset.csv')
    parser.add_argument('-d', '--path_dataset', type=str, required=True,
                        help='path to dataset with provided landmarks',
                        default='/opt/evaluation/provided')
    parser.add_argument('-r', '--path_reference', type=str, required=False,
                        help='path to complete ground truth landmarks')
    parser.add_argument('-p', '--path_comp_bm', type=str, required=False,
                        help='path to reference computer performance JSON')
    parser.add_argument('-o', '--path_output', type=str, required=True,
                        help='path to output results', default='/output/')
    # required number of submitted landmarks, match values in COL_FOUND_LNDS
    parser.add_argument('--min_landmarks', type=float, required=False, default=0.5,
                        help='ration of required landmarks in submission')
    return parser


def filter_landmarks(idx_row, path_output, path_dataset, path_reference):
    """ filter all relevant landmarks which were used and copy them to experiment

    :param (idx, {}|Series) idx_row: experiment DataFrame
    :param str path_output: path to output folder
    :param str path_dataset: path to provided landmarks
    :param str path_reference: path to the complete landmark collection
    :return (idx, float): record index and match ratio
    """
    idx, row = idx_row
    path_ref = update_path_(row[COL_POINTS_MOVE], path_reference)
    path_load = update_path_(row[COL_POINTS_MOVE], path_dataset)
    pairs = common_landmarks(load_landmarks(path_ref), load_landmarks(path_load),
                             threshold=1)
    pairs = sorted(pairs.tolist(), key=lambda p: p[1])
    ind_ref = np.asarray(pairs)[:, 0]

    # moving and reference landmarks
    for col in [COL_POINTS_REF, COL_POINTS_MOVE]:
        path_in = update_path_(row[col], path_reference)
        path_out = update_path_(row[col], path_output)
        create_folder(os.path.dirname(path_out), ok_existing=True)
        save_landmarks(path_out, load_landmarks(path_in)[ind_ref])

    # save ratio of found landmarks
    len_lnds_ref = len(load_landmarks(update_path_(row[COL_POINTS_REF], path_reference)))
    ratio_matches = len(pairs) / float(len_lnds_ref)
    return idx, ratio_matches


def normalize_exec_time(df_experiments, path_experiments, path_comp_bm=None):
    """ normalize execution times if reference and experiment computer is given

    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_comp_bm: path to reference comp. benchmark
    """
    path_comp_bm_expt = os.path.join(path_experiments, NAME_JSON_COMPUTER)
    if COL_TIME not in df_experiments.columns:
        logging.warning('Missing %s among result columns.', COL_TIME)
        return
    if not path_comp_bm:
        logging.warning('Reference comp. perform. not specified.')
        return
    elif not all(os.path.isfile(p) for p in [path_comp_bm, path_comp_bm_expt]):
        logging.warning('Missing one of the JSON files: \n %s (%s)\n %s (%s)',
                        path_comp_bm, os.path.isfile(path_comp_bm),
                        path_comp_bm_expt, os.path.isfile(path_comp_bm_expt))
        return

    logging.info('Normalizing the Execution time.')
    with open(path_comp_bm, 'r') as fp:
        comp_ref = json.load(fp)
    with open(path_comp_bm_expt, 'r') as fp:
        comp_exp = json.load(fp)

    time_ref = np.mean([comp_ref['registration @%s-thread' % i] for i in CMP_THREADS])
    time_exp = np.mean([comp_exp['registration @%s-thread' % i] for i in CMP_THREADS])
    coef = time_ref / time_exp
    df_experiments[COL_NORM_TIME] = df_experiments[COL_TIME] * coef


def parse_landmarks(idx_row):
    """ parse the warped landmarks and reference and save them as cases

    :param (int, series) idx_row: individual row
    :return {str: float|[]}: parsed registration pair
    """
    idx, row = idx_row
    row = dict(row)
    # lnds_ref = load_landmarks(update_path_(row[COL_POINTS_REF], path_experiments))
    # lnds_warp = load_landmarks(update_path_(row[COL_POINTS_MOVE_WARP], path_experiments))
    #     if isinstance(row[COL_POINTS_MOVE_WARP], str)else np.array([[]])
    path_dir = os.path.dirname(row[COL_POINTS_MOVE])
    match_lnds = np.nan_to_num(row[COL_FOUND_LNDS]) if COL_FOUND_LNDS in row else 0.
    robust = int(row['TRE Mean (final)'] < row['TRE Mean (init)']) \
        if 'TRE Mean (final)' in row else 0.
    record = {
        'name-tissue': os.path.basename(os.path.dirname(path_dir)),
        'scale-tissue': parse_path_scale(os.path.basename(path_dir)),
        'name-reference': os.path.splitext(os.path.basename(row[COL_POINTS_REF]))[0],
        'name-source': os.path.splitext(os.path.basename(row[COL_POINTS_MOVE]))[0],
        # 'reference landmarks': np.round(lnds_ref, 1).tolist(),
        # 'warped landmarks': np.round(lnds_warp, 1).tolist(),
        'matched-landmarks': match_lnds,
        'Robustness': robust,
        'Norm-Time_minutes': row[COL_NORM_TIME]
    }
    # copy all columns with rTRE, TRE and Overlap
    record.update({col.replace(' (final)', '').replace(' ', '-'): row[col]
                   for col in row if '(final)' in col})
    return idx, record


def compute_scores(df_experiments, min_landmarks=1.):
    """ compute all main metrics

    SEE: https://anhir.grand-challenge.org/Evaluation/

    :param DF df_experiments: complete experiments
    :param float min_landmarks: required number of submitted landmarks in range (0, 1),
        match values in COL_FOUND_LNDS
    :return {}: results
    """
    # if the initial overlap and submitted overlap do not mach, drop results
    if 'overlap points (final)' not in df_experiments.columns:
        raise ValueError('Missing `overlap points (final)` column,'
                         ' because there are probably missing wrap landmarks.')
    hold_overlap = df_experiments['overlap points (init)'] == df_experiments['overlap points (final)']
    mask_incomplete = ~hold_overlap | df_experiments[COL_FOUND_LNDS] < min_landmarks
    # rewrite incomplete cases by initial stat
    if sum(mask_incomplete) > 0:
        for col_f, col_i in zip(*_filter_measure_columns(df_experiments)):
            df_experiments.loc[mask_incomplete, col_f] = df_experiments.loc[mask_incomplete, col_i]
        df_experiments.loc[mask_incomplete, COL_ROBUSTNESS] = 0
        logging.warning('There are %i cases which incomplete landmarks.',
                        sum(mask_incomplete))

    # compute summary
    df_summary = df_experiments.describe()
    df_robust = df_experiments[df_experiments[COL_ROBUSTNESS] > 0.5]
    df_summary_robust = df_robust.describe()
    pd.set_option('expand_frame_repr', False)

    # pre-compute some optional metrics
    score_used_lnds = df_summary_robust[COL_FOUND_LNDS]['mean'] \
        if COL_FOUND_LNDS in df_experiments.columns else 0
    if COL_NORM_TIME in df_experiments.columns:
        time_all = df_summary[COL_NORM_TIME]['mean']
        time_robust = df_summary_robust[COL_NORM_TIME]['mean']
    else:
        time_all, time_robust = np.nan, np.nan
    # parse final metrics
    scores = {
        'Average-Robustness': df_summary[COL_ROBUSTNESS]['mean'],
        'Average-Median-rTRE': df_summary['rTRE Median (final)']['mean'],
        'Average-Median-rTRE-Robust': df_summary_robust['rTRE Median (final)']['mean'],
        'Average-Rank-Median-rTRE': None,
        'Average-Max-rTRE': df_summary['rTRE Max (final)']['mean'],
        'Average-Max-rTRE-Robust': df_summary_robust['rTRE Max (final)']['mean'],
        'Average-Rank-Max-rTRE': None,
        'Average-used-landmarks': score_used_lnds,
        'Average-Norm-Time': time_all,
        'Average-Norm-Time-Robust': time_robust,
    }
    return scores


def _filter_measure_columns(df_experiments):
    # copy the initial to final for missing
    cols_final = [col for col in df_experiments.columns
                  if re.match(r'(r)?TRE \w+ .final.', col)]
    cols_init = [col.replace('final', 'init') for col in cols_final]
    return cols_final, cols_init


def export_summary_json(df_experiments, path_experiments, path_output, min_landmarks=1.):
    """ summarize results in particular JSON format

    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_output: path to generated results
    :param float min_landmarks: required number of submitted landmarks in range (0, 1),
        match values in COL_FOUND_LNDS
    :return str: path to exported results
    """
    # export partial results
    cases = list(wrap_execute_sequence(parse_landmarks, df_experiments.iterrows(),
                                       desc='Parsing landmarks', nb_workers=1))

    # copy the initial to final for missing
    for col, col2 in zip(*_filter_measure_columns(df_experiments)):
        mask = df_experiments[col].isnull()
        df_experiments.loc[mask, col] = df_experiments.loc[mask, col2]

    # parse final metrics
    scores = compute_scores(df_experiments, min_landmarks)

    path_comp_bm_expt = os.path.join(path_experiments, NAME_JSON_COMPUTER)
    if os.path.isfile(path_comp_bm_expt):
        with open(path_comp_bm_expt, 'r') as fp:
            comp_exp = json.load(fp)
    else:
        comp_exp = None

    results = {'aggregates': scores, 'cases': dict(cases), 'computer': comp_exp}
    path_json = os.path.join(path_output, NAME_JSON_RESULTS)
    logging.info('exporting JSON results: %s', path_json)
    with open(path_json, 'w') as fp:
        json.dump(results, fp)
    return path_json


def main(path_experiment, path_cover, path_dataset, path_output,
         path_reference=None, path_comp_bm=None, nb_workers=NB_THREADS, min_landmarks=1.):
    """ main entry point

    :param str path_experiment: path to experiment folder
    :param str path_cover: path to assignment file (requested registration pairs)
    :param str path_dataset: path to provided landmarks
    :param str path_output: path to generated results
    :param str|None path_reference: path to the complete landmark collection,
        if None use dataset folder
    :param str|None path_comp_bm: path to reference comp. benchmark
    :param int nb_workers: number of parallel processes
    :param float min_landmarks: required number of submitted landmarks in range (0, 1),
        match values in COL_FOUND_LNDS
    """

    path_results = os.path.join(path_experiment, NAME_CSV_REGISTRATION_PAIRS)
    if not os.path.isfile(path_results):
        raise AttributeError('Missing experiments results: %s' % path_results)
    path_reference = path_dataset if not path_reference else path_reference

    # drop time column from Cover which should be empty
    df_cover = pd.read_csv(path_cover).drop([COL_TIME], axis=1, errors='ignore')
    # drop Warp* column from Cover which should be empty
    df_cover = df_cover.drop([col for col in df_cover.columns if 'warped' in col.lower()],
                             axis=1, errors='ignore')
    df_results = pd.read_csv(path_results)
    df_results = df_results[[col for col in list(COVER_COLUMNS_WRAP) + [COL_TIME]
                             if col in df_results.columns]]
    df_experiments = pd.merge(df_cover, df_results, how='left', on=COVER_COLUMNS)
    df_experiments.drop([COL_IMAGE_REF_WARP, COL_POINTS_REF_WARP],
                        axis=1, errors='ignore', inplace=True)

    normalize_exec_time(df_experiments, path_experiment, path_comp_bm)

    logging.info('Filter used landmarks.')
    _filter_lnds = partial(filter_landmarks, path_output=path_output,
                           path_dataset=path_dataset, path_reference=path_reference)
    for idx, ratio in wrap_execute_sequence(_filter_lnds, df_experiments.iterrows(),
                                            desc='Filtering', nb_workers=nb_workers):
        df_experiments.loc[idx, COL_FOUND_LNDS] = np.round(ratio, 2)

    logging.info('Compute landmarks statistic.')
    _compute_lnds_stat = partial(compute_landmarks_statistic, df_experiments=df_experiments,
                                 path_dataset=path_output, path_experiment=path_experiment)
    # NOTE: this has to run in SINGLE thread so there is SINGLE table instance
    list(wrap_execute_sequence(_compute_lnds_stat, df_experiments.iterrows(),
                               desc='Statistic', nb_workers=1))

    path_results = os.path.join(path_output, os.path.basename(path_results))
    logging.debug('exporting CSV results: %s', path_results)
    df_experiments.to_csv(path_results)

    export_summary_json(df_experiments, path_experiment, path_output, min_landmarks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = parse_arg_params(create_parser())
    main(**arg_params)

    logging.info('DONE')
