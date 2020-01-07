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

Sample usage::

    python evaluate_submission.py \
        -e ./results/BmUnwarpJ \
        -t ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -r ./data_images \
        -p ./bm_experiments/computer-performances_cmpgrid-71.json \
        -o ./output \
        --min_landmarks 0.20

DOCKER
------
Running in grad-challenge.org environment::

    python evaluate_submission.py \
        -e /input \
        -t /opt/evaluation/dataset.csv \
        -d /opt/evaluation/lnds_provided \
        -r /opt/evaluation/lnds_reference \
        -p /opt/evaluation/computer-performances.json \
        -o /output \
        --min_landmarks 0.20

or run locally::

    python bm_ANHIR/evaluate_submission.py \
        -e bm_ANHIR/submission \
        -t bm_ANHIR/dataset_ANHIR/dataset_medium.csv \
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
import time
import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import create_folder, load_landmarks, save_landmarks, update_path
from birl.utilities.dataset import parse_path_scale
from birl.utilities.experiments import iterate_mproc_map, parse_arg_params, FORMAT_DATE_TIME, nb_workers
from birl.benchmark import COL_PAIRED_LANDMARKS, ImRegBenchmark, filter_paired_landmarks, _df_drop_unnamed

NB_WORKERS = nb_workers(0.9)
NAME_CSV_RESULTS = 'registration-results.csv'
NAME_JSON_COMPUTER = 'computer-performances.json'
NAME_JSON_RESULTS = 'metrics.json'
COL_NORM_TIME = 'Norm. execution time [minutes]'
COL_TISSUE = 'Tissue kind'
# FOLDER_FILTER_DATASET = 'filtered dataset'
CMP_THREADS = ('1', 'n')
#: Require having initial overlap as the warped is tricky as some image pairs do not
#   have the same nb points, so recommend to set it as False
REQUIRE_OVERLAP_INIT_TARGET = False


def create_parser():
    """ parse the input parameters
    :return dict: parameters
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--path_experiment', type=str, required=True,
                        help='path to the experiments', default='/input/')
    parser.add_argument('-t', '--path_table', type=str, required=True,
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
    # required number of submitted landmarks, match values in COL_PAIRED_LANDMARKS
    parser.add_argument('--min_landmarks', type=float, required=False, default=0.5,
                        help='ration of required landmarks in submission')
    # parser.add_argument('--nb_workers', type=int, required=False, default=NB_WORKERS,
    #                     help='number of processes in parallel')
    parser.add_argument('--details', action='store_true', required=False,
                        default=False, help='export details for each case')
    return parser


def filter_export_landmarks(idx_row, path_output, path_dataset, path_reference):
    """ filter all relevant landmarks which were used and copy them to experiment

    The case is that in certain challenge stage users had provided just a subset
     of all image landmarks which could be laos shuffled. The idea is to filter identify
     all user used (provided in dataset) landmarks and filter them from temporary
     reference dataset.

    :param tuple(idx,dict|Series) idx_row: experiment DataFrame
    :param str path_output: path to output folder
    :param str path_dataset: path to provided landmarks
    :param str path_reference: path to the complete landmark collection
    :return tuple(idx,float): record index and match ratio
    """
    idx, row = idx_row

    ratio_matches, lnds_filter_ref, lnds_filter_move = \
        filter_paired_landmarks(row, path_dataset, path_reference,
                                ImRegBenchmark.COL_POINTS_MOVE,
                                ImRegBenchmark.COL_POINTS_REF)

    # moving and reference landmarks
    for col, lnds_flt in [(ImRegBenchmark.COL_POINTS_REF, lnds_filter_ref),
                          (ImRegBenchmark.COL_POINTS_MOVE, lnds_filter_move)]:
        path_out = update_path(row[col], pre_path=path_output)
        create_folder(os.path.dirname(path_out), ok_existing=True)
        if os.path.isfile(path_out):
            assert np.array_equal(load_landmarks(path_out), lnds_flt), \
                'overwrite different set of landmarks'
        save_landmarks(path_out, lnds_flt)

    return idx, ratio_matches


def normalize_exec_time(df_experiments, path_experiments, path_comp_bm=None):
    """ normalize execution times if reference and experiment computer is given

    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_comp_bm: path to reference comp. benchmark
    """
    path_comp_bm_expt = os.path.join(path_experiments, NAME_JSON_COMPUTER)
    if ImRegBenchmark.COL_TIME not in df_experiments.columns:
        logging.warning('Missing %s among result columns.', ImRegBenchmark.COL_TIME)
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
    df_experiments[COL_NORM_TIME] = df_experiments[ImRegBenchmark.COL_TIME] * coef


def parse_landmarks(idx_row):
    """ parse the warped landmarks and reference and save them as cases

    :param tuple(int,series) idx_row: individual row
    :return {str: float|[]}: parsed registration pair
    """
    idx, row = idx_row
    row = dict(row)
    # lnds_ref = load_landmarks(update_path_(row[COL_POINTS_REF], path_experiments))
    # lnds_warp = load_landmarks(update_path_(row[COL_POINTS_MOVE_WARP], path_experiments))
    #     if isinstance(row[COL_POINTS_MOVE_WARP], str)else np.array([[]])
    path_dir = os.path.dirname(row[ImRegBenchmark.COL_POINTS_MOVE])
    match_lnds = np.nan_to_num(row[COL_PAIRED_LANDMARKS]) if COL_PAIRED_LANDMARKS in row else 0.
    item = {
        'name-tissue': os.path.basename(os.path.dirname(path_dir)),
        'scale-tissue': parse_path_scale(os.path.basename(path_dir)),
        'type-tissue': row.get(COL_TISSUE, None),
        'name-reference': os.path.splitext(os.path.basename(row[ImRegBenchmark.COL_POINTS_REF]))[0],
        'name-source': os.path.splitext(os.path.basename(row[ImRegBenchmark.COL_POINTS_MOVE]))[0],
        # 'reference landmarks': np.round(lnds_ref, 1).tolist(),
        # 'warped landmarks': np.round(lnds_warp, 1).tolist(),
        'matched-landmarks': match_lnds,
        'Robustness': np.round(row.get(ImRegBenchmark.COL_ROBUSTNESS, 0), 3),
        'Norm-Time_minutes': np.round(row.get(COL_NORM_TIME, None), 5),
        'Status': row.get(ImRegBenchmark.COL_STATUS, None),
    }

    def _round_val(row, col):
        dec = 5 if col.startswith('rTRE') else 2
        return np.round(row[col], dec)

    # copy all columns with Affine statistic
    item.update({col.replace(' ', '-'): _round_val(row, col)
                 for col in row if 'affine' in col.lower()})
    # copy all columns with rTRE, TRE and Overlap
    # item.update({col.replace(' (final)', '').replace(' ', '-'): row[col]
    #              for col in row if '(final)' in col})
    item.update({col.replace(' (elastic)', '_elastic').replace(' ', '-'): _round_val(row, col)
                 for col in row if 'TRE' in col})
    # later in JSON keys ahs to be str only
    return str(idx), item


def compute_scores(df_experiments, min_landmarks=1.):
    """ compute all main metrics

    .. ref:: https://anhir.grand-challenge.org/Evaluation/

    :param DF df_experiments: complete experiments
    :param float min_landmarks: required number of submitted landmarks in range (0, 1),
        match values in COL_PAIRED_LANDMARKS
    :return dict: results
    """
    # if the initial overlap and submitted overlap do not mach, drop results
    if 'overlap points (target)' not in df_experiments.columns:
        raise ValueError('Missing `overlap points (target)` column,'
                         ' because there are probably missing wrap landmarks.')
    unpaired = df_experiments[COL_PAIRED_LANDMARKS] < min_landmarks
    hold_overlap = df_experiments['overlap points (init)'] == df_experiments['overlap points (target)']
    mask_incomplete = unpaired.copy()
    if REQUIRE_OVERLAP_INIT_TARGET:
        mask_incomplete |= ~hold_overlap
    # rewrite incomplete cases by initial stat
    if sum(mask_incomplete) > 0:
        for col_f, col_i in zip(*_filter_tre_measure_columns(df_experiments)):
            df_experiments.loc[mask_incomplete, col_f] = df_experiments.loc[mask_incomplete, col_i]
        df_experiments.loc[mask_incomplete, ImRegBenchmark.COL_ROBUSTNESS] = 0.
        logging.warning('There are %i cases which incomplete landmarks'
                        ' - unpaired %i & missed overlap %i.',
                        sum(mask_incomplete), sum(unpaired), sum(~hold_overlap))

    df_expt_robust = df_experiments[df_experiments[ImRegBenchmark.COL_ROBUSTNESS] > 0.5]
    pd.set_option('expand_frame_repr', False)

    # pre-compute some optional metrics
    score_used_lnds = np.mean(df_expt_robust[COL_PAIRED_LANDMARKS]) \
        if COL_PAIRED_LANDMARKS in df_experiments.columns else 0
    # parse specific metrics
    scores = {'Average-used-landmarks': score_used_lnds}

    scores.update(_compute_scores_general(df_experiments, df_expt_robust))

    scores.update(_compute_scores_state_tissue(df_experiments))

    return scores


def _compute_scores_general(df_experiments, df_expt_robust):
    # parse specific metrics
    scores = {
        'Average-Robustness': np.mean(df_experiments[ImRegBenchmark.COL_ROBUSTNESS]),
        'STD-Robustness': np.std(df_experiments[ImRegBenchmark.COL_ROBUSTNESS]),
        'Median-Robustness': np.median(df_experiments[ImRegBenchmark.COL_ROBUSTNESS]),
        'Average-Rank-Median-rTRE': np.nan,
        'Average-Rank-Max-rTRE': np.nan,
    }
    # parse Mean & median specific measures
    for name, col in [('Median-rTRE', 'rTRE Median'),
                      ('Max-rTRE', 'rTRE Max'),
                      ('Average-rTRE', 'rTRE Mean'),
                      ('Norm-Time', COL_NORM_TIME)]:
        for df, sufix in [(df_experiments, ''), (df_expt_robust, '-Robust')]:
            scores['Average-' + name + sufix] = np.nanmean(df[col])
            scores['STD-' + name + sufix] = np.nanstd(df[col])
            scores['Median-' + name + sufix] = np.median(df[col])
    return scores


def _compute_scores_state_tissue(df_experiments):
    scores = {}
    if ImRegBenchmark.COL_STATUS not in df_experiments.columns:
        logging.warning('experiments (table) is missing "%s" column', ImRegBenchmark.COL_STATUS)
        df_experiments[ImRegBenchmark.COL_STATUS] = 'any'
    # filter all statuses in the experiments
    statuses = df_experiments[ImRegBenchmark.COL_STATUS].unique()
    # parse metrics according to TEST and TRAIN case
    for name, col in [('Average-rTRE', 'rTRE Mean'),
                      ('Median-rTRE', 'rTRE Median'),
                      ('Max-rTRE', 'rTRE Max'),
                      ('Robustness', 'Robustness')]:
        # iterate over common measures
        for stat_name, stat_func in [('Average', np.mean),
                                     ('Median', np.median)]:
            _sname = stat_name + '-' + name
            for status in statuses:
                df_expt_ = df_experiments[df_experiments[ImRegBenchmark.COL_STATUS] == status]
                scores[_sname + '_' + status] = stat_func(df_expt_[col])
            # parse according to Tissue
            for tissue, dfg_tissue in df_experiments.groupby(COL_TISSUE):
                scores[_sname + '_tissue_' + tissue] = stat_func(dfg_tissue[col])
                # also per state in tissue
                for status in statuses:
                    df_tiss_st_ = dfg_tissue[dfg_tissue[ImRegBenchmark.COL_STATUS] == status]
                    stat = stat_func(df_tiss_st_[col]) if not df_tiss_st_.empty else np.nan
                    scores[_sname + '_' + status + '_tissue_' + tissue] = stat
    return scores


def _filter_tre_measure_columns(df_experiments):
    """ get columns related to TRE measures

    :param DF df_experiments: experiment table
    :return tuple(list(str),list(str)):
    """
    # copy the initial to final for missing
    cols_init = [col for col in df_experiments.columns if re.match(r'(r)?IRE', col)]
    cols_final = [col.replace('IRE', 'TRE') for col in cols_init]
    assert len(cols_final) == len(cols_init), 'columns do not match for future zip'
    return cols_final, cols_init


def export_summary_json(df_experiments, path_experiments, path_output,
                        min_landmarks=1., details=True):
    """ summarize results in particular JSON format

    :param DF df_experiments: experiment DataFrame
    :param str path_experiments: path to experiment folder
    :param str path_output: path to generated results
    :param float min_landmarks: required number of submitted landmarks in range (0, 1),
        match values in COL_PAIRED_LANDMARKS
    :param bool details: exporting case details
    :return str: path to exported results
    """
    if COL_NORM_TIME not in df_experiments.columns:
        df_experiments[COL_NORM_TIME] = np.nan

    # note, we expect that the path starts with tissue and Unix sep "/" is used
    def _get_tissue(cell):
        tissue = cell.split(os.sep)[0]
        return tissue[:tissue.index('_')] if '_' in cell else tissue

    df_experiments[COL_TISSUE] = df_experiments[ImRegBenchmark.COL_POINTS_REF].apply(_get_tissue)

    # export partial results
    cases = list(iterate_mproc_map(parse_landmarks, df_experiments.iterrows(),
                                   desc='Parsing landmarks', nb_workers=1))

    # copy the initial to final for missing
    for col, col2 in zip(*_filter_tre_measure_columns(df_experiments)):
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

    results = {
        'aggregates': scores,
        'cases': dict(cases) if details else 'not exported',
        'computer': comp_exp,
        'submission-time': time.strftime(FORMAT_DATE_TIME, time.gmtime()),
        'required-landmarks': min_landmarks,
    }
    path_json = os.path.join(path_output, NAME_JSON_RESULTS)
    logging.info('exporting JSON results: %s', path_json)
    with open(path_json, 'w') as fp:
        json.dump(results, fp)
    return path_json


def replicate_missing_warped_landmarks(df_experiments, path_dataset, path_experiment):
    """ if some warped landmarks are missing replace the path by initial landmarks

    :param DF df_experiments: experiment table
    :param str path_dataset: path to dataset folder
    :param str path_experiment: path ti user experiment folder
    :return DF: experiment table
    """
    # find empty warped landmarks paths
    missing_mask = df_experiments[ImRegBenchmark.COL_POINTS_MOVE_WARP].isnull()
    if ImRegBenchmark.COL_POINTS_REF_WARP in df_experiments.columns:
        # if there ar elaso target warped landmarks, allow to use them
        missing_mask &= df_experiments[ImRegBenchmark.COL_POINTS_REF_WARP].isnull()
    # for the empty place the initial landmarks
    df_experiments.loc[missing_mask, ImRegBenchmark.COL_POINTS_MOVE_WARP] = \
        df_experiments.loc[missing_mask, ImRegBenchmark.COL_POINTS_MOVE]
    # for the empty place maximal execution time
    df_experiments.loc[missing_mask, ImRegBenchmark.COL_TIME] = \
        df_experiments[ImRegBenchmark.COL_TIME].max()

    count = 0
    # iterate over whole table and check if the path is valid
    for idx, row in df_experiments.iterrows():
        # select refence/moving warped landmarks
        use_move_warp = isinstance(row.get(ImRegBenchmark.COL_POINTS_MOVE_WARP, None), str)
        col_lnds_warp = ImRegBenchmark.COL_POINTS_MOVE_WARP \
            if use_move_warp else ImRegBenchmark.COL_POINTS_REF_WARP
        # extract the CSV path
        path_csv = update_path(row[col_lnds_warp], pre_path=path_experiment)
        if not os.path.isfile(path_csv):
            # if the path is false, put there the initial from dataset
            path_csv = update_path(row[ImRegBenchmark.COL_POINTS_MOVE], pre_path=path_dataset)
            df_experiments.loc[idx, ImRegBenchmark.COL_POINTS_MOVE_WARP] = path_csv
            count += 1

    logging.info('Missing warped landmarks: %i', count)
    return df_experiments


def swap_inverse_experiment(table, allow_inverse):
    """ optional swap of registration results from using warped moving to warped reference

    :param DF table: experiment table
    :param bool allow_inverse: allw swap from using warped moving to warped reference
    :return DF: updated experiment table
    """
    if not allow_inverse:
        return table
    if ImRegBenchmark.COL_POINTS_MOVE_WARP in table.columns:
        filled = table[ImRegBenchmark.COL_POINTS_MOVE_WARP].dropna()
        if len(filled) > 0:
            # everything seems to be fine...
            return table
    logging.warning('Missing target column "%s"', ImRegBenchmark.COL_POINTS_MOVE_WARP)
    if ImRegBenchmark.COL_POINTS_REF_WARP not in table.columns:
        raise ValueError('Missing target column "%s" to swap to'
                         % ImRegBenchmark.COL_POINTS_REF_WARP)
    logging.info('Swapping columns of Moving and Reference landmarks for both - source and warped.')
    col_ref = table[ImRegBenchmark.COL_POINTS_REF].values.tolist()
    col_move = table[ImRegBenchmark.COL_POINTS_MOVE].values.tolist()
    table[ImRegBenchmark.COL_POINTS_REF] = col_move
    table[ImRegBenchmark.COL_POINTS_MOVE] = col_ref
    table[ImRegBenchmark.COL_POINTS_MOVE_WARP] = table[ImRegBenchmark.COL_POINTS_REF_WARP]
    return table


def main(path_experiment, path_table, path_dataset, path_output, path_reference=None,
         path_comp_bm=None, min_landmarks=1., details=True, allow_inverse=False):
    """ main entry point

    :param str path_experiment: path to experiment folder
    :param str path_table: path to assignment file (requested registration pairs)
    :param str path_dataset: path to provided landmarks
    :param str path_output: path to generated results
    :param str|None path_reference: path to the complete landmark collection,
        if None use dataset folder
    :param str|None path_comp_bm: path to reference comp. benchmark
    :param int nb_workers: number of parallel processes
    :param float min_landmarks: required number of submitted landmarks in range (0, 1),
        match values in COL_PAIRED_LANDMARKS
    :param bool details: exporting case details
    :param bool allow_inverse: allow evaluate also inverse transformation,
        warped landmarks from ref to move image
    """

    path_results = os.path.join(path_experiment, ImRegBenchmark.NAME_CSV_REGISTRATION_PAIRS)
    if not os.path.isfile(path_results):
        raise AttributeError('Missing experiments results: %s' % path_results)
    path_reference = path_dataset if not path_reference else path_reference

    # drop time column from Cover which should be empty
    df_overview = pd.read_csv(path_table).drop([ImRegBenchmark.COL_TIME], axis=1, errors='ignore')
    df_overview = _df_drop_unnamed(df_overview)
    # drop Warp* column from Cover which should be empty
    df_overview = df_overview.drop([col for col in df_overview.columns if 'warped' in col.lower()],
                                   axis=1, errors='ignore')
    df_results = pd.read_csv(path_results)
    df_results = _df_drop_unnamed(df_results)
    # df_results.drop(filter(lambda c: 'Unnamed' in c, df_results.columns), axis=1, inplace=True)
    cols_ = list(ImRegBenchmark.COVER_COLUMNS_WRAP) + [ImRegBenchmark.COL_TIME]
    df_results = df_results[[col for col in cols_ if col in df_results.columns]]
    df_experiments = pd.merge(df_overview, df_results, how='left', on=ImRegBenchmark.COVER_COLUMNS)
    df_experiments = swap_inverse_experiment(df_experiments, allow_inverse)
    # df_experiments.drop([ImRegBenchmark.COL_IMAGE_REF_WARP, ImRegBenchmark.COL_POINTS_REF_WARP],
    #                     axis=1, errors='ignore', inplace=True)
    df_experiments.drop(filter(lambda c: 'Unnamed' in c, df_results.columns), axis=1, inplace=True)

    df_experiments = replicate_missing_warped_landmarks(df_experiments, path_dataset, path_experiment)

    normalize_exec_time(df_experiments, path_experiment, path_comp_bm)

    # logging.info('Filter used landmarks.')
    # path_filtered = os.path.join(path_output, FOLDER_FILTER_DATASET)
    # create_folder(path_filtered, ok_existing=True)
    # _filter_lnds = partial(filter_export_landmarks, path_output=path_filtered,
    #                        path_dataset=path_dataset, path_reference=path_reference)
    # for idx, ratio in iterate_mproc_map(_filter_lnds, df_experiments.iterrows(),
    #                                     desc='Filtering', nb_workers=nb_workers):
    #     df_experiments.loc[idx, COL_PAIRED_LANDMARKS] = np.round(ratio, 2)

    logging.info('Compute landmarks statistic.')
    _compute_lnds_stat = partial(ImRegBenchmark.compute_registration_statistic,
                                 df_experiments=df_experiments,
                                 path_dataset=path_dataset,
                                 path_experiment=path_experiment,
                                 path_reference=path_reference)
    # NOTE: this has to run in SINGLE thread so there is SINGLE table instance
    list(iterate_mproc_map(_compute_lnds_stat, df_experiments.iterrows(),
                           desc='Statistic', nb_workers=1))

    name_results, _ = os.path.splitext(os.path.basename(path_results))
    path_results = os.path.join(path_output, name_results + '_NEW.csv')
    logging.debug('exporting CSV results: %s', path_results)
    df_experiments.to_csv(path_results)

    path_json = export_summary_json(df_experiments, path_experiment, path_output,
                                    min_landmarks, details)
    return path_json


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = parse_arg_params(create_parser())
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
