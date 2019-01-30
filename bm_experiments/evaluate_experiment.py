"""
Evaluating passed experiments, for instance if metric was changed

EXAMPLE
-------
>> python evaluate_experiment.py \
    -e ./results/BmUnwarpJ \
    --visual

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.experiments import wrap_execute_sequence, parse_arg_params
from benchmark.cls_benchmark import (NAME_CSV_REGISTRATION_PAIRS, export_summary,
                                     compute_landmarks_statistic, visualise_registration)

NB_THREADS = int(mproc.cpu_count() * .5)
NAME_CSV_RESULTS = 'registration-results_NEW.csv'
NAME_CSV_SUMMARY = 'results-summary_NEW.csv'
NAME_TXT_SUMMARY = 'results-summary_NEW.txt'


def create_parser():
    """ parse the input parameters
    :return dict: {str: any}
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--path_experiment', type=str, required=True,
                        help='path to the experiments')
    parser.add_argument('-d', '--path_dataset', type=str, required=False,
                        help='path to dataset with provided landmarks')
    parser.add_argument('--visual', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False, default=NB_THREADS,
                        help='number of processes running in parallel')
    return parser


def main(path_experiment, path_dataset, visual=False, nb_jobs=1):
    path_results = os.path.join(path_experiment, NAME_CSV_REGISTRATION_PAIRS)
    assert os.path.isfile(path_results)

    df_experiments = pd.read_csv(path_results)
    df_results = df_experiments.copy()
    _compute_lnds_stat = partial(compute_landmarks_statistic, df_experiments=df_results,
                                 path_dataset=path_dataset, path_experiment=path_experiment)

    list(wrap_execute_sequence(_compute_lnds_stat, df_experiments.iterrows(),
                               desc='Statistic', nb_jobs=nb_jobs))

    path_csv = os.path.join(path_experiment, NAME_CSV_RESULTS)
    logging.debug('exporting CSV results: %s', path_csv)
    df_results.to_csv(path_csv)
    export_summary(df_results, path_experiment, None, name_csv=NAME_CSV_SUMMARY,
                   name_txt=NAME_TXT_SUMMARY)

    if visual:
        _visualise_regist = partial(visualise_registration, path_dataset=path_dataset,
                                    path_experiment=path_experiment)
        list(wrap_execute_sequence(_visualise_regist, df_experiments.iterrows(),
                                   desc='Visualisation', nb_jobs=nb_jobs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = parse_arg_params(create_parser())
    main(**arg_params)

    logging.info('DONE')
