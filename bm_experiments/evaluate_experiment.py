"""
Evaluating passed experiments produced by this benchmark,
for instance if metric was changed or some visualisations are missing

The expected experiment structure is following:

 * `registration-results.csv` coordinate file with path to landmarks and images
 * particular experiment with warped landmarks

Sample Usage
------------
.. code-block:: bash

    python evaluate_experiment.py \
        -e ./results/BmUnwarpJ \
        -d ./data-images \
        --visual

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import argparse
import logging
import os
import sys
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import iterate_mproc_map, parse_arg_params, nb_workers
from birl.benchmark import ImRegBenchmark, export_summary_results

#: default number of used threads
NB_WORKERS = nb_workers(0.75)
#: file name of new table with registration results
NAME_CSV_RESULTS = 'registration-results_NEW.csv'
#: file name of new table with registration summary
NAME_CSV_SUMMARY = 'results-summary_NEW.csv'
#: file with formatted registration summary
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
    parser.add_argument('--nb_workers', type=int, required=False, default=NB_WORKERS,
                        help='number of processes running in parallel')
    return parser


def main(path_experiment, path_dataset, visual=False, nb_workers=NB_WORKERS):
    """ main entry points

    :param str path_experiment: path to the experiment folder
    :param str path_dataset: path to the dataset with all landmarks
    :param bool visual: whether visualise the registration results
    :param int nb_workers: number of parallel jobs
    """
    path_results = os.path.join(path_experiment, ImRegBenchmark.NAME_CSV_REGISTRATION_PAIRS)
    assert os.path.isfile(path_results)

    df_experiments = pd.read_csv(path_results)
    df_results = df_experiments.copy()
    _compute_lnds_stat = partial(ImRegBenchmark.compute_registration_statistic,
                                 df_experiments=df_results,
                                 path_dataset=path_dataset,
                                 path_experiment=path_experiment)
    # NOTE: this has to run in SINGLE thread so there is SINGLE table instance
    list(iterate_mproc_map(_compute_lnds_stat, df_experiments.iterrows(),
                           desc='Statistic', nb_workers=1))

    path_csv = os.path.join(path_experiment, NAME_CSV_RESULTS)
    logging.debug('exporting CSV results: %s', path_csv)
    df_results.to_csv(path_csv, index=None)
    export_summary_results(df_results, path_experiment, None,
                           name_csv=NAME_CSV_SUMMARY,
                           name_txt=NAME_TXT_SUMMARY)

    if visual:
        _visualise_regist = partial(ImRegBenchmark.visualise_registration,
                                    path_dataset=path_dataset,
                                    path_experiment=path_experiment)
        list(iterate_mproc_map(_visualise_regist, df_experiments.iterrows(),
                               desc='Visualisation', nb_workers=nb_workers))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = parse_arg_params(create_parser())
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
