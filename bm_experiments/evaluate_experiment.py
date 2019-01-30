"""
Evaluating passed experiments, for instance if metric was changed

EXAMPLE
-------
>> python evaluate_experiment.py \
    -e results/BmUnwarpJ \
    -c data_images/pairs-imgs-lnds_mix.csv \
    -d data_images \
    -t data_images \
    --visual


Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import multiprocessing as mproc

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.experiments import wrap_execute_sequence, parse_arg_params
from benchmark.cls_benchmark import (NAME_CSV_REGISTRATION_PAIRS, COVER_COLUMNS,
                                     compute_landmarks_statistic)

NB_THREADS = int(mproc.cpu_count() * .5)


def arg_parse_params():
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
    parser.add_argument('-t', '--path_landmarks', type=str, required=False,
                        help='path to complete ground truth landmarks')
    parser.add_argument('--visual', action='store_true', required=False,
                        default=False, help='visualise the landmarks in images')
    parser.add_argument('--nb_jobs', type=int, required=False, default=NB_THREADS,
                        help='number of processes running in parallel')
    args = parse_arg_params(parser)
    return args


# TODO


def main(path_experiment, path_cover, path_dataset, path_landmarks=None, visual=False, nb_jobs=1):
    path_results = os.path.join(path_experiment, NAME_CSV_REGISTRATION_PAIRS)
    assert os.path.isfile(path_results)
    path_landmarks = path_dataset if path_landmarks is None else path_dataset

    df_cover = pd.read_csv(path_cover)
    df_results = pd.read_csv(path_results)
    df_experiments = pd.merge(df_cover, df_results, how='left', on=COVER_COLUMNS)

    list(wrap_execute_sequence(compute_landmarks_statistic, df_experiments.iterrows(),
                               desc='Statistic', nb_jobs=nb_jobs))

    print ()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    main(**arg_params)

    logging.info('DONE')
