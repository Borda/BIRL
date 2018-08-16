"""
General experiments methods


Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import time
import random
import logging
import argparse
import traceback

import numpy as np

import benchmark.utilities.data_io as tl_io

FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
FILE_LOGS = 'logging.txt'


def create_experiment_folder(path_out, dir_name, name='', stamp_unique=True):
    """ create the experiment folder and iterate while there is no available

    :param str path_out: path to the base experiment directory
    :param str name: special experiment name
    :param str dir_name: special folder name
    :param bool stamp_unique: whether add at the end of new folder unique tag

    >>> p = create_experiment_folder('.', 'my_test', stamp_unique=False)
    >>> os.path.exists(p)
    True
    >>> os.rmdir(p)

    """
    assert os.path.exists(path_out), '%s' % path_out
    date = time.gmtime()
    if isinstance(name, str) and len(name) > 0:
        dir_name = '{}_{}'.format(dir_name, name)
    if stamp_unique:
        dir_name += '_' + time.strftime(FORMAT_DATE_TIME, date)
    path_exp = os.path.join(path_out, dir_name)
    while stamp_unique and os.path.exists(path_exp):
        logging.warning('particular out folder already exists')
        path_exp += ':' + str(random.randint(0, 9))
    logging.info('creating experiment folder "{}"'.format(path_exp))
    path_exp = tl_io.create_dir(path_exp)
    return path_exp


def release_logger_files():
    """ close all handlers to a file

    >>> release_logger_files()
    >>> len([lh for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    0
    """
    for hl in logging.getLogger().handlers:
        if isinstance(hl, logging.FileHandler):
            hl.close()
            logging.getLogger().removeHandler(hl)


def set_experiment_logger(path_out, file_name=FILE_LOGS, reset=True):
    """ set the logger to file

    :param str path_out: path to the output folder
    :param str file_name: log file name
    :param bool reset: reset all previous logging into a file

    >>> set_experiment_logger('.')
    >>> len([lh for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    1
    >>> release_logger_files()
    >>> os.remove(FILE_LOGS)
    """
    log = logging.getLogger()
    # log.setLevel(logging.DEBUG)
    if reset:
        release_logger_files()
    path_logger = os.path.join(path_out, file_name)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s@%(filename)s:%(processName)s - %(message)s',
        datefmt="%H:%M:%S"))
    log.addHandler(fh)


def string_dict(d, headline='DICTIONARY:', offset=25):
    """ format the dictionary into a string

    :param dict d: {str: val} dictionary with parameters
    :param str headline: headline before the printed dictionary
    :param int offset: max size of the string name
    :return str: formatted string

    >>> string_dict({'a': 1, 'b': 2}, 'TEST:', 5)
    'TEST:\\n"a":  1\\n"b":  2'

    """
    template = '{:%is} {}' % offset
    rows = [template.format('"{}":'.format(n), d[n]) for n in sorted(d)]
    s = headline + '\n' + '\n'.join(rows)
    return s


def create_basic_parse():
    """ create the basic arg parses

    :return object:

    >>> parser = create_basic_parse()
    >>> type(parser)
    <class 'argparse.ArgumentParser'>
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_cover', type=str, required=True,
                        help='path to the csv cover file')
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--unique', dest='unique', action='store_true',
                        help='whether each experiment have unique time stamp')
    parser.add_argument('--visual', dest='unique', action='store_true',
                        help='whether visualise partial results')
    parser.add_argument('--lock_expt', dest='lock_thread', action='store_true',
                        help='whether lock to run experiment in single therad')
    parser.add_argument('--nb_jobs', type=int, required=False, default=1,
                        help='number of registration running in parallel')
    return parser


def check_paths(args, restrict_dir=None):
    """

    :param {} args:
    :param [str] restrict_dir:

    >>> check_paths({'sample': 123})
    True
    >>> check_paths({'path_': '.'})
    True
    >>> check_paths({'path_out': './nothing'}, restrict_dir=['path_out'])
    True
    >>> check_paths({'path_out': './nothing'})  # doctest: +ELLIPSIS
    False
    """
    if restrict_dir is None:
        restrict_dir = []
    status = True
    for k in (k for k in args if 'path' in k):
        if '*' in os.path.basename(args[k]) or k in restrict_dir:
            p = tl_io.update_path(os.path.dirname(args[k]))
            args[k] = os.path.join(p, os.path.basename(args[k]))
        else:
            args[k] = tl_io.update_path(args[k])
            p = args[k]
        if not os.path.exists(p):
            logging.error('missing "%s"' % p)
            status = False
    return status


def parse_params(parser):
    """ parse all params

    :param parser: object of parser
    :return: {str: any}, int

    >>> args = create_basic_parse()
    >>> parse_params(args)  # doctest: +SKIP
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    args = vars(parser.parse_args())
    logging.debug('ARG PARAMS: \n %s', repr(args))
    # remove all None parameters
    args = {k: args[k] for k in args if args[k] is not None}
    # extend nd test all paths in params
    assert check_paths(args), 'missing paths: %s' % repr(args)
    return args


def run_command_line(cmd, path_logger=None):
    """ run the given command in system Command Line

    :param str cmd: command to be executed
    :param str path_logger: path to the logger
    :return bool: whether the command passed

    >>> run_command_line('cd .')
    True
    """
    logging.debug('CMD -> \n%s', cmd)
    if path_logger is not None and not os.path.exists(path_logger):
        cmd += " >> " + path_logger
    try:
        # TODO: out = subprocess.call(cmd, timeout=TIMEOUT, shell=True)
        os.system(cmd)
        return True
    except Exception:
        logging.error(traceback.format_exc())
        return False


def compute_points_dist_statistic(points1, points2):
    """ compute distance as between related points in two sets
    and make a statistic on those distances - mean, std, median, min, max

    :param points1: np.array<nb_points, dim>
    :param points2: np.array<nb_points, dim>
    :return: (np.array<nb_points, 1>, {str: float})

    >>> points1 = np.array([[1, 2], [3, 4], [2, 1]])
    >>> points2 = np.array([[3, 4], [2, 1], [1, 2]])
    >>> dist, stat = compute_points_dist_statistic(points1, points1)
    >>> dist
    array([ 0.,  0.,  0.])
    >>> all(v == 0 for v in stat.values())
    True
    >>> dist, stat = compute_points_dist_statistic(points1, points2)
    >>> dist
    array([ 2.82842712,  3.16227766,  1.41421356])
    >>> stat['Mean']
    2.4683061157625548
    """
    assert points1.shape == points2.shape, \
        'points sizes do not match %s != %s' \
        % (repr(points1.shape), repr(points2.shape))
    diffs = np.sqrt(np.sum(np.power(points1 - points2, 2), axis=1))
    dict_stat = {
        'Mean': np.mean(diffs),
        'STD': np.std(diffs),
        'Median': np.median(diffs),
        'Min': np.min(diffs),
        'Max': np.max(diffs),
    }
    return diffs, dict_stat
