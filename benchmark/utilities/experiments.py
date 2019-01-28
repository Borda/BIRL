"""
General experiments methods

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import time
import logging
import argparse
import subprocess
import multiprocessing.pool
import multiprocessing as mproc
from functools import wraps

import tqdm
import numpy as np

import benchmark.utilities.data_io as tl_io

NB_THREADS = mproc.cpu_count()
FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
FILE_LOGS = 'logging.txt'
STR_LOG_FORMAT = '%(asctime)s:%(levelname)s@%(filename)s:%(processName)s - %(message)s'
LOG_FILE_FORMAT = logging.Formatter(STR_LOG_FORMAT, datefmt="%H:%M:%S")


def create_experiment_folder(path_out, dir_name, name='', stamp_unique=True):
    """ create the experiment folder and iterate while there is no available

    :param str path_out: path to the base experiment directory
    :param str name: special experiment name
    :param str dir_name: special folder name
    :param bool stamp_unique: whether add at the end of new folder unique tag

    >>> from functools import partial
    >>> create = partial(create_experiment_folder, dir_name='my_test', stamp_unique=True)
    >>> dirs = list(wrap_execute_sequence(create, ['.'] * 10))
    >>> all(map(os.path.isdir, dirs))
    True
    >>> _= [os.rmdir(p) for p in set(dirs)]
    """
    assert os.path.exists(path_out), 'missing "%s"' % path_out
    date = time.gmtime()
    if isinstance(name, str) and name:
        dir_name = '{}_{}'.format(dir_name, name)
    if stamp_unique:
        dir_name += '_' + time.strftime(FORMAT_DATE_TIME, date)
    path_exp = os.path.join(path_out, dir_name)
    while stamp_unique and os.path.isdir(path_exp):
        logging.warning('particular out folder already exists')
        path_exp += ':' + str(np.random.randint(0, 100))
    logging.info('creating experiment folder "{}"'.format(path_exp))
    path_exp = tl_io.create_folder(path_exp)
    return path_exp


def release_logger_files():
    """ close all handlers to a file

    >>> release_logger_files()
    >>> len([1 for lh in logging.getLogger().handlers
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
    >>> len([1 for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    1
    >>> release_logger_files()
    >>> os.remove(FILE_LOGS)
    """
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    if reset:
        release_logger_files()
    path_logger = os.path.join(path_out, file_name)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(LOG_FILE_FORMAT)
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
    parser.add_argument('-c', '--path_cover', type=str, required=True,
                        help='path to the csv cover file')
    parser.add_argument('-d', '--path_dataset', type=str, required=False,
                        help='path to the dataset location, '
                             'if missing in cover', default=None)
    parser.add_argument('-o', '--path_out', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--unique', dest='unique', action='store_true',
                        help='whether each experiment have unique time stamp')
    parser.add_argument('--visual', dest='visual', action='store_true',
                        help='whether visualise partial results')
    parser.add_argument('--lock_expt', dest='lock_thread', action='store_true',
                        help='whether lock to run experiment in single therad')
    parser.add_argument('--nb_jobs', type=int, required=False, default=1,
                        help='number of registration running in parallel')
    return parser


def update_paths(args, upper_dirs=None, pattern='path'):
    """ find params with not existing paths

    :param {} args: dictionary with all parameters
    :param [str] upper_dirs: list of keys in parameters
        with item for which only the parent folder must exist
    :param str pattern: patter specifying key with path
    :return [str]: key of missing paths

    >>> update_paths({'sample': 123})[1]
    []
    >>> update_paths({'path_': '.'})[1]
    []
    >>> params = {'path_out': './nothing'}
    >>> update_paths(params)[1]
    ['path_out']
    >>> update_paths(params, upper_dirs=['path_out'])[1]
    []
    """
    if upper_dirs is None:
        upper_dirs = []
    missing = []
    for k in (k for k in args if pattern in k):
        if '*' in os.path.basename(args[k]) or k in upper_dirs:
            p = tl_io.update_path(os.path.dirname(args[k]))
            args[k] = os.path.join(p, os.path.basename(args[k]))
        else:
            args[k] = tl_io.update_path(args[k])
            p = args[k]
        if not os.path.exists(p):
            logging.warning('missing "%s": %s', k, p)
            missing.append(k)
    return args, missing


def parse_arg_params(parser, upper_dirs=None):
    """ parse all params

    :param parser: object of parser
    :param [str] upper_dirs: list of keys in parameters
        with item for which only the parent folder must exist
    :return {str: any}:

    >>> args = create_basic_parse()
    >>> parse_arg_params(args)  # doctest: +SKIP
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    args = vars(parser.parse_args())
    logging.info('ARGUMENTS: \n %r', args)
    # remove all None parameters
    args = {k: args[k] for k in args if args[k] is not None}
    # extend and test all paths in params
    args, missing = update_paths(args, upper_dirs=upper_dirs)
    assert not missing, 'missing paths: %r' % {k: args[k] for k in missing}
    return args


def run_command_line(commands, path_logger=None, timeout=None):
    """ run the given commands in system Command Line

    SEE: https://stackoverflow.com/questions/1996518
    https://www.quora.com/Whats-the-difference-between-os-system-and-subprocess-call-in-Python

    :param [str] commands: commands to be executed
    :param str path_logger: path to the logger
    :param int timeout: timeout for max commands length
    :return bool: whether the commands passed

    >>> run_command_line(('ls', 'ls -l'), path_logger='./sample-output.log')
    True
    >>> run_command_line('mv sample-output.log moved-output.log', timeout=10)
    True
    >>> os.remove('./moved-output.log')
    >>> run_command_line('cp sample-output.log moved-output.log')
    False
    """
    logging.debug('CMD ->> \n%s', commands)
    options = dict(stderr=subprocess.STDOUT)
    # timeout in check_output is not supported by Python 2
    if timeout is not None and timeout > 0 and sys.version_info.major >= 3:
        options['timeout'] = timeout
    if isinstance(commands, str):
        commands = [commands]
    outputs = []
    success = True
    # try to execute all commands in stack
    for cmd in commands:
        try:
            outputs += [subprocess.check_output(cmd.split(), **options)]
        except subprocess.CalledProcessError as e:
            logging.exception(commands)
            outputs += [e.output]
            success = False
    # export the output if path exists
    if path_logger is not None and outputs:
        if isinstance(outputs[0], bytes):
            outputs = [out.decode() for out in outputs]
        elif isinstance(outputs[0], str):
            outputs = [out.decode().encode('utf-8') for out in outputs]
        with open(path_logger, 'a') as fp:
            fp.write('\n'.join(outputs))
    return success


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
    >>> all(stat[k] == 0 for k in stat if k not in ['Overlap'])
    True
    >>> dist, stat = compute_points_dist_statistic(points1, points2)
    >>> dist  #doctest: +ELLIPSIS
    array([ 2.828...,  3.162...,  1.414...])
    >>> stat['Mean']  #doctest: +ELLIPSIS
    2.468...
    """
    lnd_sizes = [len(points1), len(points2)]
    nb_common = min(lnd_sizes)
    assert nb_common > 0, 'no common landamrks for metric'
    points1 = np.asarray(points1)[:nb_common]
    points2 = np.asarray(points2)[:nb_common]
    diffs = np.sqrt(np.sum(np.power(points1 - points2, 2), axis=1))
    dict_stat = {
        'Mean': np.mean(diffs),
        'STD': np.std(diffs),
        'Median': np.median(diffs),
        'Min': np.min(diffs),
        'Max': np.max(diffs),
        'Overlap': nb_common / float(max(lnd_sizes))
    }
    return diffs, dict_stat


class NonDaemonPool(multiprocessing.pool.Pool):
    """ We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    because the latter is only a wrapper function, not a proper class.

    See: https://github.com/nipy/nipype/pull/2754

    FIXME: fails on Windows

    >>> NonDaemonPool(1)  # doctest: +SKIP
    """
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc


def wrap_execute_sequence(wrap_func, iterate_vals, nb_jobs=NB_THREADS,
                          desc='', ordered=False):
    """ wrapper for execution parallel of single thread as for...

    :param wrap_func: function which will be excited in the iterations
    :param [] iterate_vals: list or iterator which will ide in iterations
    :param int nb_jobs: number og jobs running in parallel
    :param str|None desc: description for the bar,
        if it is set None, bar is suppressed
    :param bool ordered: whether enforce ordering in the parallelism

    >>> list(wrap_execute_sequence(np.sqrt, range(5), nb_jobs=1, ordered=True))  # doctest: +ELLIPSIS
    [0.0, 1.0, 1.41..., 1.73..., 2.0]
    >>> list(wrap_execute_sequence(sum, [[0, 1]] * 5, nb_jobs=2, desc=None))
    [1, 1, 1, 1, 1]
    """
    iterate_vals = list(iterate_vals)

    if desc is not None:
        desc = '%s @%i-threads' % (desc, nb_jobs)
        tqdm_bar = tqdm.tqdm(total=len(iterate_vals), desc=desc)
    else:
        tqdm_bar = None

    if nb_jobs > 1:
        logging.debug('perform parallel in %i threads', nb_jobs)
        # Standard mproc.Pool created a demon processes which can be called
        # inside its children, cascade or multiprocessing
        # https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
        pool = mproc.Pool(nb_jobs)
        pooling = pool.imap if ordered else pool.imap_unordered

        for out in pooling(wrap_func, iterate_vals):
            yield out
            if tqdm_bar is not None:
                tqdm_bar.update()
        pool.close()
        pool.join()
    else:
        logging.debug('perform sequential')
        for out in map(wrap_func, iterate_vals):
            yield out
            if tqdm_bar is not None:
                tqdm_bar.update()


def try_decorator(func):
    """ costume decorator to wrap function in try/except

    :param func:
    :return:
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logging.exception('%r with %r and %r', func.__name__, args, kwargs)
    return wrap
