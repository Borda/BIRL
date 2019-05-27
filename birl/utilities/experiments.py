"""
General experiments methods

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import types
import logging
import argparse
import subprocess
import collections
import multiprocessing as mproc
from pathos.multiprocessing import ProcessPool
from functools import wraps

import tqdm
import numpy as np

from birl.utilities.data_io import create_folder, update_path

#: number of available threads on this computer
NB_THREADS = int(mproc.cpu_count())
#: default date-time format
FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
#: default logging tile
FILE_LOGS = 'logging.txt'
#: default logging template - log location/source for logging to file
STR_LOG_FORMAT = '%(asctime)s:%(levelname)s@%(filename)s:%(processName)s - %(message)s'
#: default logging template - date-time for logging to file
LOG_FILE_FORMAT = logging.Formatter(STR_LOG_FORMAT, datefmt="%H:%M:%S")
#: define all types to be assume list like
ITERABLE_TYPES = (list, tuple, types.GeneratorType)


# fixing ImportError: No module named 'copy_reg' for Python2
# if sys.version_info.major == 2:
#     import copy_reg
#
#     def _reduce_method(m):
#         # SOLVING issue: cPickle.PicklingError:
#         #   Can't pickle <type 'instancemethod'>:
#         #       attribute lookup __builtin__.instancemethod failed
#         tp = m.im_class if m.im_self is None else m.im_self
#         return getattr, (tp, m.im_func.func_name)
#
#     copy_reg.pickle(types.MethodType, _reduce_method)


def create_experiment_folder(path_out, dir_name, name='', stamp_unique=True):
    """ create the experiment folder and iterate while there is no available

    :param str path_out: path to the base experiment directory
    :param str name: special experiment name
    :param str dir_name: special folder name
    :param bool stamp_unique: whether add at the end of new folder unique tag

    >>> p_dir = create_experiment_folder('.', 'my_test', stamp_unique=False)
    >>> os.rmdir(p_dir)
    >>> p_dir = create_experiment_folder('.', 'my_test', stamp_unique=True)
    >>> p_dir  # doctest: +ELLIPSIS
    '...my_test_...-...'
    >>> os.rmdir(p_dir)
    """
    assert os.path.isdir(path_out), 'missing base folder "%s"' % path_out
    date = time.gmtime()
    if isinstance(name, str) and name:
        dir_name = '%r_%r' % (dir_name, name)
    # if you require time stamp
    if stamp_unique:
        path_stamp = time.strftime(FORMAT_DATE_TIME, date)
        # prepare experiment path with initial timestamp - now
        path_exp = os.path.join(path_out, '%s_%s' % (dir_name, path_stamp))
        path_created = None
        while not path_created:
            # try to generate new time stamp
            path_stamp_new = time.strftime(FORMAT_DATE_TIME, date)
            # if the new one is different use it; this may over come too long stamps
            if path_stamp != path_stamp_new:
                path_stamp = path_stamp_new
                path_exp = os.path.join(path_out, '%s_%s' % (dir_name, path_stamp))
            logging.warning('particular out folder already exists')
            if path_created is not None:
                path_exp += '-' + str(np.random.randint(0, 100))
            path_created = create_folder(path_exp, ok_existing=False)
    else:
        path_exp = os.path.join(path_out, dir_name)
        path_created = create_folder(path_exp, ok_existing=False)
    logging.info('created experiment folder "%r"', path_created)
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


def string_dict(ds, headline='DICTIONARY:', offset=25):
    """ format the dictionary into a string

    :param dict ds: {str: val} dictionary with parameters
    :param str headline: headline before the printed dictionary
    :param int offset: max size of the string name
    :return str: formatted string

    >>> string_dict({'a': 1, 'b': 2}, 'TEST:', 5)
    'TEST:\\n"a":  1\\n"b":  2'
    """
    template = '{:%is} {}' % offset
    rows = [template.format('"{}":'.format(n), ds[n]) for n in sorted(ds)]
    s = headline + '\n' + '\n'.join(rows)
    return s


def create_basic_parse():
    """ create the basic arg parses

    :return object:

    >>> parser = create_basic_parse()
    >>> type(parser)
    <class 'argparse.ArgumentParser'>
    >>> parse_arg_params(parser)  # doctest: +SKIP
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
    parser.add_argument('--preprocessing', type=str, required=False, nargs='+',
                        help='use some image pre-processing, the other matter',
                        choices=['gray', 'hist-matching'])
    # parser.add_argument('--lock_expt', dest='lock_thread', action='store_true',
    #                     help='whether lock to run experiment in single thread')
    parser.add_argument('--run_comp_benchmark', action='store_true',
                        help='run computation benchmark on the end')
    parser.add_argument('--nb_workers', type=int, required=False, default=1,
                        help='number of registration running in parallel')
    return parser


def update_paths(args, upper_dirs=None, pattern='path'):
    """ find params with not existing paths

    :param dict args: dictionary with all parameters
    :param list(str) upper_dirs: list of keys in parameters
        with item for which only the parent folder must exist
    :param str pattern: patter specifying key with path
    :return list(str): key of missing paths

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
            p = update_path(os.path.dirname(args[k]))
            args[k] = os.path.join(p, os.path.basename(args[k]))
        else:
            args[k] = update_path(args[k])
            p = args[k]
        if not os.path.exists(p):
            logging.warning('missing "%s": %s', k, p)
            missing.append(k)
    return args, missing


def parse_arg_params(parser, upper_dirs=None):
    """ parse all params

    :param parser: object of parser
    :param list(str) upper_dirs: list of keys in parameters
        with item for which only the parent folder must exist
    :return dict: parameters
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


def exec_commands(commands, path_logger=None, timeout=None):
    """ run the given commands in system Command Line

    See refs:

    * https://stackoverflow.com/questions/1996518
    * https://www.quora.com/Whats-the-difference-between-os-system-and-subprocess-call-in-Python

    :param list(str) commands: commands to be executed
    :param str path_logger: path to the logger
    :param int timeout: timeout for max commands length
    :return bool: whether the commands passed

    >>> exec_commands(('ls', 'ls -l'), path_logger='./sample-output.log')
    True
    >>> exec_commands('mv sample-output.log moved-output.log', timeout=10)
    True
    >>> os.remove('./moved-output.log')
    >>> exec_commands('cp sample-output.log moved-output.log')
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
        outputs += [cmd.encode('utf-8')]
        try:
            # os.system(cmd)
            # NOTE: for some reason " in the command makes it crash, e.g with DROP
            cmd_elems = cmd.split()
            cmd_elems[0] = os.path.expanduser(cmd_elems[0])
            outputs += [subprocess.check_output(cmd_elems, **options)]
        except subprocess.CalledProcessError as e:
            logging.exception(cmd)
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


# class NonDaemonPool(ProcessPool):
#     """ We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
#     because the latter is only a wrapper function, not a proper class.
#
#     See: https://github.com/nipy/nipype/pull/2754
#
#     Examples:
#     `PicklingError: Can't pickle <class 'birl.utilities.experiments.NonDaemonProcess'>:
#      it's not found as birl.utilities.experiments.NonDaemonProcess`
#     `PicklingError: Can't pickle <function _wrap_func at 0x03152EF0>:
#      it's not found as birl.utilities.experiments._wrap_func`
#
#     >>> NonDaemonPool(1).map(sum, [(1, )] * 5)
#     [1, 1, 1, 1, 1]
#     """
#     def Process(self, *args, **kwds):
#         proc = super(NonDaemonPool, self).Process(*args, **kwds)
#
#         class NonDaemonProcess(proc.__class__):
#             """Monkey-patch process to ensure it is never daemonized"""
#             @property
#             def daemon(self):
#                 return False
#
#             @daemon.setter
#             def daemon(self, val):
#                 pass
#
#         proc.__class__ = NonDaemonProcess
#         return proc


def iterate_mproc_map(wrap_func, iterate_vals, nb_workers=NB_THREADS, desc=''):
    """ create a multi-porocessing pool and execute a wrapped function in separate process

    :param func wrap_func: function which will be excited in the iterations
    :param list iterate_vals: list or iterator which will ide in iterations,
        if -1 then use all available threads
    :param int nb_workers: number og jobs running in parallel
    :param str|None desc: description for the bar,
        if it is set None, bar is suppressed

    Waiting reply on:

    * https://github.com/celery/billiard/issues/280
    * https://github.com/uqfoundation/pathos/issues/169

    See:

    * https://sebastianraschka.com/Articles/2014_multiprocessing.html
    * https://github.com/nipy/nipype/pull/2754
    * https://medium.com/contentsquare-engineering-blog/multithreading-vs-multiprocessing-in-python-ece023ad55a
    * http://mindcache.me/2015/08/09/python-multiprocessing-module-daemonic-processes-are-not-allowed-to-have-children.html
    * https://medium.com/@bfortuner/python-multithreading-vs-multiprocessing-73072ce5600b

    >>> list(iterate_mproc_map(np.sqrt, range(5), nb_workers=1))  # doctest: +ELLIPSIS
    [0.0, 1.0, 1.41..., 1.73..., 2.0]
    >>> list(iterate_mproc_map(sum, [[0, 1]] * 5, nb_workers=2, desc=None))
    [1, 1, 1, 1, 1]
    >>> list(iterate_mproc_map(max, [(2, 1)] * 5, nb_workers=2, desc=''))
    [2, 2, 2, 2, 2]
    """
    iterate_vals = list(iterate_vals)
    nb_workers = 1 if not nb_workers else int(nb_workers)
    nb_workers = NB_THREADS if nb_workers < 0 else nb_workers

    tqdm_bar = None
    if desc is not None:
        desc = '%r @%i-threads' % (desc, nb_workers)
        tqdm_bar = tqdm.tqdm(total=len(iterate_vals), desc=desc)

    if nb_workers > 1:
        logging.debug('perform parallel in %i threads', nb_workers)
        # Standard mproc.Pool created a demon processes which can be called
        # inside its children, cascade or multiprocessing
        # https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

        # pool = mproc.Pool(nb_workers)
        # pool = NonDaemonPool(nb_workers)
        pool = ProcessPool(nb_workers)
        # pool = Pool(nb_workers)
        mapping = pool.imap
    else:
        logging.debug('perform sequential')
        pool = None
        mapping = map

    for out in mapping(wrap_func, iterate_vals):
        tqdm_bar.update() if tqdm_bar is not None else None
        yield out

    if pool:
        pool.close()
        pool.join()
        pool.clear()

    tqdm_bar.close() if tqdm_bar is not None else None


# from pathos.helpers import mp
# def perform_parallel(func, arg_params, deamonic=False):
#     """ run all processes in  parallel and wait until all is finished
#
#     :param func: the function to be executed
#     :param list arg_params: list or tuple of parameters
#     :return list: list of outputs
#
#     https://sebastianraschka.com/Articles/2014_multiprocessing.html
#
#     ERROR: Failing on Windows with PickleError
#      or is it uses `dill` it terminates on infinite recursive call/load
#
#     >>> perform_parallel(max, [(2, 1)] * 5)
#     [2, 2, 2, 2, 2]
#     >>> power2 = lambda x: np.power(x, 2)
#     >>> perform_parallel(power2, [1] * 5)
#     [1, 1, 1, 1, 1]
#     """
#     # Define an output queue
#     outputs = mp.Queue()
#
#     # define a wrapper which puts function returns to a queue
#     @wraps(func)
#     def _wrap_func(*argv):
#         out = func(*argv)
#         outputs.put(out)
#
#     # in case the parameters are just single variable
#     # arg_params = [argv if is_iterable(argv) else [argv] for argv in arg_params]
#
#     # Setup a list of processes that we want to run
#     processes = [mp.Process(target=_wrap_func, args=(argv, )) for argv in arg_params]
#
#     # Run processes
#     for ps in processes:
#         ps.daemon = deamonic
#         ps.start()
#
#     # Exit the completed processes
#     for ps in processes:
#         ps.join()
#
#     # Get process results from the output queue
#     results = [outputs.get() for _ in processes]
#
#     return results


def try_decorator(func):
    """ costume decorator to wrap function in try/except

    :param func: decorated function
    :return func: output of the decor. function
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logging.exception('%r with %r and %r', func.__name__, args, kwargs)
    return wrap


def is_iterable(var, iterable_types=ITERABLE_TYPES):
    """ check if the variable is iterable

    :param var: tested variable
    :return bool: iterable

    >>> is_iterable('abc')
    False
    >>> is_iterable([0])
    True
    >>> is_iterable((1, ))
    True
    """
    is_iter = any(isinstance(var, cls) for cls in iterable_types)
    return is_iter


def dict_deep_update(dict_base, dict_update):
    """ update recursively

    :param dict dict_base:
    :param dict dict_update:
    :return dict:

    >>> d = {'level1': {'level2': {'levelA': 0, 'levelB': 1}}}
    >>> u = {'level1': {'level2': {'levelB': 10}}}
    >>> import json
    >>> d = json.dumps(dict_deep_update(d, u), sort_keys=True, indent=2)
    >>> print(d)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {
      "level1": {
        "level2": {
          "levelA": 0,
          "levelB": 10
        }
      }
    }
    """
    for k in dict_update:
        val = dict_update[k]
        if isinstance(val, collections.Mapping):
            dict_base[k] = dict_deep_update(dict_base.get(k, dict), val)
        else:
            dict_base[k] = val
    return dict_base
