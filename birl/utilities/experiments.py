"""
General experiments methods.

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import copy
import time
import types
import logging
import argparse
import subprocess
import collections
import platform
import uuid
import multiprocessing as mproc
from functools import wraps

import tqdm
import numpy as np
from pathos.multiprocessing import ProcessPool

from birl.utilities.data_io import create_folder, save_config_yaml, update_path
from birl.utilities.dataset import CONVERT_RGB

#: number of available CPUs on this computer
CPU_COUNT = int(mproc.cpu_count())
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


def nb_workers(ratio):
    """Given usage `ratio` return nb of cpu to use."""
    return max(1, int(CPU_COUNT * ratio))


class Experiment(object):
    """
    Tha basic template for experiment running with specific initialisation.
    None, all required parameters used in future have to come in init phase.

    The workflow is following:

    1. prepare experiment folder, copy configurations
    2. `._prepare()` prepares experiment according its specification
    3. `._load_data()` loads required input data and annotations
    4. `._run()` performs the experimental body, run the method on input data
    5. `._summarise()` evaluates result against annotation and summarize
    6. terminate the experimnt

    Particular specifics:

    * each experiment is created in own/folder (if timestamp required)
    * at the beginning experiment configs are copied to the folder
    * logging: INFO level is used for console and DEBUG to file
    * if several sources can be processed independently, you may parallelize it

    Example
    -------
    >>> import birl.utilities.data_io as tl_io
    >>> path_out = tl_io.create_folder('output')
    >>> params = {'path_out': path_out, 'name': 'my_Experiment'}
    >>> expt = Experiment(params, False)
    >>> 'path_exp' in expt.params
    True
    >>> expt.run()
    True
    >>> del expt
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: default output file for exporting experiment configuration
    NAME_CONFIG_YAML = 'config.yml'
    #: default file for exporting results in formatted text format
    NAME_RESULTS_TXT = 'results.txt'
    #: default file for exporting results in table format
    NAME_RESULTS_CSV = 'results.csv'
    #: required experiment parameters
    REQUIRED_PARAMS = ['path_out']

    def __init__(self, exp_params, stamp_unique=True):
        """Initialise the experiment, create experiment folder and set logger.

        :param dict exp_params: experiment configuration {str: value}
        :param bool stamp_unique: add at the end of experiment folder unique
            time stamp (actual date and time)
        """
        self._main_thread = True
        self.params = copy.deepcopy(exp_params)
        self.params['class'] = self.__class__.__name__
        self._check_required_params()
        self.__check_exist_path()
        self.__create_folder(stamp_unique)
        set_experiment_logger(self.params['path_exp'], FILE_LOGS)
        # set stream logging to info level
        for lh in logging.getLogger().handlers:
            if isinstance(lh, logging.StreamHandler) and \
                    not isinstance(lh, logging.FileHandler):
                lh.setLevel(logging.INFO)
        logging.info('initialise experiment...')
        logging.info(string_dict(self.params, 'PARAMETERS:'))
        logging.info('COMPUTER: %r', computer_info())

    def _check_required_params(self):
        """Check some extra required parameters for this experiment."""
        logging.debug('.. check if Experiment have all required parameters')
        for n in self.REQUIRED_PARAMS:
            assert n in self.params, 'missing "%s" among %r' % (n, self.params.keys())

    def run(self):
        """Running the complete experiment.

        This ain method consist of following steps:

        1. `_prepare()` prepares experiment, some extra procedures if needed
        2. `_load_data()` loads required input data (and annotations)
        3. `_run()` performs the experimented method on input data
        4. `_summarise()` evaluates result against annotation and summarize

        .. note:: all the particular procedures are empty and has to be completed
         according to specification of the experiment (do some extra preparation
         like copy extra configs, define how to load the data, perform custom method,
         summarise results with ground truth / annotation)
        """
        logging.info('running experiment...')
        self._prepare()
        self._load_data()
        self._run()
        self._evaluate()
        self._summarise()
        return True

    @classmethod
    def _prepare(self):
        """Prepare the experiment folder."""
        logging.warning('-> preparing EMPTY experiments...')

    @classmethod
    def _load_data(self):
        """Loading data - source and annotations."""
        logging.warning('-> loading EMPTY data...')

    @classmethod
    def _run(self):
        """Perform experiment itself with given method and source data."""
        logging.warning('-> perform EMPTY experiment...')

    @classmethod
    def _evaluate(self):
        """Evaluate experiment - prediction & annotation."""
        logging.warning('-> evaluate EMPTY experiment...')

    @classmethod
    def _summarise(self):
        """Summarise experiment result against annotation."""
        logging.warning('-> summarise EMPTY experiment...')

    def __check_exist_path(self):
        """Check existence of all paths in parameters.

        check existence of all parameters dictionary which has contains words:
        'path', 'dir', 'file'
        """
        assert 'path_out' in self.params, 'missing "path_out" among parameters'
        self.params['path_out'] = update_path(self.params.get('path_out'))
        list_names = [n for n in self.params
                      if any(m in n.lower() for m in ['path', 'dir', 'file'])]
        for n in list_names:
            p = os.path.abspath(os.path.expanduser(self.params[n]))
            if not os.path.exists(p):
                raise Exception('given path/file/dir "%s" does not exist!' % p)
            self.params[n] = p
        for n in [n for n in self.params if 'exec' in n]:
            # in case you define executable in your home
            if os.path.expanduser(self.params[n]) != self.params[n]:
                self.params[n] = os.path.expanduser(self.params[n])

    def __create_folder(self, stamp_unique=True):
        """Create the experiment folder (iterate if necessary).

        * create unique folder if timestamp is requested
        * export experiment configuration to the folder
        """
        assert 'path_out' in self.params, 'missing "path_out" among %r' \
                                          % self.params.keys()
        # create results folder for experiments
        path_exp = create_experiment_folder(
            self.params.get('path_out'), self.__class__.__name__,
            self.params.get('name'), stamp_unique)
        self.params['path_exp'] = path_exp
        save_config_yaml(os.path.join(path_exp, self.NAME_CONFIG_YAML), self.params)

    def __del__(self):
        """Terminating experiment.

        close the logger if the termination instance is the main one
        """
        if self and hasattr(self, '_main_thread') and self._main_thread:
            logging.info('terminating experiment...')
            release_logger_files()
        logging.debug('terminating child experiment...')


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
        dir_name = '%s_%s' % (dir_name, name)
    # if you require time stamp
    if stamp_unique:
        path_stamp = time.strftime(FORMAT_DATE_TIME, date)
        # prepare experiment path with initial timestamp - now
        path_exp = os.path.join(path_out, '%s_%s' % (dir_name, path_stamp))
        if os.path.isdir(path_exp):
            logging.warning('particular out folder already exists')
            path_exp += '-' + str(uuid.uuid4().hex)
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


def create_basic_parser(name=''):
    """ create the basic arg parses

    :param str name: name of the methods
    :return object:

    >>> parser = create_basic_parser()
    >>> type(parser)
    <class 'argparse.ArgumentParser'>
    >>> parse_arg_params(parser)  # doctest: +SKIP
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser('Benchmark on Image Registration - %s' % name)
    parser.add_argument('-n', '--name', type=str, required=False, default=None,
                        help='custom experiment name')
    parser.add_argument('-t', '--path_table', type=str, required=True,
                        help='path to the csv cover file')
    parser.add_argument('-d', '--path_dataset', type=str, required=False, default=None,
                        help='path to the dataset location, if missing in table')
    parser.add_argument('-o', '--path_out', type=str, required=True,
                        help='path to the output directory')
    parser.add_argument('--unique', dest='unique', action='store_true',
                        help='whether each experiment have unique time stamp')
    parser.add_argument('--visual', dest='visual', action='store_true',
                        help='whether visualise partial results')
    parser.add_argument('-pproc', '--preprocessing', type=str, required=False, nargs='+',
                        help='use some image pre-processing, the other matter',
                        choices=['gray'] + ['matching-%s' % clr for clr in CONVERT_RGB])
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

    .. note:: Timeout in check_output is not supported by Python 2.x

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
        # except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ex:
        except subprocess.CalledProcessError as ex:
            logging.exception(ex)
            outputs += [ex.output]
            success = False
        # assume to be subprocess.TimeoutExpired
        except Exception as ex:
            # catching this exception directly is not possible because Py2 does not know it
            if hasattr(ex, 'timeout'):
                logging.warning('subprocess.TimeoutExpired:'
                                ' Command "%s" timed out after %i seconds', cmd, ex.timeout)
                outputs += [ex.output]
            else:
                logging.exception(ex)
            success = False
    # export the output if path exists
    if path_logger is not None and outputs:
        outputs_str = []
        for out in outputs:
            # convert output to string
            out = out.decode("utf-8") if isinstance(outputs[0], bytes) \
                else out.decode().encode('utf-8')
            outputs_str.append(out)
        with open(path_logger, 'a') as fp:
            fp.write('\n\n'.join(outputs_str))
    return success


# class NoDaemonProcess(mp.Process):
#     """ `pathos` pools are wrappers around multiprocess pools.
#     That's the raw `multiprocess.Pool` object without the pathos interface wrapper.
#
#     See: https://github.com/uqfoundation/pathos/issues/169
#     """
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#
#     def _set_daemon(self, value):
#         pass
#
#     daemon = property(_get_daemon, _set_daemon)
#
#
# class NoDaemonProcessPool(ProcessPool):
#     """ The raw `multiprocess.Pool` object without the pathos interface wrapper.
#
#     See: https://github.com/uqfoundation/pathos/issues/169
#
#     Crashing in CI
#
#     >>> NoDaemonProcessPool(2).map(sum, [(1, )] * 5)
#     [1, 1, 1, 1, 1]
#     >>> list(NoDaemonProcessPool(1).imap(lambda x: x ** 2, range(5)))
#     [0, 1, 4, 9, 16]
#     >>> NoDaemonProcessPool(2).map(sum, NoDaemonProcessPool(2).imap(lambda x: x, [range(3)] * 5))
#     [3, 3, 3, 3, 3]
#     """
#     Process = NoDaemonProcess


def iterate_mproc_map(wrap_func, iterate_vals, nb_workers=CPU_COUNT, desc='', ordered=True):
    """ create a multi-porocessing pool and execute a wrapped function in separate process

    :param func wrap_func: function which will be excited in the iterations
    :param list iterate_vals: list or iterator which will ide in iterations,
        if -1 then use all available threads
    :param int nb_workers: number og jobs running in parallel
    :param str|None desc: description for the bar,
        if it is set None, bar is suppressed
    :param bool ordered: whether enforce ordering in the parallelism

    Waiting reply on:

    * https://github.com/celery/billiard/issues/280
    * https://github.com/uqfoundation/pathos/issues/169

    See:

    * https://sebastianraschka.com/Articles/2014_multiprocessing.html
    * https://github.com/nipy/nipype/pull/2754
    * https://medium.com/contentsquare-engineering-blog/multithreading-vs-multiprocessing-in-python-ece023ad55a
    * http://mindcache.me/2015/08/09/python-multiprocessing-module-daemonic-processes-are-not-allowed-to-have-children.html
    * https://medium.com/@bfortuner/python-multithreading-vs-multiprocessing-73072ce5600b

    >>> list(iterate_mproc_map(np.sqrt, range(5), nb_workers=1, desc=None))  # doctest: +ELLIPSIS
    [0.0, 1.0, 1.41..., 1.73..., 2.0]
    >>> list(iterate_mproc_map(sum, [[0, 1]] * 5, nb_workers=2, ordered=False))
    [1, 1, 1, 1, 1]
    >>> list(iterate_mproc_map(max, [(2, 1)] * 5, nb_workers=2, desc=''))
    [2, 2, 2, 2, 2]
    """
    iterate_vals = list(iterate_vals)
    nb_workers = 1 if not nb_workers else int(nb_workers)
    nb_workers = CPU_COUNT if nb_workers < 0 else nb_workers

    if desc is not None:
        pbar = tqdm.tqdm(total=len(iterate_vals),
                         desc=str('%r @%i-threads' % (desc, nb_workers)))
    else:
        pbar = None

    if nb_workers > 1:
        logging.debug('perform parallel in %i threads', nb_workers)
        # Standard mproc.Pool created a demon processes which can be called
        # inside its children, cascade or multiprocessing
        # https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

        # pool = mproc.Pool(nb_workers)
        # pool = NonDaemonPool(nb_workers)
        pool = ProcessPool(nb_workers)
        # pool = Pool(nb_workers)
        mapping = pool.imap if ordered else pool.uimap
    else:
        logging.debug('perform sequential')
        pool = None
        mapping = map

    for out in mapping(wrap_func, iterate_vals):
        pbar.update() if pbar else None
        yield out

    if pool:
        pool.close()
        pool.join()
        pool.clear()

    pbar.close() if pbar else None


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
    return isinstance(var, iterable_types)


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


def _get_ram():
    """ get the RAM of the computer

    :return int: RAM value in GB
    """
    try:
        from psutil import virtual_memory
        ram = virtual_memory().total / 1024. ** 3
    except Exception:
        logging.exception('Retrieving info about RAM memory failed.')
        ram = np.nan
    return ram


def computer_info():
    """cet basic computer information.

    :return dict:

    >>> len(computer_info())
    9
    """
    return {
        'system': platform.system(),
        'architecture': platform.architecture(),
        'name': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'virtual CPUs': mproc.cpu_count(),
        'total RAM': _get_ram(),
    }
