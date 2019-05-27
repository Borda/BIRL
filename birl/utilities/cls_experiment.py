"""
General template for experiments

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import copy
import logging

from birl.utilities.data_io import save_config_yaml, update_path
from birl.utilities.experiments import (
    set_experiment_logger, string_dict, create_experiment_folder, release_logger_files,
    FILE_LOGS)

#: default output file for exporting experiment configuration
CONFIG_YAML = 'config.yml'
#: default file for exporting results in formatted text format
RESULTS_TXT = 'results.txt'
#: default file for exporting results in table format
RESULTS_CSV = 'results.csv'


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
        save_config_yaml(os.path.join(path_exp, CONFIG_YAML), self.params)

    def __del__(self):
        """Terminating experiment.

        close the logger if the termination instance is the main one
        """
        if hasattr(self, '_main_thread') and self._main_thread:
            logging.info('terminating experiment...')
            release_logger_files()
        logging.debug('terminating child experiment...')
