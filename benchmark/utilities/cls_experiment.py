"""
General template for experiments

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import json
import copy
import logging

import benchmark.utilities.experiments as tl_expt

FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
CONFIG_JSON = 'config.json'
RESULTS_TXT = 'results.txt'
RESULTS_CSV = 'results.csv'
FILE_LOGS = 'logging.txt'


class Experiment(object):
    """
    Tha basic template for experiment running with specific initialisation
    None, all required parameters used in future have to come in init phase

    >>> import benchmark.utilities.data_io as tl_io
    >>> path_out = tl_io.create_dir('output')
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

    def __init__(self, exp_params, stamp_unique=True):
        """ initialise the experiment, create experiment folder and set logger

        :param dict exp_params: {str: value}
        :param bool stamp_unique: add at the end of experiment folder unique
            time stamp (actual date and time)
        """
        self.params = copy.deepcopy(exp_params)
        self.params['class'] = self.__class__.__name__
        self._check_required_params()
        self.__check_exist_path()
        self.__create_folder(stamp_unique)
        tl_expt.set_experiment_logger(self.params['path_exp'], FILE_LOGS)
        # set stream logging to info level
        for lh in logging.getLogger().handlers:
            if isinstance(lh, logging.StreamHandler) and \
                    not isinstance(lh, logging.FileHandler):
                lh.setLevel(logging.INFO)
        logging.info('initialise experiment...')
        logging.info(tl_expt.string_dict(self.params, 'PARAMETERS:'))

    @classmethod
    def _check_required_params(self):
        """ check some extra required parameters for this experiment """
        logging.debug('.. check if Experiment have all required parameters')

    def run(self):
        """ running experiment """
        logging.info('running experiment...')
        self._prepare()
        self._load_data()
        self._run()
        self._summarise()
        return True

    @classmethod
    def _prepare(self):
        """ prepare the benchmark folder """
        logging.warning('-> preparing EMPTY experiments...')

    @classmethod
    def _load_data(self):
        """ loading data """
        logging.warning('-> loading EMPTY data...')

    @classmethod
    def _run(self):
        """ perform experiment """
        logging.warning('-> perform EMPTY experiment...')

    @classmethod
    def _summarise(self):
        """ summarise experiment """
        logging.warning('-> summarise EMPTY experiment...')

    def __check_exist_path(self):
        """ check existence of all paths """
        list_names = [n for n in self.params
                      if any(m in n.lower() for m in ['path', 'dir', 'file'])]
        for n in list_names:
            p = os.path.abspath(os.path.expanduser(self.params[n]))
            if not os.path.exists(p):
                raise Exception('given path "%s" does not exist!' % p)
            self.params[n] = p

    def __create_folder(self, stamp_unique=True):
        """ create the experiment folder (iterate if necessary) """
        assert 'path_out' in self.params, 'missing "path_out" among %s' \
                                          % repr(self.params.keys())
        # create results folder for experiments
        path_exp = tl_expt.create_experiment_folder(
            self.params.get('path_out'), self.__class__.__name__,
            self.params.get('name'), stamp_unique)
        self.params['path_exp'] = path_exp
        with open(os.path.join(path_exp, CONFIG_JSON), 'w') as f:
            json.dump(self.params, f)

    @classmethod
    def __del__(self):
        """ terminating experiment """
        logging.info('terminating experiment...')
        tl_expt.release_logger_files()
