"""
General template for experiments

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import json
import copy
import logging
import shutil

import benchmarks.general_utils.experiments as tl_experiments

FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
CONFIG_JSON = 'config.json'
RESULTS_TXT = 'results.txt'
RESULTS_CSV = 'results.csv'
FILE_LOGS = 'logging.txt'


class Experiment(object):
    """
    Tha basic template for experiment running with specific initialisation
    None, all required parameters used in future have to come in init phase

    >>> expt = Experiment({'path_out': '.', 'name': 'my_Test'}, False)
    >>> 'path_exp' in expt.params
    True
    >>> expt.run()
    True
    >>> del expt
    >>> shutil.rmtree('Experiment_my_Test')

    """

    def __init__(self, dict_params, stamp_unique=True):
        """ initialise the experiment, create experiment folder and set logger

        :param dict_params: {str: valeue}
        :param stamp_unique: bool, add at the end of experiment folder unique
            time stamp (actual date and time)
        """
        self.params = copy.deepcopy(dict_params)
        self.params['class'] = self.__class__.__name__
        self.__check_exist_path()
        self.__create_folder(stamp_unique)
        tl_experiments.set_experiment_logger(self.params['path_exp'], FILE_LOGS)
        if logging.getLogger().getEffectiveLevel() > logging.INFO:
            logging.getLogger().setLevel(logging.INFO)
        logging.info('initialise experiment...')
        logging.info(tl_experiments.string_dict(self.params, 'PARAMS:'))

    def run(self):
        """ running experiment """
        logging.info('running experiment...')
        self._load_data()
        self._perform()
        self._summarise()
        return True

    def _load_data(self):
        """ loading data """
        logging.info('-> loading data...')

    def _perform(self):
        """ perform experiment """
        logging.info('-> perform experiment...')

    def _summarise(self):
        """ summarise experiment """
        logging.info('-> summarise experiment...')

    def __check_exist_path(self):
        """ check existence of all paths """
        list_names = [n for n in self.params
                      if any(m in n.lower() for m in ['path', 'dir', 'file'])]
        for n in list_names:
            p = os.path.abspath(os.path.expanduser(self.params[n]))
            if not os.path.exists(p):
                raise Exception('given path "{}" does not exist!'.format(p))
            self.params[n] = p

    def __create_folder(self, stamp_unique=True):
        """ create the experiment folder (iterate if necessary) """
        assert 'path_out' in self.params
        # create results folder for experiments
        path_exp = tl_experiments.create_experiment_folder(
                        self.params.get('path_out'), self.__class__.__name__,
                        self.params.get('name'), stamp_unique)
        self.params['path_exp'] = path_exp
        with open(os.path.join(path_exp, CONFIG_JSON), 'w') as f:
            json.dump(self.params, f)

    def __del__(self):
        """ terminating experiment """
        logging.info('terminating experiment...')
        tl_experiments.close_file_loggers()
