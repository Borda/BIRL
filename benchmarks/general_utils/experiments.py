"""
General experiments methods


Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import time
import random
import logging

FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
FILE_LOGS = 'logging.txt'


def create_experiment_folder(path_out, dir_name, name='', stamp_unique=True):
    """ create the experiment folder and iterate while there is no available

    :param path_out: str, path to the base experiment directory
    :param name: str, special experiment name
    :param dir_name: str, special folder name
    :param stamp_unique: bool, whether add at the end of new folder unique tag

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
    if not os.path.exists(path_exp):
        os.mkdir(path_exp)
    return path_exp


def set_experiment_logger(path_out, file_name=FILE_LOGS, reset=True):
    """ set the logger to file

    :param path_out: str, path to the output folder
    :param file_name: str, log file name
    :param reset: bool, reset all previous logging into a file


    >>> set_experiment_logger('.')
    >>> len([h for h in logging.getLogger().handlers
    ...      if isinstance(h, logging.FileHandler)])
    1
    >>> os.remove(FILE_LOGS)
    """
    log = logging.getLogger()
    if reset:
        close_file_loggers()
    path_logger = os.path.join(path_out, file_name)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)


def close_file_loggers():
    """ close all handlers to a file

    >>> close_file_loggers()
    >>> len([h for h in logging.getLogger().handlers
    ...      if isinstance(h, logging.FileHandler)])
    0
    """
    log = logging.getLogger()
    log.handlers = [h for h in log.handlers
                    if not isinstance(h, logging.FileHandler)]


def string_dict(d, headline='DICTIONARY:', offset=25):
    """ format the dictionary into a string

    :param d: {str: val} dictionary with parameters
    :param headline: str, headline before the printed dictionary
    :param offset: int, max size of the string name
    :return: str

    >>> string_dict({'a': 1, 'b': 2}, 'TEST:', 5)
    'TEST:\\n"a":  1\\n"b":  2'

    """
    template = '{:%is} {}' % offset
    rows = [template.format('"{}":'.format(n), d[n]) for n in sorted(d)]
    s = headline + '\n' + '\n'.join(rows)
    return s
