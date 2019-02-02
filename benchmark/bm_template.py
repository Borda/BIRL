"""
All other benchmarks should be deviated this way so the core functionality
of the benchmarks such as loading or evaluation is not overwritten

INSTALLATION:
1. ...

EXAMPLE (usage):
>> mkdir results
>> python benchmark/bm_template.py \
    -c data_images/pairs-imgs-lnds_mix.csv -o results --visual --unique \
    --an_executable none

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import benchmark.utilities.experiments as tl_expt
import benchmark.cls_benchmark as bm


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('--an_executable', type=str, required=True,
                          help='some extra parameters')
    return a_parser


class BmTemplate(bm.ImRegBenchmark):
    """ a Template showing what method can be overwritten
     * _check_required_params
     * _prepare_registration
     * _generate_regist_command
     * _extract_warped_images_landmarks
     * _clear_after_registration

    This template benchmark also presents that method can have registered
    image as output but the transformed landmarks are in different frame
    (reference landmarks in moving image).

    Running in single thread:
    >>> from benchmark.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> main({'nb_jobs': 1, 'unique': False, 'visual': True,
    ...       'path_out': path_out, 'path_cover': path_csv,
    ...       'an_executable': ''})  # doctest: +ELLIPSIS
    '...'
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)

    Running in multiple parallel threads:
    >>> from benchmark.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_jobs': 2, 'unique': False, 'visual': True,
    ...           'path_out': path_out, 'path_cover':
    ...            path_csv, 'an_executable': ''}
    >>> benchmark = BmTemplate(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = bm.ImRegBenchmark.REQUIRED_PARAMS + ['an_executable']

    def _prepare(self):
        logging.info('-> copy configuration...')

    def _prepare_registration(self, record):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict record: {str: value}, dictionary with regist. params
        :return dict: {str: value}
        """
        logging.debug('.. no preparing before registration experiment')
        return record

    def _generate_regist_command(self, record):
        """ generate the registration command

        :param record: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        logging.debug('.. simulate registration: '
                      'copy the source image and landmarks, like regist. failed')
        path_reg_dir = self._get_path_reg_dir(record)
        name_img = os.path.basename(record[bm.COL_IMAGE_MOVE])
        cmd_img = 'cp %s %s' % (self._update_path(record[bm.COL_IMAGE_MOVE]),
                                os.path.join(path_reg_dir, name_img))
        name_lnds = os.path.basename(record[bm.COL_POINTS_MOVE])
        cmd_lnds = 'cp %s %s' % (self._update_path(record[bm.COL_POINTS_MOVE]),
                                 os.path.join(path_reg_dir, name_lnds))
        command = [cmd_img, cmd_lnds]
        return command

    def _extract_warped_images_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to ...
        """
        # detect image
        path_img = os.path.join(record[bm.COL_REG_DIR],
                                os.path.basename(record[bm.COL_IMAGE_MOVE]))
        # detect landmarks
        path_lnd = os.path.join(record[bm.COL_REG_DIR],
                                os.path.basename(record[bm.COL_POINTS_MOVE]))
        return None, path_img, None, path_lnd

    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param record: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. no cleaning after registration experiment')
        return record


def main(params):
    """ run the Main of blank experiment

    :param params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = BmTemplate(params)
    benchmark.run()
    path_expt = benchmark.params['path_exp']
    del benchmark
    logging.info('Done.')
    return path_expt


# RUN by given parameters
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = tl_expt.create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = tl_expt.parse_arg_params(arg_parser)
    path_expt = main(arg_params)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Here you can call the separate benchmark'
                     ' to measure your computer performances.')

        # from bm_experiments import bm_comp_perform
        # bm_comp_perform.main(path_expt)
