"""
All other benchmarks should be deviated this way so the core functionality
of the benchmarks such as loading or evaluation is not overwritten

INSTALLATION:
1. ...

EXAMPLE (usage):
>> mkdir results
>> python benchmarks/bm_template.py \
    -in data_images/pairs-imgs-lnds_mix.csv -out results --visual --unique \
    --an_executable none

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import benchmark.utilities.experiments as tl_expt
import benchmark.utilities.data_io as tl_io
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
     * _evaluate_registration
     * _clear_after_registration

    This template benchmark also presents that method can have registered
    image as output but the transformed landmarks are in different frame
    (reference landmarks in moving image).

    Running in single thread:
    >>> import benchmark.utilities.data_io as tl_io
    >>> path_out = tl_io.create_dir('temp_results')
    >>> path_csv = os.path.join(tl_io.update_path('data_images'),
    ...                         'pairs-imgs-lnds_mix.csv')
    >>> main({'nb_jobs': 1, 'unique': False, 'path_out': path_out,
    ...       'path_cover': path_csv, 'an_executable': ''})
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)

    Running in 2 threads:
    >>> import benchmark.utilities.data_io as tl_io
    >>> path_out = tl_io.create_dir('temp_results')
    >>> path_csv = os.path.join(tl_io.update_path('data_images'),
    ...                         'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_jobs': 2, 'unique': False, 'path_out': path_out,
    ...           'path_cover': path_csv, 'an_executable': ''}
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

    def _prepare_registration(self, dict_row):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict dict_row: {str: value}, dictionary with regist. params
        :return dict: {str: value}
        """
        logging.debug('.. no preparing before registration experiment')
        return dict_row

    def _generate_regist_command(self, dict_row):
        """ generate the registration command

        :param dict_row: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        logging.debug('.. simulate registration: '
                      'copy the original image and landmarks')
        reg_dir = dict_row[bm.COL_REG_DIR]
        name_img = os.path.basename(dict_row[bm.COL_IMAGE_MOVE])
        name_lnds = os.path.basename(dict_row[bm.COL_POINTS_MOVE])
        cmd_img = 'cp %s %s' % (tl_io.update_path(dict_row[bm.COL_IMAGE_MOVE]),
                                os.path.join(reg_dir, name_img))
        cmd_lnds = 'cp %s %s' % (tl_io.update_path(dict_row[bm.COL_POINTS_MOVE]),
                                 os.path.join(reg_dir, name_lnds))
        command = ' && '.join([cmd_img, cmd_lnds])
        return command

    def _extract_warped_images_landmarks(self, dict_row):
        """ get registration results - warped registered images and landmarks

        :param dict_row: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to ...
        """
        # detect image
        path_img = os.path.join(dict_row[bm.COL_REG_DIR],
                                os.path.basename(dict_row[bm.COL_IMAGE_MOVE]))
        # detect landmarks
        path_lnd = os.path.join(dict_row[bm.COL_REG_DIR],
                                os.path.basename(dict_row[bm.COL_POINTS_REF]))
        return None, path_img, None, path_lnd

    def _clear_after_registration(self, dict_row):
        """ clean unnecessarily files after the registration

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. no cleaning after registration experiment')
        return dict_row


def main(params):
    """ run the Main of blank experiment

    :param arg_params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = BmTemplate(params)
    benchmark.run()
    del benchmark
    logging.info('Done.')


# RUN by given parameters
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = tl_expt.create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = tl_expt.parse_params(arg_parser)
    main(arg_params)
