"""
All other benchmarks should be deviated this way so the core functionality
of the benchmarks such as loading or evaluation is not overwritten

INSTALLATION:
1. ...

EXAMPLE (usage):
>> mkdir results
>> python benchmarks/bm_template.py \
    -in data_images/list_pairs_imgs_lnds.csv -out results --unique \
    --an_executable none

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
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
     * _evaluate_registration
     * _clear_after_registration

    This template benchmark also presents that method can have registered
    image as output but the transformed landmarks are in different frame
    (reference landmarks in moving image).

    Running in single thread:
    >>> import benchmark.utilities.data_io as tl_io
    >>> path_out = tl_io.create_dir('temp_results')
    >>> path_csv = os.path.join(tl_io.update_path('data_images'),
    ...                         'list_pairs_imgs_lnds.csv')
    >>> main({'nb_jobs': 1, 'unique': False, 'path_out': path_out,
    ...       'path_cover': path_csv, 'an_executable': ''})
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)

    Running in 2 threads:
    >>> import benchmark.utilities.data_io as tl_io
    >>> path_out = tl_io.create_dir('temp_results')
    >>> path_csv = os.path.join(tl_io.update_path('data_images'),
    ...                         'list_pairs_imgs_lnds.csv')
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
        logging.debug('.. no preparing before regist. experiment')
        return dict_row

    def _generate_regist_command(self, dict_row):
        """ generate the registration command

        :param dict_row: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        target_img = os.path.join(dict_row[bm.COL_REG_DIR],
                                  os.path.basename(dict_row[bm.COL_IMAGE_MOVE]))
        target_lnd = os.path.join(dict_row[bm.COL_REG_DIR],
                                  os.path.basename(dict_row[bm.COL_POINTS_REF]))
        cmds = ['cp %s %s' % (os.path.abspath(dict_row[bm.COL_IMAGE_MOVE]),
                              target_img),
                'cp %s %s' % (os.path.abspath(dict_row[bm.COL_POINTS_REF]),
                              target_lnd)]
        command = ' && '.join(cmds)
        return command

    def _evaluate_registration(self, dict_row):
        """ evaluate rests of the experiment and identity the registered image
        and landmarks when the process finished

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. simulate registration: '
                      'copy the original image and landmarks')
        # detect image
        path_img = os.path.join(dict_row[bm.COL_REG_DIR],
                                os.path.basename(dict_row[bm.COL_IMAGE_MOVE]))
        if os.path.exists(path_img):
            dict_row[bm.COL_IMAGE_REF_WARP] = path_img
        # detect landmarks
        path_lnd = os.path.join(dict_row[bm.COL_REG_DIR],
                                os.path.basename(dict_row[bm.COL_POINTS_REF]))
        # for inverse landmarks estimate
        if os.path.exists(path_img):
            dict_row[bm.COL_POINTS_MOVE_WARP] = path_lnd

        return dict_row

    def _clear_after_registration(self, dict_row):
        """ clean unnecessarily files after the registration

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. no cleaning after regist. experiment')
        return dict_row


def main(arg_params):
    """ run the Main of blank experiment

    :param arg_params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = BmTemplate(arg_params)
    benchmark.run()
    del benchmark
    logging.info('Done.')


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = tl_expt.create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = tl_expt.parse_params(arg_parser)
    main(arg_params)
