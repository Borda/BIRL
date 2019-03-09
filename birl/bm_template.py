"""
All other benchmarks should be deviated this way so the core functionality
of the benchmarks such as loading or evaluation is not overwritten

INSTALLATION:
1. ...

EXAMPLE (usage):
>> mkdir ./results
>> python birl/bm_template.py \
    -c ./data_images/pairs-imgs-lnds_anhir.csv -d ./data_images \
    -o ./results --visual --unique \
    --path_sample_config none

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import ImRegBenchmark, COL_REG_DIR, COL_IMAGE_MOVE, COL_POINTS_MOVE


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('--path_sample_config', type=str, required=True,
                          help='some extra parameters')
    return a_parser


class BmTemplate(ImRegBenchmark):
    """ Basic template showing utilization by inheriting general workflow.
    This case serves as an example of using the general image regist. benchmark.

    :param {str: str|float} params: dictionary with experiment configuration,
        the required options are names in `REQUIRED_PARAMS`,
        note that the basic parameters are inherited

    General methods that should be overwritten:
     * _check_required_params
     * _prepare_registration
     * _generate_regist_command
     * _extract_warped_image_landmarks
     * _extract_execution_time
     * _clear_after_registration

    NOTE: The actual implementation simulates the "WORSE" registration while
    it blindly copies the moving landmarks as results of the registration.
    It also copies the moving images so there is correct "warping" between
    image and landmarks. It means that there was no registration performed.

    Running in single thread:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> open('sample_config.txt', 'w').close()
    >>> main({'nb_workers': 1, 'unique': False, 'visual': True,
    ...       'path_out': path_out, 'path_cover': path_csv,
    ...       'path_sample_config': 'sample_config.txt'}, BmTemplate)  # doctest: +ELLIPSIS
    '...'
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    >>> os.remove('sample_config.txt')

    Running in multiple parallel threads:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_workers': 2, 'unique': False, 'visual': True,
    ...           'path_out': path_out, 'path_cover':
    ...            path_csv, 'path_sample_config': ''}
    >>> benchmark = BmTemplate(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_sample_config']

    def _prepare(self):
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_sample_config')

    def _prepare_registration(self, record):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. no preparing before registration experiment')
        return record

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        logging.debug('.. simulate registration: '
                      'copy the source image and landmarks, like regist. failed')
        _, path_im_move, _, path_lnds_move = self._get_paths(record)
        path_reg_dir = self._get_path_reg_dir(record)
        name_img = os.path.basename(record[COL_IMAGE_MOVE])
        cmd_img = 'cp %s %s' % (path_im_move, os.path.join(path_reg_dir, name_img))
        name_lnds = os.path.basename(record[COL_POINTS_MOVE])
        cmd_lnds = 'cp %s %s' % (path_lnds_move, os.path.join(path_reg_dir, name_lnds))
        command = [cmd_img, cmd_lnds]
        return command

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to ...
        """
        # detect image
        path_img = os.path.join(record[COL_REG_DIR],
                                os.path.basename(record[COL_IMAGE_MOVE]))
        # detect landmarks
        path_lnd = os.path.join(record[COL_REG_DIR],
                                os.path.basename(record[COL_POINTS_MOVE]))
        return None, path_img, None, path_lnd

    def _extract_execution_time(self, record):
        """ if needed update the execution time

        :param record: {str: value}, dictionary with registration params
        :return float|None: time in minutes
        """
        return 1. / 60  # running constant time 1 sec.

    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param {str: value} record: dictionary with regist. information
        :return {str: value}: the same or updated regist. info
        """
        logging.debug('.. no cleaning after registration experiment')
        return record


def main(params, cls_benchmark):
    """ run the Main of selected experiment

    :param {str: str|float} params: set of input parameters
    :param cls_benchmark: class of selected benchmark
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = cls_benchmark(params)
    benchmark.run()
    path_expt = benchmark.params['path_exp']
    del benchmark
    logging.info('Done.')
    return path_expt


# RUN by given parameters
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmTemplate)

    if arg_params.get('run_comp_benchmark', False):
        # from bm_experiments import bm_comp_perform
        # bm_comp_perform.main(path_expt)
        logging.info('Here you can call the separate benchmark'
                     ' to measure your computer performances.')
