"""
All other benchmarks should be deviated this way so the core functionality
of the benchmarks such as loading or evaluation is not overwritten

INSTALLATION:

1. ...

Sample run::

    mkdir ./results
    touch sample-config.yaml
    python birl/bm_template.py \
        -t ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        --visual --unique \
        -cfg ./sample-config.yaml

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import create_basic_parser
from birl.benchmark import ImRegBenchmark


class BmTemplate(ImRegBenchmark):
    """ Basic template showing utilization by inheriting general workflow.
    This case serves as an example of using the general image regist. benchmark.

    :param dict params: dictionary with experiment configuration,
        the required options are names in `REQUIRED_PARAMS`,
        note that the basic parameters are inherited

    General methods that should be overwritten:

     * `_check_required_params`
     * `_prepare_img_registration`
     * `_execute_img_registration`/`_generate_regist_command`
     * `_extract_warped_image_landmarks`
     * `_extract_execution_time`
     * `_clear_after_registration`

    .. note:: The actual implementation simulates the "WORSE" registration while
     it blindly copies the moving landmarks as results of the registration.
     It also copies the moving images so there is correct "warping" between
     image and landmarks. It means that there was no registration performed.

    Examples
    --------
    >>> # Running in single thread:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> open('sample_config.yaml', 'w').close()
    >>> BmTemplate.main({
    ...       'path_table': path_csv,
    ...       'path_out': path_out,
    ...       'nb_workers': 1,
    ...       'unique': False,
    ...       'visual': True,
    ...       'path_config': 'sample_config.yaml'
    ... })  # doctest: +ELLIPSIS
    ({...}, '...BmTemplate')
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    >>> os.remove('sample_config.yaml')

    >>> # Running in multiple parallel threads:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_table': path_csv,
    ...           'path_out': path_out,
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'visual': True,
    ...           'path_config': ''}
    >>> benchmark = BmTemplate(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_config']

    def _prepare(self):
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_config')

    def _prepare_img_registration(self, item):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict item: dictionary with regist. params
        :return dict: the same or updated registration info
        """
        logging.debug('.. no preparing before registration experiment')
        return item

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        logging.debug('.. simulate registration: '
                      'copy the source image and landmarks, like regist. failed')
        _, path_im_move, _, path_lnds_move = self._get_paths(item)
        path_reg_dir = self._get_path_reg_dir(item)
        name_img = os.path.basename(item[self.COL_IMAGE_MOVE])
        cmd_img = 'cp %s %s' % (path_im_move, os.path.join(path_reg_dir, name_img))
        name_lnds = os.path.basename(item[self.COL_POINTS_MOVE])
        cmd_lnds = 'cp %s %s' % (path_lnds_move, os.path.join(path_reg_dir, name_lnds))
        commands = [cmd_img, cmd_lnds]
        return commands

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        path_reg_dir = self._get_path_reg_dir(item)
        # detect image
        path_img = os.path.join(path_reg_dir, os.path.basename(item[self.COL_IMAGE_MOVE]))
        # detect landmarks
        path_lnd = os.path.join(path_reg_dir, os.path.basename(item[self.COL_POINTS_MOVE]))
        # return formatted results
        return {self.COL_IMAGE_MOVE_WARP: path_img,
                self.COL_POINTS_MOVE_WARP: path_lnd}

    def _extract_execution_time(self, item):
        """ if needed update the execution time

        :param dict item: dictionary with registration params
        :return float|None: time in minutes
        """
        return 1. / 60  # running constant time 1 sec.

    def _clear_after_registration(self, item):
        """ clean unnecessarily files after the registration

        :param dict item: dictionary with regist. information
        :return dict: the same or updated regist. info
        """
        logging.debug('.. no cleaning after registration experiment')
        return item

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:

        >>> parser = BmTemplate.extend_parse(create_basic_parser())
        >>> type(parser)
        <class 'argparse.ArgumentParser'>
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-cfg', '--path_config', type=str, required=True,
                                help='some extra parameters')
        return arg_parser


# RUN by given parameters
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params, path_expt = BmTemplate.main()

    if arg_params.get('run_comp_benchmark', False):
        # from bm_experiments import bm_comp_perform
        # bm_comp_perform.main(path_expt)
        logging.info('Here you can call the separate benchmark'
                     ' to measure your computer performances.')
