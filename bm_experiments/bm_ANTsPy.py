"""
Benchmark for ANTs

See references:

* http://stnava.github.io/ANTs
* https://sourceforge.net/projects/advants/
* https://github.com/ANTsX/ANTsPy

Installation
------------
1. Install it as python package::

    pip install git+https://github.com/ANTsX/ANTsPy.git

Usage
-----
Run the basic ANTs registration with original parameters::

    python bm_experiments/bm_ANTsPy.py \
        -c ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -py python3 \
        -script ./scripts/Python/run_ANTsPy.py


.. note:: required to use own compiled last version since some previous releases
 do not contain `ants.apply_transforms_to_points`

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import shutil

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import ImRegBenchmark, COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP
from birl.bm_template import main
from bm_experiments import bm_comp_perform


NAME_IMAGE_WARPED = 'warped-image.jpg'
NAME_LNDS_WARPED = 'warped-landmarks.csv'
NAME_TIME_EXEC = 'time.txt'


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-py', '--exec_Python', type=str, required=True,
                          help='path to the Python executable with ANTsPy', default='python3')
    a_parser.add_argument('-script', '--path_script', required=True,
                          type=str, help='path to the image registration script')
    return a_parser


class BmANTsPy(ImRegBenchmark):
    """ Benchmark for ANTs wrapper in Python
    no run test while this method requires manual installation of ANTsPy package

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> params = {'nb_workers': 1, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': os.path.join(update_path('data_images'),
    ...                                      'pairs-imgs-lnds_mix.csv'),
    ...           'exec_Python': 'python', 'path_script': '.'}
    >>> benchmark = BmANTsPy(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_Python', 'path_script']

    def _prepare(self):
        """ prepare BM - copy configurations """
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_script')

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        path_dir = self._get_path_reg_dir(record)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(record)

        cmd = ' '.join([
            self.params['exec_Python'],
            self.params['path_script'],
            path_im_ref,
            path_im_move,
            path_lnds_move,
            path_dir,
        ])

        return cmd

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param {str: value} record: dictionary with registration params
        :return {str: str}: paths to ...
        """
        path_dir = self._get_path_reg_dir(record)
        _, path_im_move, _, path_lnds_move = self._get_paths(record)
        path_im_warp, path_lnds_warp = None, None

        if os.path.isfile(os.path.join(path_dir, NAME_IMAGE_WARPED)):
            name_im_move = os.path.splitext(os.path.basename(path_im_move))[0]
            ext_img = os.path.splitext(NAME_IMAGE_WARPED)[-1]
            path_im_warp = os.path.join(path_dir, name_im_move + ext_img)
            os.rename(os.path.join(path_dir, NAME_IMAGE_WARPED), path_im_warp)
        if os.path.isfile(os.path.join(path_dir, NAME_LNDS_WARPED)):
            path_lnds_warp = os.path.join(path_dir, os.path.basename(path_lnds_move))
            os.rename(os.path.join(path_dir, NAME_LNDS_WARPED), path_lnds_warp)

        return {COL_IMAGE_MOVE_WARP: path_im_warp,
                COL_POINTS_MOVE_WARP: path_lnds_warp}

    def _extract_execution_time(self, record):
        """ if needed update the execution time

        :param record: {str: value}, dictionary with registration params
        :return float|None: time in minutes
        """
        path_dir = self._get_path_reg_dir(record)
        path_time = os.path.join(path_dir, NAME_TIME_EXEC)
        with open(path_time, 'r') as fp:
            t_exec = float(fp.read()) / 60.
        return t_exec


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmANTsPy)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
