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
        -t ./data_images/pairs-imgs-lnds_histol.csv \
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
from birl.benchmark import ImRegBenchmark
from bm_experiments import bm_comp_perform


class BmANTsPy(ImRegBenchmark):
    """ Benchmark for ANTs wrapper in Python
    no run test while this method requires manual installation of ANTsPy package

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_table': path_csv,
    ...           'path_out': path_out,
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'exec_Python': 'python',
    ...           'path_script': '.'}
    >>> benchmark = BmANTsPy(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_Python', 'path_script']
    #: file with exported image registration time
    NAME_IMAGE_WARPED = 'warped-image.jpg'
    #: file with warped landmarks after performed registration
    NAME_LNDS_WARPED = 'warped-landmarks.csv'
    #: file with warped image after performed registration
    NAME_TIME_EXEC = 'time.txt'

    def _prepare(self):
        """ prepare BM - copy configurations """
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_script')

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(item)

        cmd = ' '.join([
            self.params['exec_Python'],
            self.params['path_script'],
            path_im_ref,
            path_im_move,
            path_lnds_move,
            path_dir,
        ])

        return cmd

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to ...
        """
        path_dir = self._get_path_reg_dir(item)
        _, path_im_move, _, path_lnds_move = self._get_paths(item)
        path_img_warp, path_lnds_warp = None, None

        if os.path.isfile(os.path.join(path_dir, self.NAME_IMAGE_WARPED)):
            name_im_move = os.path.splitext(os.path.basename(path_im_move))[0]
            ext_img = os.path.splitext(self.NAME_IMAGE_WARPED)[-1]
            path_img_warp = os.path.join(path_dir, name_im_move + ext_img)
            os.rename(os.path.join(path_dir, self.NAME_IMAGE_WARPED), path_img_warp)
        if os.path.isfile(os.path.join(path_dir, self.NAME_LNDS_WARPED)):
            path_lnds_warp = os.path.join(path_dir, os.path.basename(path_lnds_move))
            os.rename(os.path.join(path_dir, self.NAME_LNDS_WARPED), path_lnds_warp)

        return {self.COL_IMAGE_MOVE_WARP: path_img_warp,
                self.COL_POINTS_MOVE_WARP: path_lnds_warp}

    def _extract_execution_time(self, item):
        """ if needed update the execution time

        :param dict item: dictionary with registration params
        :return float|None: time in minutes
        """
        path_dir = self._get_path_reg_dir(item)
        path_time = os.path.join(path_dir, self.NAME_TIME_EXEC)
        with open(path_time, 'r') as fp:
            t_exec = float(fp.read()) / 60.
        return t_exec

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-py', '--exec_Python', type=str, required=True,
                                help='path to the Python executable with ANTsPy', default='python3')
        arg_parser.add_argument('-script', '--path_script', required=True,
                                type=str, help='path to the image registration script')
        return arg_parser


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_params, path_expt = BmANTsPy.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
