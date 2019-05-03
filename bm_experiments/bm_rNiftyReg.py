"""
Benchmark for R package - RNiftyReg
LINKS:
* http://cran.r-project.org/web/packages/RNiftyReg/RNiftyReg.pdf
* https://github.com/jonclayden/RNiftyReg

Installation
------------
1. Install the R environment (https://stackoverflow.com/questions/31114991)::

    apt install r-base-core r-base-dev
    sudo apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev

2. Run R and install required R packages::

    install.packages(c("png", "jpeg", "OpenImageR", "devtools"))
    devtools::install_github("jonclayden/RNiftyReg")

Usage
-----
Run the basic R script::

    Rscript scripts/Rscript/RNiftyReg_linear.r \
        data_images/rat-kidney_/scale-5pc/Rat-Kidney_HE.jpg \
        data_images/rat-kidney_/scale-5pc/Rat-Kidney_PanCytokeratin.jpg \
        data_images/rat-kidney_/scale-5pc/Rat-Kidney_HE.csv \
        output/

Run the RNiftyReg benchmark::

    python bm_experiments/bm_rNiftyReg.py \
        -c ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -rr Rscript \
        -script ./scripts/Rscript/RNiftyReg_linear.r

.. note:: tested for RNiftyReg > 2.x

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import shutil

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import load_landmarks, save_landmarks
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import ImRegBenchmark, COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP
from birl.bm_template import main
from bm_experiments import bm_comp_perform

NAME_FILE_TIME = 'time.txt'
NAME_FILE_LANDMARKS = 'points.txt'
NAME_FILE_IMAGE = 'warped.jpg'


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-rr', '--exec_R', type=str, required=True,
                          help='path to the Rscript executable', default='Rscript')
    a_parser.add_argument('-script', '--path_R_script', required=True,
                          type=str, help='path to the R script with registration')
    return a_parser


class BmRNiftyReg(ImRegBenchmark):
    """ Benchmark for R package - RNiftyReg
    no run test while this method requires manual installation of RNiftyReg

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('scripts'), 'Rscript', n)
    >>> params = {'nb_workers': 1, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': os.path.join(update_path('data_images'),
    ...                                      'pairs-imgs-lnds_mix.csv'),
    ...           'exec_R': 'Rscript',
    ...           'path_R_script': fn_path_conf('RNiftyReg_linear.r')}
    >>> benchmark = BmRNiftyReg(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_R',
                                                        'path_R_script']

    def _prepare(self):
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_R_script')

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(record)
        path_dir = self._get_path_reg_dir(record) + os.path.sep
        cmd = ' '.join([
            self.params['exec_R'],
            self.params['path_R_script'],
            path_im_ref,
            path_im_move,
            path_lnds_move,
            path_dir,
        ])
        return cmd

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to
        """
        logging.debug('.. warp the registered image and get landmarks')
        path_dir = self._get_path_reg_dir(record)
        _, path_img_move, _, path_lnds_move = self._get_paths(record)
        path_lnds_warp, path_img_warp = None, None

        # load warped landmarks from TXT
        path_lnds = os.path.join(path_dir, NAME_FILE_LANDMARKS)
        if os.path.isfile(path_lnds):
            points_warp = load_landmarks(path_lnds)
            path_lnds_warp = os.path.join(path_dir, os.path.basename(path_lnds_move))
            save_landmarks(path_lnds_warp, points_warp)
            os.remove(path_lnds)

        path_regist = os.path.join(path_dir, NAME_FILE_IMAGE)
        if os.path.isfile(path_regist):
            name_img_move = os.path.splitext(os.path.basename(path_img_move))[0]
            ext_img_warp = os.path.splitext(NAME_FILE_IMAGE)[-1]
            path_img_warp = os.path.join(path_dir, name_img_move + ext_img_warp)
            os.rename(path_regist, path_img_warp)

        return {COL_IMAGE_MOVE_WARP: path_img_warp,
                COL_POINTS_MOVE_WARP: path_lnds_warp}

    def _extract_execution_time(self, record):
        """ if needed update the execution time
        :param record: {str: value}, dictionary with registration params
        :return float|None: time in minutes
        """
        path_dir = self._get_path_reg_dir(record)
        path_time = os.path.join(path_dir, NAME_FILE_TIME)
        with open(path_time, 'r') as fp:
            t_exec = float(fp.read()) / 60.
        return t_exec


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmRNiftyReg)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
