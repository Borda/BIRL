"""
Benchmark for ImageJ plugin - RVSS

.. seealso::  https://imagej.net/Register_Virtual_Stack_Slices

Installation
------------
1. Enter the application folder in this project::

    cd <BIRL>/applications

2. Download Fiji - https://fiji.sc/ ::

    wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip

3. Extract the downloaded application::

    unzip fiji-linux64.zip

4. Try to run Fiji::

    Fiji.app/ImageJ-linux64

Usage
-----
Run the basic RVSS registration with original parameters::

    python bm_experiments/bm_RVSS.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji ~/Applications/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_RVSS_histol.yaml \
        --visual --unique

.. note:: tested for version ImageJ 1.52i & 2.35

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import os
import shutil
import sys

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.benchmark import ImRegBenchmark
from birl.utilities.data_io import load_config_yaml, load_landmarks, save_landmarks
from birl.utilities.experiments import dict_deep_update, exec_commands
from bm_experiments import bm_comp_perform
from bm_experiments.bm_bUnwarpJ import BmUnwarpJ


class BmRVSS(ImRegBenchmark):
    """ Benchmark for ImageJ plugin - RVSS
    no run test while this method requires manual installation of ImageJ

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> path_csv = os.path.join(update_path('data-images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_table': path_csv,
    ...           'path_out': path_out,
    ...           'exec_Fiji': 'ImageJ-linux64',
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'path_config': fn_path_conf('ImageJ_RVSS_histol.yaml')}
    >>> benchmark = BmRVSS(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_Fiji', 'path_config']
    #: path/name of image registration script
    PATH_SCRIPT_REGISTRATION = os.path.join(BmUnwarpJ.PATH_IJ_SCRIPTS, 'apply-RVSS-registration.bsh')
    #: path/name of image/landmarks warping script
    PATH_SCRIPT_WARP_LANDMARKS = os.path.join(BmUnwarpJ.PATH_IJ_SCRIPTS, 'apply-RVSS-transform.bsh')
    #: internal folder name for copy input image pairs
    DIR_INPUTS = 'input'
    #: internal folder name for registration results - images and transformations
    DIR_OUTPUTS = 'output'
    # PATH_SCRIPT_HIST_MATCH_IJM = os.path.join(PATH_IJ_SCRIPTS, 'histogram-matching-for-macro.bsh')
    #: command for executing the image registration
    COMMAND_REGISTRATION = \
        '%(exec_Fiji)s --headless %(path_bsh)s' \
        ' %(dir_input)s/ %(dir_output)s/ %(dir_output)s/' \
        ' %(ref_name)s %(params)s'
    #: command for executing the warping image and landmarks
    COMMAND_WARP_LANDMARKS = \
        '%(exec_Fiji)s --headless %(path_bsh)s' \
        ' %(source)s %(target)s' \
        ' %(output)s/' + BmUnwarpJ.NAME_LANDMARKS + \
        ' %(output)s/' + BmUnwarpJ.NAME_LANDMARKS_WARPED + \
        ' %(transf)s' \
        ' %(warp)s'
    #: required parameters in the configuration file for RVSS
    REQUIRED_PARAMS_RVSS = ('shrinkingConstraint', 'featuresModelIndex', 'registrationModelIndex')
    #: default RVSS parameters
    DEFAULT_PARAMS = {
        'RVSS': {
            'shrinkingConstraint': 1,  # (0 to use reference image,
            #  or 1 to use shrinking constraint mode)
            'featuresModelIndex': 1,  # (0=TRANSLATION, 1=RIGID, 2=SIMILARITY, 3=AFFINE)
            #: Index of the registration model
            'registrationModelIndex': 3  # (0=TRANSLATION, 1=RIGID, 2=SIMILARITY, 3=AFFINE,
            #  4=ELASTIC, 5=MOVING_LEAST_SQUARES)
        },
        'SIFT': BmUnwarpJ.DEFAULT_PARAMS['SIFT']
    }

    # assert all(k in DEFAULT_PARAMS['RVSS'] for k in REQUIRED_PARAMS_RVSS), \
    #     'default params are missing some required parameters for RVSS'
    # assert all(k in DEFAULT_PARAMS['SIFT'] for k in REQUIRED_PARAMS_SIFT), \
    #     'default params are missing some required parameters for SIFT'

    def _prepare(self):
        """ prepare Benchmark - copy configurations """
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_config')

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        path_im_ref, path_im_move, _, _ = self._get_paths(item, prefer_pproc=True)
        path_dir = self._get_path_reg_dir(item)

        # creating the internal folders
        path_dir_in = os.path.join(path_dir, self.DIR_INPUTS)
        path_dir_out = os.path.join(path_dir, self.DIR_OUTPUTS)
        for p_dir in (path_dir_in, path_dir_out):
            os.mkdir(p_dir)
        # copy both images
        name_ref = os.path.basename(path_im_ref)
        shutil.copy(path_im_ref, os.path.join(path_dir_in, name_ref))
        shutil.copy(path_im_move, os.path.join(path_dir_in, os.path.basename(path_im_move)))

        config = self.DEFAULT_PARAMS
        config = dict_deep_update(config, load_config_yaml(self.params['path_config']))

        config_rvss = [config['RVSS'][k] for k in self.REQUIRED_PARAMS_RVSS]
        config_sift = [config['SIFT'][k] for k in BmUnwarpJ.REQUIRED_PARAMS_SIFT]

        cmd = self.COMMAND_REGISTRATION % {
            'exec_Fiji': self.params['exec_Fiji'],
            'path_bsh': self.PATH_SCRIPT_REGISTRATION,
            'dir_input': path_dir_in,
            'dir_output': path_dir_out,
            'ref_name': name_ref,
            'params': ' '.join(map(str, config_rvss + config_sift)),
        }
        return cmd

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        logging.debug('.. warp the registered image and get landmarks')
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(item, prefer_pproc=False)
        path_log = os.path.join(path_dir, self.NAME_LOG_REGISTRATION)

        # warp moving landmarks to reference frame
        path_dir_out = os.path.join(path_dir, self.DIR_OUTPUTS)
        # name_ref, _ = os.path.splitext(os.path.basename(path_im_ref))
        name_move, _ = os.path.splitext(os.path.basename(path_im_move))
        path_img_warp = os.path.join(path_dir, os.path.basename(path_im_move))
        dict_params = {
            'exec_Fiji': self.params['exec_Fiji'],
            'path_bsh': self.PATH_SCRIPT_WARP_LANDMARKS,
            'source': path_im_move,
            'target': path_im_ref,
            'output': path_dir,
            # 'transf': os.path.join(path_dir_out, name_ref + '.xml'),
            'transf': os.path.join(path_dir_out, name_move + '.xml'),
            'warp': path_img_warp,
        }

        # export source points to TXT
        pts_source = load_landmarks(path_lnds_move)
        save_landmarks(os.path.join(path_dir, BmUnwarpJ.NAME_LANDMARKS), pts_source)
        # execute transformation
        exec_commands(self.COMMAND_WARP_LANDMARKS % dict_params, path_logger=path_log, timeout=self.EXECUTE_TIMEOUT)
        # load warped landmarks from TXT
        path_lnds_warp = os.path.join(path_dir, BmUnwarpJ.NAME_LANDMARKS_WARPED)
        if os.path.isfile(path_lnds_warp):
            points_warp = load_landmarks(path_lnds_warp)
            path_lnds_warp = os.path.join(path_dir, os.path.basename(path_lnds_move))
            save_landmarks(path_lnds_warp, points_warp)
        else:
            path_lnds_warp = None

        # return results
        return {
            self.COL_IMAGE_MOVE_WARP: path_img_warp,
            self.COL_POINTS_MOVE_WARP: path_lnds_warp,
        }

    def _clear_after_registration(self, item):
        path_dir = self._get_path_reg_dir(item)

        for p_dir in (os.path.join(path_dir, self.DIR_INPUTS), os.path.join(path_dir, self.DIR_OUTPUTS)):
            shutil.rmtree(p_dir)

        return item

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-Fiji', '--exec_Fiji', type=str, required=True, help='path to the Fiji executable')
        arg_parser.add_argument('-cfg', '--path_config', required=True, type=str, help='path to the RVSS configuration')
        return arg_parser


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmRVSS.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
