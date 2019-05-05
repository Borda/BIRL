"""
Benchmark for ImageJ plugin - bUnwarpJ

.. ref::  http://imagej.net/BUnwarpJ

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
Run the basic bUnwarpJ registration with original parameters::

    python bm_experiments/bm_bUnwarpJ.py \
        -c ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -Fiji ~/Applications/Fiji.app/ImageJ-linux64 \
        -config ./configs/ImageJ_bUnwarpJ_histol.yaml \
        --preprocessing hist-matching \
        --visual --unique

The bUnwarpJ is supporting SIFT and MOPS feature extraction as landmarks
see: http://imagej.net/BUnwarpJ#SIFT_and_MOPS_plugin_support ::

    python bm_experiments/bm_bUnwarpJ.py \
        -c ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -Fiji ~/Applications/Fiji.app/ImageJ-linux64 \
        -config ./configs/ImageJ_bUnwarpJ-SIFT_histol.yaml \
        --preprocessing hist-matching \
        --visual --unique

.. note:: tested for version ImageJ 1.52i & 2.35

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import shutil

import yaml

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import update_path, load_landmarks, save_landmarks
from birl.utilities.experiments import (create_basic_parse, parse_arg_params, exec_commands,
                                        dict_deep_update)
from birl.cls_benchmark import (ImRegBenchmark, NAME_LOG_REGISTRATION,
                                COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP)
from birl.bm_template import main
from bm_experiments import bm_comp_perform

PATH_IJ_SCRIPTS = os.path.join(update_path('scripts'), 'ImageJ')
PATH_SCRIPT_REGISTRATION_BASE = os.path.join(PATH_IJ_SCRIPTS, 'apply-bUnwarpJ-registration.bsh')
PATH_SCRIPT_REGISTRATION_SIFT = os.path.join(PATH_IJ_SCRIPTS, 'apply-SIFT-bUnwarpJ-registration.bsh')
PATH_SCRIPT_WARP_LANDMARKS = os.path.join(PATH_IJ_SCRIPTS, 'apply-bUnwarpJ-transform.bsh')
# PATH_SCRIPT_HIST_MATCH_IJM = os.path.join(PATH_IJ_SCRIPTS, 'histogram-matching-for-macro.bsh')
NAME_LANDMARKS = 'source_landmarks.txt'
NAME_LANDMARKS_WARPED = 'warped_source_landmarks.txt'
COMMAND_WARP_LANDMARKS = '%(exec_Fiji)s --headless %(path_bsh)s' \
                         ' %(source)s %(target)s' \
                         ' %(output)s/' + NAME_LANDMARKS + \
                         ' %(output)s/' + NAME_LANDMARKS_WARPED + \
                         ' %(output)s/transform-inverse.txt' \
                         ' %(output)s/transform-direct.txt' \
                         ' %(warp)s'
COMMAND_REGISTRATION = '%(exec_Fiji)s --headless %(path_bsh)s' \
                       ' %(source)s %(target)s %(params)s' \
                       ' %(output)s/transform-direct.txt %(output)s/transform-inverse.txt'
REQUIRED_PARAMS_BUNWARPJ = (
    'mode', 'subsampleFactor', 'minScale', 'maxScale', 'divWeight', 'curlWeight',
    'landmarkWeight', 'imageWeight', 'consistencyWeight', 'stopThreshold')
REQUIRED_PARAMS_SIFT = (
    'initialSigma', 'steps', 'minOctaveSize', 'maxOctaveSize', 'fdSize', 'fdBins', 'rod',
    'maxEpsilon', 'minInlierRatio', 'modelIndex')
DEFAULT_PARAMS = {
    'bUnwarpJ': {
        'mode': 1,
        'subsampleFactor': 0,
        'minScale': 0,
        'maxScale': 3,
        'divWeight': 0.1,
        'curlWeight': 0.1,
        'landmarkWeight': 0.,
        'imageWeight': 1.,
        'consistencyWeight': 10.,
        'stopThreshold': 0.01,
    },
    'SIFT': {
        'initialSigma': 1.6,
        'steps': 3,
        'minOctaveSize': 64,
        'maxOctaveSize': 1024,
        'fdSize': 8,
        'fdBins': 8,
        'rod': 0.92,
        'maxEpsilon': 25,
        'minInlierRatio': 0.05,
        'modelIndex': 1
    }
}
assert all(k in DEFAULT_PARAMS['bUnwarpJ'] for k in REQUIRED_PARAMS_BUNWARPJ), \
    'default params are missing some required parameters for bUnwarpJ'
assert all(k in DEFAULT_PARAMS['SIFT'] for k in REQUIRED_PARAMS_SIFT), \
    'default params are missing some required parameters for SIFT'


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-Fiji', '--exec_Fiji', type=str, required=True,
                          help='path to the Fiji executable')
    a_parser.add_argument('-config', '--path_config', required=True,
                          type=str, help='path to the bUnwarpJ configuration')
    return a_parser


class BmUnwarpJ(ImRegBenchmark):
    """ Benchmark for ImageJ plugin - bUnwarpJ
    no run test while this method requires manual installation of ImageJ

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_cover': path_csv,
    ...           'path_out': path_out,
    ...           'exec_Fiji': 'ImageJ-linux64',
    ...           'preprocessing': ['hist-matching'],
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'path_config': fn_path_conf('ImageJ_bUnwarpJ_histol.yaml')}
    >>> benchmark = BmUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> params['path_config'] = fn_path_conf('ImageJ_bUnwarpJ-SIFT_histol.yaml')
    >>> benchmark = BmUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_Fiji', 'path_config']

    def _prepare(self):
        """ prepare Benchmark - copy configurations """
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_config')

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        path_im_ref, path_im_move, _, _ = self._get_paths(record, prefer_pproc=True)
        path_dir = self._get_path_reg_dir(record)
        config = DEFAULT_PARAMS
        with open(self.params['path_config'], 'r') as fp:
            config = dict_deep_update(config, yaml.load(fp))
        assert config['bUnwarpJ']['mode'] < 2, 'Mono mode does not supports inverse transform' \
                                               ' which is need for landmarks warping.'

        config_sift = [config['SIFT'][k] for k in REQUIRED_PARAMS_SIFT] \
            if config.get('SIFT', False) else []
        config_bunwarpj = [config['bUnwarpJ'][k] for k in REQUIRED_PARAMS_BUNWARPJ]
        path_reg_script = PATH_SCRIPT_REGISTRATION_SIFT if config_sift else PATH_SCRIPT_REGISTRATION_BASE

        cmd = COMMAND_REGISTRATION % {
            'exec_Fiji': self.params['exec_Fiji'],
            'path_bsh': path_reg_script,
            'target': path_im_ref,
            'source': path_im_move,
            'output': path_dir,
            'params': ' '.join(map(str, config_sift + config_bunwarpj)),
        }
        return cmd

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param {str: value} record: dictionary with registration params
        :return {str: str}: paths to ...
        """
        logging.debug('.. warp the registered image and get landmarks')
        path_dir = self._get_path_reg_dir(record)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(record, prefer_pproc=False)
        path_log = os.path.join(path_dir, NAME_LOG_REGISTRATION)

        # warp moving landmarks to reference frame
        path_regist = os.path.join(path_dir, os.path.basename(path_im_move))
        dict_params = {
            'exec_Fiji': self.params['exec_Fiji'],
            'path_bsh': PATH_SCRIPT_WARP_LANDMARKS,
            'source': path_im_move,
            'target': path_im_ref,
            'output': path_dir,
            'warp': path_regist,
        }
        # export source points to TXT
        pts_source = load_landmarks(path_lnds_move)
        save_landmarks(os.path.join(path_dir, NAME_LANDMARKS), pts_source)
        # execute transformation
        exec_commands(COMMAND_WARP_LANDMARKS % dict_params, path_logger=path_log)
        # load warped landmarks from TXT
        path_lnds = os.path.join(path_dir, NAME_LANDMARKS_WARPED)
        if os.path.isfile(path_lnds):
            points_warp = load_landmarks(path_lnds)
            path_lnds = os.path.join(path_dir, os.path.basename(path_lnds_move))
            save_landmarks(path_lnds, points_warp)
        else:
            path_lnds = None
        # return results
        return {COL_IMAGE_MOVE_WARP: path_regist,
                COL_POINTS_MOVE_WARP: path_lnds}


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmUnwarpJ)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
