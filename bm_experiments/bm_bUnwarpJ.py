"""
Benchmark for ImageJ plugin - bUnwarpJ

.. seealso:: http://imagej.net/BUnwarpJ

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
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji ~/Applications/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_bUnwarpJ_histol.yaml \
        --visual --unique

The bUnwarpJ is supporting SIFT and MOPS feature extraction as landmarks
see: http://imagej.net/BUnwarpJ#SIFT_and_MOPS_plugin_support ::

    python bm_experiments/bm_bUnwarpJ.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji ~/Applications/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_bUnwarpJ-SIFT_histol.yaml \
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
from birl.utilities.data_io import load_config_yaml, load_landmarks, save_landmarks, update_path
from birl.utilities.experiments import dict_deep_update, exec_commands
from bm_experiments import bm_comp_perform


class BmUnwarpJ(ImRegBenchmark):
    """ Benchmark for ImageJ plugin - bUnwarpJ
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
    ...           'path_config': fn_path_conf('ImageJ_bUnwarpJ_histol.yaml')}
    >>> benchmark = BmUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> params['path_config'] = fn_path_conf('ImageJ_bUnwarpJ-SIFT_histol.yaml')
    >>> benchmark = BmUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_Fiji', 'path_config']
    #: path to IJ scripts
    PATH_IJ_SCRIPTS = os.path.join(update_path('scripts'), 'ImageJ')
    #: path/name of image registration script
    PATH_SCRIPT_REGISTRATION_BASE = os.path.join(PATH_IJ_SCRIPTS, 'apply-bUnwarpJ-registration.bsh')
    #: path/name of image registration script with features
    PATH_SCRIPT_REGISTRATION_SIFT = os.path.join(PATH_IJ_SCRIPTS, 'apply-SIFT-bUnwarpJ-registration.bsh')
    #: path/name of image/landmarks warping script
    PATH_SCRIPT_WARP_LANDMARKS = os.path.join(PATH_IJ_SCRIPTS, 'apply-bUnwarpJ-transform.bsh')
    # PATH_SCRIPT_HIST_MATCH_IJM = os.path.join(PATH_IJ_SCRIPTS,
    #                                           'histogram-matching-for-macro.bsh')
    #: command for executing the image registration
    COMMAND_REGISTRATION = (
        '%(exec_Fiji)s --headless %(path_bsh)s'
        ' %(source)s %(target)s %(params)s'
        ' %(output)s/transform-direct.txt'
        ' %(output)s/transform-inverse.txt'
    )
    #: internal name of converted landmarks for tranf. script
    NAME_LANDMARKS = 'source_landmarks.pts'
    #: name of warped moving landmarks by tranf. script
    NAME_LANDMARKS_WARPED = 'warped_source_landmarks.pts'
    #: resulting inverse transformation
    NAME_TRANSF_INVERSE = 'transform-inverse.txt'
    #: resulting direct transformation
    NAME_TRANSF_DIRECT = 'transform-direct.txt'
    #: command for executing the warping image and landmarks
    COMMAND_WARP_LANDMARKS = (
        '%(exec_Fiji)s --headless %(path_bsh)s'
        ' %(source)s %(target)s'
        ' %(output)s/' + NAME_LANDMARKS + ' %(output)s/' + NAME_LANDMARKS_WARPED + ' %(transf-inv)s'
        ' %(transf-dir)s'
        ' %(warp)s'
    )
    #: required parameters in the configuration file for bUnwarpJ
    REQUIRED_PARAMS_BUNWARPJ = (
        'mode', 'subsampleFactor', 'minScale', 'maxScale', 'divWeight', 'curlWeight', 'landmarkWeight', 'imageWeight',
        'consistencyWeight', 'stopThreshold'
    )
    #: required parameters in the configuration file for SIFT features
    REQUIRED_PARAMS_SIFT = (
        'initialSigma', 'steps', 'minOctaveSize', 'maxOctaveSize', 'fdSize', 'fdBins', 'rod', 'maxEpsilon',
        'minInlierRatio', 'modelIndex'
    )

    #: default bUnwarpJ and SIFT parameters
    DEFAULT_PARAMS = {
        'bUnwarpJ': {
            'mode': 1,  #: (0-Accurate, 1-Fast, 2-Mono)
            'subsampleFactor': 0,  # (0 = 2^0, 7 = 2^7)
            'minScale': 0,  # (0-Very Coarse, 1-Coarse, 2-Fine, 3-Very Fine)
            'maxScale': 3,  # (0-Very Coarse, 1-Coarse, 2-Fine, 3-Very Fine, 4-Super Fine)
            # weight to penalize divergence
            'divWeight': 0.1,
            #: weight to penalize curl
            'curlWeight': 0.1,
            #: weight to penalize landmark location error
            'landmarkWeight': 0.,
            #: weight to penalize intensity difference
            'imageWeight': 1.,
            #: weight to penalize consistency difference
            'consistencyWeight': 10.,
            #: error function stopping threshold value
            'stopThreshold': 0.01,
        },
        'SIFT': {
            # initial Gaussian blur sigma
            'initialSigma': 1.6,
            #: steps per scale octave
            'steps': 3,
            #: minimum image size in pixels
            'minOctaveSize': 64,
            #: maximum image size in pixels
            'maxOctaveSize': 1024,
            #: feature descriptor size
            'fdSize': 8,
            #: feature descriptor orientation bins
            'fdBins': 8,
            #: closest/next closest ratio
            'rod': 0.92,
            #: maximal alignment error in pixels
            'maxEpsilon': 25,
            #: inlier ratio
            'minInlierRatio': 0.05,
            #: expected transformation of range
            'modelIndex': 1,  # (0:Translation, 1:Rigid, 2:Similarity, 3:Affine, 4:Perspective)
        }
    }

    # assert all(k in DEFAULT_PARAMS['bUnwarpJ'] for k in REQUIRED_PARAMS_BUNWARPJ), \
    #     'default params are missing some required parameters for bUnwarpJ'
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
        config = self.DEFAULT_PARAMS
        config = dict_deep_update(config, load_config_yaml(self.params['path_config']))
        if config['bUnwarpJ']['mode'] >= 2:
            raise ValueError('Mono mode does not supports inverse transform which is need for landmarks warping.')

        config_sift = [config['SIFT'][k] for k in self.REQUIRED_PARAMS_SIFT] \
            if config.get('SIFT', False) else []
        config_bunwarpj = [config['bUnwarpJ'][k] for k in self.REQUIRED_PARAMS_BUNWARPJ]
        path_reg_script = self.PATH_SCRIPT_REGISTRATION_SIFT if config_sift \
            else self.PATH_SCRIPT_REGISTRATION_BASE

        cmd = self.COMMAND_REGISTRATION % {
            'exec_Fiji': self.params['exec_Fiji'],
            'path_bsh': path_reg_script,
            'target': path_im_ref,
            'source': path_im_move,
            'output': path_dir,
            'params': ' '.join(map(str, config_sift + config_bunwarpj)),
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
        path_img_warp = os.path.join(path_dir, os.path.basename(path_im_move))
        dict_params = {
            'exec_Fiji': self.params['exec_Fiji'],
            'path_bsh': self.PATH_SCRIPT_WARP_LANDMARKS,
            'source': path_im_move,
            'target': path_im_ref,
            'output': path_dir,
            'transf-inv': os.path.join(path_dir, self.NAME_TRANSF_INVERSE),
            'transf-dir': os.path.join(path_dir, self.NAME_TRANSF_DIRECT),
            'warp': path_img_warp,
        }
        # export source points to TXT
        pts_source = load_landmarks(path_lnds_move)
        save_landmarks(os.path.join(path_dir, self.NAME_LANDMARKS), pts_source)
        # execute transformation
        exec_commands(self.COMMAND_WARP_LANDMARKS % dict_params, path_logger=path_log, timeout=self.EXECUTE_TIMEOUT)
        # load warped landmarks from TXT
        path_lnds_warp = os.path.join(path_dir, self.NAME_LANDMARKS_WARPED)
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

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-Fiji', '--exec_Fiji', type=str, required=True, help='path to the Fiji executable')
        arg_parser.add_argument(
            '-cfg', '--path_config', required=True, type=str, help='path to the bUnwarpJ configuration'
        )
        return arg_parser


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmUnwarpJ.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
