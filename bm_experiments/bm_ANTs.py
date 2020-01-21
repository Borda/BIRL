"""
Benchmark for ANTs
see:
* http://stnava.github.io/ANTs
* https://sourceforge.net/projects/advants/
* https://github.com/stnava/ANTsDoc/issues/1

INSTALLATION:
See: https://brianavants.wordpress.com/2012/04/13/updated-ants-compile-instructions-april-12-2012/

* Do NOT download the binary code, there is an issue:

    - https://sourceforge.net/projects/advants/files/ANTS/ANTS_Latest
    - https://github.com/ANTsX/ANTs/issues/733

* Compile from source::

    git clone git://github.com/stnava/ANTs.git
    mkdir antsbin
    cd antsbin
    ccmake ../ANTs
    make -j$(nproc)

Discussion
----------
I. believes he found the "problem" and indeed it has to do with the file format we use (JPEG).
I. converts the kidney images to 8-bit and then .NII.GZ and the whole pipeline works fine.

.. note:: For showing parameter/options setting run `antsRegistration --help`

1) Convert images to 8-bit using Fiji (this is only because I didn't see any ITK format to store RGB images).
2) Convert the 8-bit images to .nii.gz (using ANTs script `ConvertImagePixelType`)::

    ConvertImagePixelType \
        Rat_Kidney_PanCytokeratin-8bit.png \
        Rat_Kidney_PanCytokeratin.nii.gz \
        1

3) Register images using `antsRegistrationSyN.sh`::

    antsRegistrationSyN.sh \
        -d 2 \
        -m Rat_Kidney_PanCytokeratin.nii.gz \
        -f Rat_Kidney_HE.nii.gz \
        -j 1 \
        -t s \
        -o output \
        > stdout-reg.txt 2> stderr-reg.txt

4) Apply transform to points::

    antsApplyTransformsToPoints \
        -d 2 \
        -i Rat_Kidney_PanCytokeratin.csv \
        -o testPointsHE.csv \
        -t [ output0GenericAffine.mat, 1 ] \
        -t output1InverseWarp.nii.gz

Usage
-----
Run the basic ANT registration with original parameters::

    python bm_experiments/bm_ANTs.py \
        -t ./data_images/pairs-imgs-lnds_anhir.csv \
        -d ./data_images \
        -o ./results \
        --path_ANTs ~/Applications/antsbin/bin \
        --path_config ./configs/ANTs_SyN.txt


.. note:: it was needed to use own compiled version

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import glob
import shutil

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import (
    load_landmarks, save_landmarks, load_config_args,
    convert_image_to_nifti_gray, convert_image_from_nifti)
from birl.utilities.experiments import exec_commands
from birl.benchmark import ImRegBenchmark
from bm_experiments import bm_comp_perform


class BmANTs(ImRegBenchmark):
    """ Benchmark for ANTs
    no run test while this method requires manual compilation of ANTs

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
    ...           'path_ANTs': '.',
    ...           'path_config': '.'}
    >>> benchmark = BmANTs(params)
    >>> benchmark.EXECUTE_TIMEOUT
    10800
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: timeout for executing single image registration
    EXECUTE_TIMEOUT = 3 * 60 * 60  # default = 3 hour
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_config']
    #: executable for performing image registration
    EXEC_REGISTRATION = 'antsRegistration'
    #: executable for performing image transformation
    EXEC_TRANSFORM_IMAGE = 'antsApplyTransforms'
    #: executable for performing landmarks transformation
    EXEC_TRANSFORM_POINTS = 'antsApplyTransformsToPoints'
    #: command for executing the image registration
    COMMAND_REGISTER = '%(antsRegistration)s \
        --dimensionality 2 \
        %(config)s \
        --output [%(output)s/trans]'
    #: command for executing the warping image
    COMMAND_WARP_IMAGE = '%(antsApplyTransforms)s \
        --dimensionality 2 \
        --input %(img_source)s \
        --output %(output)s/%(img_name)s.nii \
        --reference-image %(img_target)s \
        --transform %(transfs)s \
        --interpolation Linear'
    #: command for executing the warping landmarks
    COMMAND_WARP_POINTS = '%(antsApplyTransformsToPoints)s \
        --dimensionality 2 \
        --input %(path_points)s \
        --output %(output)s/%(pts_name)s.csv \
        --transform %(transfs)s'
    #: column name of temporary Nifty reference image
    COL_IMAGE_REF_NII = ImRegBenchmark.COL_IMAGE_REF + ' Nifty'
    #: column name of temporary Nifty noving image
    COL_IMAGE_MOVE_NII = ImRegBenchmark.COL_IMAGE_MOVE + ' Nifty'

    def _prepare(self):
        """ prepare BM - copy configurations """
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_config')

        # this is not possible since the executables can be in std path
        # REQUIRED_EXECUTABLES = (self.EXEC_REGISTRATION,
        #                         self.EXEC_TRANSFORM_IMAGE,
        #                         self.EXEC_TRANSFORM_POINTS)
        # path_execs = [os.path.join(self.params['path_ANTs'], execute)
        #               for execute in REQUIRED_EXECUTABLES]
        # assert all(os.path.isfile(p) for p in path_execs), \
        #     'Some executables are missing: %r' \
        #     % [p for p in path_execs if not os.path.isfile(p)]

        p_ants = self.params.get('path_ANTs', '')
        if p_ants and os.path.isdir(p_ants):
            logging.info('using local executions from: %s', p_ants)

        def _exec_update(executable):
            is_path = p_ants and os.path.isdir(p_ants)
            return os.path.join(p_ants, executable) if is_path else executable

        self.exec_register = _exec_update(self.EXEC_REGISTRATION)
        self.exec_transf_img = _exec_update(self.EXEC_TRANSFORM_IMAGE)
        self.exec_transf_pts = _exec_update(self.EXEC_TRANSFORM_POINTS)

    def _prepare_img_registration(self, item):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict item: dictionary with regist. params
        :return dict: the same or updated registration info
        """
        logging.debug('.. generate command before registration experiment')
        # set the paths for this experiment
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, _ = self._get_paths(item)

        # Convert images to Nifty
        try:  # catching issue with too large images
            item[self.COL_IMAGE_REF_NII] = convert_image_to_nifti_gray(path_im_ref, path_dir)
        except Exception:
            logging.exception('Converting: %s', path_im_ref)
            return None
        try:  # catching issue with too large images
            item[self.COL_IMAGE_MOVE_NII] = convert_image_to_nifti_gray(path_im_move, path_dir)
        except Exception:
            logging.exception('Converting: %s', path_im_move)
            return None

        return item

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary {str: str|float} with registration params
        :return str|list(str): the execution commands
        """
        path_dir = self._get_path_reg_dir(item)

        config = load_config_args(self.params['path_config'])
        config = config % {
            'target-image': item[self.COL_IMAGE_REF_NII],
            'source-image': item[self.COL_IMAGE_MOVE_NII]
        }
        cmd = self.COMMAND_REGISTER % {
            'config': config,
            'antsRegistration': self.exec_register,
            'output': path_dir
        }

        return cmd

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary {str: value} with registration params
        :return dict: paths to results
        """
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(item)
        name_im_move, _ = os.path.splitext(os.path.basename(path_lnds_move))
        name_lnds_move, _ = os.path.splitext(os.path.basename(path_lnds_move))

        # simplified version of landmarks
        lnds = load_landmarks(path_lnds_move)
        path_lnds_warp = os.path.join(path_dir, name_lnds_move + '.csv')
        # https://github.com/ANTsX/ANTs/issues/733#issuecomment-472049427
        pd.DataFrame(lnds * -1, columns=['x', 'y']).to_csv(path_lnds_warp, index=None)

        # list output transformations
        tf_elast_inv = sorted(glob.glob(os.path.join(path_dir, 'trans*InverseWarp.nii*')))
        tf_elast = [os.path.join(os.path.dirname(p), os.path.basename(p).replace('Inverse', ''))
                    for p in tf_elast_inv]
        tf_affine = sorted(glob.glob(os.path.join(path_dir, 'trans*GenericAffine.mat')))
        # generate commands
        cmd_warp_img = self.COMMAND_WARP_IMAGE % {
            'antsApplyTransforms': self.exec_transf_img,
            'output': path_dir,
            'img_target': item[self.COL_IMAGE_REF_NII],
            'img_source': item[self.COL_IMAGE_MOVE_NII],
            'transfs': ' -t '.join(sorted(tf_affine + tf_elast, reverse=True)),
            'img_name': name_im_move
        }
        cmd_warp_pts = self.COMMAND_WARP_POINTS % {
            'antsApplyTransformsToPoints': self.exec_transf_pts,
            'output': path_dir,
            'path_points': path_lnds_warp,
            'transfs': ' -t '.join(['[ %s , 1]' % tf if 'Affine' in tf else tf
                                    for tf in sorted(tf_affine + tf_elast_inv)]),
            'pts_name': name_lnds_move
        }
        # execute commands
        exec_commands([cmd_warp_img, cmd_warp_pts],
                      path_logger=os.path.join(path_dir, 'warping.log'))

        path_im_nii = os.path.join(path_dir, name_im_move + '.nii')
        if os.path.isfile(path_im_nii):
            path_img_warp = convert_image_from_nifti(path_im_nii)
        else:
            path_img_warp = None

        if os.path.isfile(path_lnds_warp):
            lnds = pd.read_csv(path_lnds_warp, index_col=None).values
            save_landmarks(path_lnds_warp, lnds * -1)
        else:
            path_lnds_warp = None

        return {self.COL_IMAGE_MOVE_WARP: path_img_warp,
                self.COL_POINTS_MOVE_WARP: path_lnds_warp}

    def _clear_after_registration(self, item):
        """ clean unnecessarily files after the registration

        :param dict item: dictionary with regist. information
        :return dict: the same or updated registration info
        """
        for ext in ['*.nii', '*.nii.gz', '*.mat']:
            for p in glob.glob(os.path.join(self._get_path_reg_dir(item), ext)):
                os.remove(p)
        del item[self.COL_IMAGE_REF_NII]
        del item[self.COL_IMAGE_MOVE_NII]
        return item

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-ANTs', '--path_ANTs', type=str, required=False,
                                help='path to the ANTs executables'
                                     ' (if they are not directly callable)')
        arg_parser.add_argument('-cfg', '--path_config', required=True,
                                type=str, help='path to the ANTs regist. configuration')
        return arg_parser


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmANTs.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
