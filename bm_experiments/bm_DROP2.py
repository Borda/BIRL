"""
Benchmark for DROP.

DROP is a approach for image registration and motion estimation based on Markov Random Fields.

.. seealso:: https://github.com/biomedia-mira/drop2

Related Publication:

    1. Deformable Medical Image Registration: Setting the State of the Art with Discrete Methods
        Authors: Ben Glocker, Aristeidis Sotiras, Nikos Komodakis, Nikos Paragios
        Published in: Annual Review of Biomedical Engineering, Vol. 12, 2011, pp. 219-244


**Installation for Linux**

    1. Download source code: https://github.com/biomedia-mira/drop2
    2. Install all required libraries such as ITK, and build following the instructions
        OR run building script `build.sh` included in the repository
    3. Test calling the executable `./build/drop/apps/dropreg/dropreg` which should return something like::

        Usage: dropreg --help

Usage
-----

Sample run of DROP2::

    ./dropreg --mode2d
        -s S1.jpg -t HE.jpg -o S1_to_HE.png
        -l --ltype 0 --lsim 1 --llevels 32 32 32 16 16 16 --lsampling 0.2
        -n --nffd 1000 --nsim 1 --nlevels 16 16 16 8 8 8 --nlambda 0.5 --npin

Sample run::

    mkdir ./results
    python bm_experiments/bm_DROP2.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -DROP ~/Applications/DROP2/dropreg \
        --path_config ./configs/DROP2.txt \
        --visual --unique

.. note:: experiments was tested on Ubuntu (Linux) based OS system

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import load_config_args, load_landmarks, save_landmarks
from bm_experiments import bm_comp_perform
from bm_experiments.bm_DROP import BmDROP


class BmDROP2(BmDROP):
    """ Benchmark for DROP2
    no run test while this method requires manual installation of DROP2

    For the app installation details, see module details.

    Example
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data-images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_table': path_csv,
    ...           'path_out': path_out,
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'visual': True,
    ...           'exec_DROP': 'dropreg',
    ...           'path_config': os.path.join(update_path('configs'), 'DROP2.txt')}
    >>> benchmark = BmDROP2(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: command for executing the image registration
    COMMAND_REGISTER = '%(dropRegistration)s \
        --mode2d \
        --source %(source)s \
        --target %(target)s \
        --output %(output)s.jpeg \
        %(config)s'

    def _prepare_img_registration(self, item):
        """ converting the input images to gra-scale and MHD format

        :param dict item: dictionary with registration params
        :return dict: the same or updated registration info
        """
        # this version uses full images
        return item

    def _generate_regist_command(self, item):
        """ generate the registration command

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        logging.debug('.. prepare DROP registration command')
        config = load_config_args(self.params['path_config'])

        path_im_ref, path_im_move, _, _ = self._get_paths(item)
        path_dir = self._get_path_reg_dir(item)

        def __cmd(p_target, p_source):
            command = self.COMMAND_REGISTER % {
                'dropRegistration': self.params['exec_DROP'],
                'source': p_source,
                'target': p_target,
                'output': os.path.join(path_dir, 'output'),
                'config': config,
            }
            return command

        cmd_direct = __cmd(path_im_ref, path_im_move) + ' --ocompose'
        return [cmd_direct]

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        path_reg_dir = self._get_path_reg_dir(item)
        _, path_im_move, path_lnds_ref, path_lnds_move = self._get_paths(item)

        path_img_warp = os.path.join(path_reg_dir, os.path.basename(path_im_move))
        shutil.move(os.path.join(path_reg_dir, 'output.jpeg'), path_img_warp)

        # load transform and warp landmarks
        lnds_ = load_landmarks(path_lnds_ref)
        # this was in case you run inverse registration co you could warp the landmarks directly
        # lnds_ = load_landmarks(path_lnds_move)
        lnds_name = os.path.basename(path_lnds_move)
        path_lnds_warp = os.path.join(path_reg_dir, lnds_name)
        if lnds_ is None:
            raise ValueError('missing landmarks to be transformed "%s"' % lnds_name)

        # extract deformation
        path_deform_x = os.path.join(path_reg_dir, 'output_field_x.nii.gz')
        path_deform_y = os.path.join(path_reg_dir, 'output_field_y.nii.gz')
        try:
            shift = self.extract_landmarks_shift_from_nifty(path_deform_x, path_deform_y, lnds_)
        except Exception:
            logging.exception(path_reg_dir)
            shift = np.zeros(lnds_.shape)

        lnds_warp = lnds_ + shift
        save_landmarks(path_lnds_warp, lnds_warp)

        # return formatted results
        return {
            self.COL_IMAGE_MOVE_WARP: path_img_warp,
            self.COL_POINTS_REF_WARP: path_lnds_warp,
        }

    def _clear_after_registration(self, item, patterns=('output*', '*.nii.gz')):
        """ clean unnecessarily files after the registration

        :param dict item: dictionary with registration information
        :param list(str) patterns: string patterns of file names
        :return dict: the same or updated registration info
        """
        return super(BmDROP2, self)._clear_after_registration(item, patterns)

    @staticmethod
    def extract_landmarks_shift_from_nifty(path_deform_x, path_deform_y, lnds):
        """ given pair of deformation fields and landmark positions get shift

        :param str path_deform_x: path to deformation field in X axis
        :param str path_deform_y: path to deformation field in Y axis
        :param ndarray lnds: landmarks
        :return ndarray: shift for each landmarks
        """

        # define function for parsing particular shift from MHD
        def __parse_shift(path_deform_, axis, lnds):
            if not os.path.isfile(path_deform_):
                raise FileNotFoundError('missing deformation: %s' % path_deform_)
            img_ = sitk.ReadImage(path_deform_)
            spacing = img_.GetSpacing()
            deform_ = sitk.GetArrayFromImage(img_)[0].T
            # deform_ = nibabel.load(path_deform_).get_data()[:, :, 0].T
            if deform_ is None:
                raise ValueError('loaded deformation is Empty - %s' % path_deform_)
            lnds_max = np.max(lnds, axis=0)
            if not all(ln < dim for ln, dim in zip(lnds_max, deform_.shape)):
                raise ValueError(
                    'landmarks max %s is larger then (exceeded) deformation shape %s' %
                    (lnds_max.tolist(), deform_.shape)
                )
            # see: https://github.com/biomedia-mira/drop2/issues/2#issuecomment-547340836
            shift_ = deform_[lnds[:, 0], lnds[:, 1]] / spacing[axis]
            return shift_

        lnds = np.array(np.round(lnds), dtype=int)
        # get shift in both axis
        shift_x = __parse_shift(path_deform_x, 0, lnds)
        shift_y = __parse_shift(path_deform_y, 1, lnds)
        # concatenate
        shift = np.array([shift_x, shift_y]).T
        return shift


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_params, path_expt = BmDROP2.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
