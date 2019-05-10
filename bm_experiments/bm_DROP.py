"""
Developed a new approach for image registration and motion estimation based
on Markov Random Fields.

.. ref:: https://www.mrf-registration.net

Related Publication
Deformable Medical Image Registration: Setting the State of the Art with Discrete Methods
Authors: Ben Glocker, Aristeidis Sotiras, Nikos Komodakis, Nikos Paragios
Published in: Annual Review of Biomedical Engineering, Vol. 12, 2011, pp. 219-244


Installation for Linux
----------------------

1. Download executable according your operation system,
 https://www.mrf-registration.net/deformable/index.html
2. Copy/extract executables and libraries to you favourite destination
3. Install all missing libraries such as QT4 with OpenGL support
4. Test calling the executable `./dropreg2d` which should return something like::

    Usage: dropreg2d <source> <target> <result> <paramfile> [mask]

Usage
-----

To see the explanation of particular parameters see the User Manual
 http://www.mrf-registration.net/download/drop_user_guide_V1.05.pdf

Sample run::

    mkdir ./results
    python bm_experiments/bm_DROP.py \
        -c ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -DROP ~/Applications/DROP/dropreg2d \
        --path_config ./configs/drop.txt \
        --visual --unique

.. note:: experiments was tested on Linux Ubuntu based system

.. note:: to check whether uoi have all needed libraries on Linux use `ldd dropreg2d`,
 see: https://askubuntu.com/a/709271/863070
 AND set path to the `libdroplib.so` as `export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib`,
 see: https://unix.stackexchange.com/a/67783 ; https://stackoverflow.com/a/49660575/4521646

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import glob
import shutil
import logging

import numpy as np
import SimpleITK as sitk

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import (
    convert_to_mhd, convert_from_mhd, save_landmarks, load_landmarks)
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import (
    ImRegBenchmark, COL_IMAGE_REF, COL_IMAGE_MOVE, COL_IMAGE_EXT_TEMP,
    COL_IMAGE_MOVE_WARP, COL_POINTS_REF_WARP)
from birl.bm_template import main
from bm_experiments import bm_comp_perform


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-DROP', '--exec_DROP', type=str, required=True,
                          help='path to DROP executable, use `dropreg2d`')
    a_parser.add_argument('-config', '--path_config', type=str, required=True,
                          help='parameters for DROP registration')
    return a_parser


class BmDROP(ImRegBenchmark):
    """ Benchmark for DROP
    no run test while this method requires manual installation of DROP

    For the app installation details, see module details.

    .. note:: DROP requires gray scale images in MHD format where pixel values
    are in range (0, 255) of uint8.

    Example
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_cover': path_csv,
    ...           'path_out': path_out,
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'visual': True,
    ...           'exec_DROP': 'dropreg2d',
    ...           'path_config': os.path.join(update_path('configs'), 'drop.txt')}
    >>> benchmark = BmDROP(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_DROP', 'path_config']

    def _prepare(self):
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_config')

    def _prepare_img_registration(self, record):
        """ converting the input images to gra-scale and MHD format

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. converting images to MHD')
        path_im_ref, path_im_move, _, _ = self._get_paths(record)

        convert_queue = [(path_im_ref, COL_IMAGE_REF), (path_im_move, COL_IMAGE_MOVE)]

        for path_img, col in convert_queue:
            record[col + COL_IMAGE_EXT_TEMP] = \
                convert_to_mhd(path_img, to_gray=True, overwrite=False)

        # def __wrap_convert_mhd(path_img, col):
        #     path_img = convert_to_mhd(path_img, to_gray=True, overwrite=False)
        #     return path_img, col
        #
        # for path_img, col in iterate_mproc_map(__wrap_convert_mhd, convert_queue):
        #     record[col + COL_IMAGE_EXT_TEMP] = path_img

        return record

    def _generate_regist_command(self, record):
        """ generate the registration command

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        logging.debug('.. prepare DROP registration command')
        path_im_ref, path_im_move, _, _ = self._get_paths(record)
        path_dir = self._get_path_reg_dir(record)

        # NOTE: for some reason " in the command makes it crash in Run mode
        # somehow it works fine even with " in Debug mode
        command = ' '.join([
            self.params['exec_DROP'],
            path_im_move,
            path_im_ref,
            os.path.join(path_dir, 'output'),
            self.params['path_config'],
        ])

        return command

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param {str: value} record: dictionary with registration params
        :return {str: str}: paths to ...
        """
        path_reg_dir = self._get_path_reg_dir(record)
        _, path_im_move, path_lnds_ref, _ = self._get_paths(record)
        # convert MHD image
        path_img_ = convert_from_mhd(os.path.join(path_reg_dir, 'output.mhd'))
        img_name = os.path.splitext(os.path.basename(path_im_move))[0]
        ext_img = os.path.splitext(os.path.basename(path_img_))[1]
        path_img = path_img_.replace('output' + ext_img, img_name + ext_img)
        shutil.move(path_img_, path_img)

        # load transform and warp landmarks
        lnds_name = os.path.basename(path_lnds_ref)
        path_lnd = os.path.join(path_reg_dir, lnds_name)
        # lnds_move = load_landmarks(path_lnds_move)
        lnds_ref = load_landmarks(path_lnds_ref)

        path_deform_x = os.path.join(path_reg_dir, 'output_x.mhd')
        path_deform_y = os.path.join(path_reg_dir, 'output_y.mhd')
        shift = extract_landmarks_shift_from_mhd(path_deform_x, path_deform_y, lnds_ref)

        # lnds_warp = lnds_move - shift
        lnds_warp = lnds_ref + shift
        save_landmarks(path_lnd, lnds_warp)

        # return formatted results
        return {COL_IMAGE_MOVE_WARP: path_img,
                COL_POINTS_REF_WARP: path_lnd}

    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param {str: value} record: dictionary with regist. information
        :return {str: value}: the same or updated regist. info
        """
        logging.debug('.. cleaning after registration experiment, remove `output`')
        path_reg_dir = self._get_path_reg_dir(record)
        for p_file in glob.glob(os.path.join(path_reg_dir, 'output*')):
            os.remove(p_file)
        return record


def extract_landmarks_shift_from_mhd(path_deform_x, path_deform_y, lnds):
    """ given pair of deformation fields and landmark positions get shift

    :param str path_deform_x: path to deformation field in X axis
    :param str path_deform_y: path to deformation field in Y axis
    :param ndarray lnds: landmarks
    :return ndarray: shift for each landmarks
    """
    # define function for parsing particular shift from MHD
    def __parse_shift(path_deform_, lnds):
        assert os.path.isfile(path_deform_), 'missing deformation: %s' % path_deform_
        deform_ = sitk.GetArrayFromImage(sitk.ReadImage(path_deform_))
        assert deform_ is not None, 'loaded deformation is Empty - %s' % path_deform_
        shift_ = deform_[lnds[:, 1], lnds[:, 0]]
        return shift_

    # get shift in both axis
    shift_x = __parse_shift(path_deform_x, lnds)
    shift_y = __parse_shift(path_deform_y, lnds)
    # concatenate
    shift = np.array([shift_x, shift_y]).T
    return shift


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmDROP)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
