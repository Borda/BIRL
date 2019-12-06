"""
Benchmark for DROP.

DROP is a approach for image registration and motion estimation based on Markov Random Fields.

.. ref:: https://www.mrf-registration.net

Related Publication:
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
        -t ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -DROP ~/Applications/DROP/dropreg2d \
        --path_config ./configs/DROP.txt \
        --visual --unique

.. note:: experiments was tested on Ubuntu (Linux) based OS system

.. note:: to check whether you have all needed libraries on Linux use `ldd dropreg2d`,
 see: https://askubuntu.com/a/709271/863070
 AND set path to the `libdroplib.so` as `export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib`,
 see: https://unix.stackexchange.com/a/67783 ; https://stackoverflow.com/a/49660575/4521646

.. note:: This method is not optimized nor suitable for large images, so all used images
 are first scaled to be 2000x2000 pixels and then the registration is performed.
  After registration is resulting image scaled back. The landmarks are scalded accordingly.

Glocker, Ben, et al. "Deformable medical image registration: setting the state of the art
 with discrete methods." Annual review of biomedical engineering 13 (2011): 219-244.
 https://pdfs.semanticscholar.org/0c5f/277d357e3667b18b5420fd660f221c935fcc.pdf

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import shutil
import logging
import time

import numpy as np
import SimpleITK as sitk

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import (
    convert_image_to_mhd, convert_image_from_mhd, save_landmarks, load_landmarks, image_sizes)
from birl.benchmark import ImRegBenchmark
from bm_experiments import bm_comp_perform


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
    >>> params = {'path_table': path_csv,
    ...           'path_out': path_out,
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'visual': True,
    ...           'exec_DROP': 'dropreg2d',
    ...           'path_config': os.path.join(update_path('configs'), 'DROP.txt')}
    >>> benchmark = BmDROP(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_DROP', 'path_config']
    #: maximal image size (diagonal in pixels) recommended for DROP registration
    MAX_IMAGE_DIAGONAL = int(np.sqrt(2e3 ** 2 + 2e3 ** 2))
    #: time need for image conversion and optional scaling
    COL_TIME_CONVERT = 'conversion time [s]'

    def _prepare(self):
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_config')

    def _prepare_img_registration(self, item):
        """ converting the input images to gra-scale and MHD format

        :param dict item: dictionary with registration params
        :return dict: the same or updated registration info
        """
        logging.debug('.. converting images to MHD')
        path_im_ref, path_im_move, _, _ = self._get_paths(item)
        path_reg_dir = self._get_path_reg_dir(item)

        diags = [image_sizes(p_img)[1] for p_img in (path_im_ref, path_im_move)]
        item['scaling'] = max(1, max(diags) / float(self.MAX_IMAGE_DIAGONAL))

        t_start = time.time()
        for path_img, col in [(path_im_ref, self.COL_IMAGE_REF),
                              (path_im_move, self.COL_IMAGE_MOVE)]:
            item[col + self.COL_IMAGE_EXT_TEMP] = \
                convert_image_to_mhd(path_img, path_out_dir=path_reg_dir, overwrite=False,
                                     to_gray=True, scaling=item.get('scaling', 1.))
        item[self.COL_TIME_CONVERT] = time.time() - t_start
        return item

    def _generate_regist_command(self, item):
        """ generate the registration command

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        logging.debug('.. prepare DROP registration command')
        path_im_ref, path_im_move, _, _ = self._get_paths(item)
        path_dir = self._get_path_reg_dir(item)

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

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        path_reg_dir = self._get_path_reg_dir(item)
        _, path_im_move, path_lnds_ref, _ = self._get_paths(item)
        # convert MHD image
        path_img_ = convert_image_from_mhd(os.path.join(path_reg_dir, 'output.mhd'),
                                           scaling=item.get('scaling', 1.))
        img_name, _ = os.path.splitext(os.path.basename(path_im_move))
        _, img_ext = os.path.splitext(os.path.basename(path_img_))
        path_img_warp = path_img_.replace('output' + img_ext, img_name + img_ext)
        shutil.move(path_img_, path_img_warp)

        # load transform and warp landmarks
        # lnds_move = load_landmarks(path_lnds_move)
        lnds_ref = load_landmarks(path_lnds_ref)
        lnds_name = os.path.basename(path_lnds_ref)
        path_lnds_warp = os.path.join(path_reg_dir, lnds_name)
        assert lnds_ref is not None, 'missing landmarks to be transformed "%s"' % lnds_name

        # down-scale landmarks if defined
        lnds_ref = lnds_ref / item.get('scaling', 1.)
        # extract deformation
        path_deform_x = os.path.join(path_reg_dir, 'output_x.mhd')
        path_deform_y = os.path.join(path_reg_dir, 'output_y.mhd')
        try:
            shift = self.extract_landmarks_shift_from_mhd(path_deform_x, path_deform_y, lnds_ref)
        except Exception:
            logging.exception(path_reg_dir)
            shift = np.zeros(lnds_ref.shape)

        # lnds_warp = lnds_move - shift
        lnds_warp = lnds_ref + shift
        # upscale landmarks if defined
        lnds_warp = lnds_warp * item.get('scaling', 1.)
        save_landmarks(path_lnds_warp, lnds_warp)

        # return formatted results
        return {self.COL_IMAGE_MOVE_WARP: path_img_warp,
                self.COL_POINTS_REF_WARP: path_lnds_warp}

    def _clear_after_registration(self, item, patterns=('output*', '*.mhd', '*.raw')):
        """ clean unnecessarily files after the registration

        :param dict item: dictionary with registration information
        :param list(str) patterns: string patterns of file names
        :return dict: the same or updated registration info
        """
        logging.debug('.. cleaning after registration experiment, remove `output`')
        path_reg_dir = self._get_path_reg_dir(item)
        for ptn in patterns:
            for p_file in glob.glob(os.path.join(path_reg_dir, ptn)):
                os.remove(p_file)
        return item

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-DROP', '--exec_DROP', type=str, required=True,
                                help='path to DROP executable, use `dropreg2d`')
        arg_parser.add_argument('-cfg', '--path_config', type=str, required=True,
                                help='parameters for DROP registration')
        return arg_parser

    @staticmethod
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
            lnds_max = np.max(lnds, axis=0)[::-1]
            assert all(ln < dim for ln, dim in zip(lnds_max, deform_.shape)), \
                'landmarks max %s is larger then (exceeded) deformation shape %s' \
                % (lnds_max.tolist(), deform_.shape)
            shift_ = deform_[lnds[:, 1], lnds[:, 0]]
            return shift_

        lnds = np.array(np.round(lnds), dtype=int)
        # get shift in both axis
        shift_x = __parse_shift(path_deform_x, lnds)
        shift_y = __parse_shift(path_deform_y, lnds)
        # concatenate
        shift = np.array([shift_x, shift_y]).T
        return shift


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmDROP.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
