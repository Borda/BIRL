"""
Developed a new approach for image registration and motion estimation based
on Markov Random Fields. On this website, you can download our software and test
it for your own research and applications. From time to time, we will provide
an updated version of the software including latest developments and/or new features.
See: https://www.mrf-registration.net/

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

Sample run::

    mkdir ./results
    python birl/bm_DROP.py \
        -c ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -DROP ~/Applications/DROP/dropreg2d \
        --path_config ./configs/drop.txt
        --visual --unique

.. note:: experiments was tested on Linux Ubuntu based system

.. note:: to check whether uoi have all needed libraries on Linux use `ldd dropreg2d`
 https://askubuntu.com/a/709271/863070
 and set path to the `libdroplib.so` as `export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib`
 https://unix.stackexchange.com/a/67783

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import glob
import shutil
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import convert_to_mhd, convert_from_mhd
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import (
    ImRegBenchmark, COL_IMAGE_REF, COL_IMAGE_MOVE, COL_POINTS_MOVE, COL_IMAGE_EXT_TEMP,
    COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP)
from birl.bm_template import main
from bm_experiments import bm_comp_perform


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-DROP', '--path_drop', type=str, required=True,
                          help='path to DROP executable, use `dropreg2d`')
    a_parser.add_argument('--path_config', type=str, required=True,
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
    >>> params = {'nb_workers': 2, 'unique': False, 'visual': True,
    ...           'path_out': path_out, 'path_cover': path_csv,
    ...           'path_config': ''}
    >>> benchmark = BmDROP(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_drop',
                                                        'path_config']

    def _prepare(self):
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_config')

    def _prepare_img_registration(self, record):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. converting images to MHD')
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(record)

        for col, path_img in [(COL_IMAGE_REF, path_im_ref),
                              (COL_IMAGE_MOVE, path_im_move)]:
            record[col + COL_IMAGE_EXT_TEMP] = convert_to_mhd(path_img, to_gray=True)

        return record

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        logging.debug('.. prepare DROP registration command')
        path_im_ref, path_im_move, _, _ = self._get_paths(record)
        path_dir = self._get_path_reg_dir(record)

        command = '%s "%s" "%s" %s %s' % (self.params['path_drop'],
                                          path_im_move,
                                          path_im_ref,
                                          os.path.join(path_dir, 'output'),
                                          self.params['path_config'])

        return command

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param {str: value} record: dictionary with registration params
        :return {str: str}: paths to ...
        """
        path_reg_dir = self._get_path_reg_dir(record)
        _, path_im_move, _, path_lnds_move = self._get_paths(record)
        # convert MHD image
        path_img_ = convert_from_mhd(os.path.join(path_reg_dir, 'output.mhd'))
        img_name, ext_img = os.path.splitext(os.path.basename(path_im_move))[0]
        path_img = path_img_.replace('output' + ext_img, img_name + ext_img)
        shutil.move(path_img_, path_img)

        lnds_name = os.path.basename(path_lnds_move)
        path_lnd = os.path.join(path_reg_dir, lnds_name)

        # TODO - load transform and warp landmarks

        # return formatted results
        return {COL_IMAGE_MOVE_WARP: path_img,
                COL_POINTS_MOVE_WARP: path_lnd}

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


# def loadTransformedLandmarksPartialAxis_MHD (matlab, libs, transMHD, landmarks, fLog=None) :
#     # FIXME - REMOVE
#     transTXT = transMHD.replace('.mhd', '.txt')
#     convertTransform_mhd2txt (matlab, libs, transMHD, transTXT, fLog)
#
#     lndAxis = [lnd[1] for lnd in landmarks]
#     lndAxisNew = [0] * len(landmarks)
#
#     i = 0
#     f = open(transTXT, 'r')
#     for line in f :
#         if i in lndAxis :
#             elems = string.split(line.rstrip(),',')
#             # elems = [float(e) for e in elems]
#             for j in range(len(landmarks)) :
#                 lnd = landmarks[j]
#                 if lnd[1] == i :
#                     # logger.debug('grid size is ~{}x{} anf lnd position is {}'.format(len(elems), i, lnd))
#                     lndAxisNew[j] = float( elems[ lnd[0] ] )
#                     # overflow the transform dim - mozna je transformace zmasena vzhledem k velikosti obrazku...
#                     # if len(elems) >= lnd[0] :
#                     #     lndAxisNew[j] = float( elems[ lnd[0] ] )
#                     # else : # if out of range
#                     #     lndAxisNew[j] = float( elems[-1] )
#                     #     logger.warning('elem size is {} but landmark pos was {}'.format(len(elems), lnd[0]))
#         i += 1
#     f.close()
#     return lndAxisNew
#
# def loadTransformedLandmarks_MHD (matlab, libs, transMHD, landmarks, fLog=None) :
#     # FIXME - REMOVE
#     logger.info ( ' loadTransformedLandmarks_MHD: -> exist: {} - file with transform: {}'.format(os.path.isfile(transMHD), transMHD) )
#     logger.debug('landmarks: {}'.format(landmarks))
#     # read X Trans
#     fNameAxis_X = transMHD.replace('.mhd','_x.mhd')
#     lndShift_X = loadTransformedLandmarksPartialAxis_MHD (matlab, libs, fNameAxis_X, landmarks, fLog)
#     logger.debug ( ' loadTransformedLandmarks_MHD: {}'.format(lndShift_X) )
#
#     # read Y Trans
#     fNameAxis_Y = transMHD.replace('.mhd','_y.mhd')
#     lndShift_Y = loadTransformedLandmarksPartialAxis_MHD (matlab, libs, fNameAxis_Y, landmarks, fLog)
#     logger.debug ( ' loadTransformedLandmarks_MHD: {}'.format(lndShift_Y) )
#
#     # landmarks = np.array(landmarks)
#     lnd_X = [landmarks[i][0]-lndShift_Y[i] for i in range(len(landmarks))]
#     lnd_Y = [landmarks[i][1]-lndShift_X[i] for i in range(len(landmarks))]
#     # lndTrans = np.array( zip(lnd_X, lnd_Y) )
#     lndTrans = zip(lnd_X, lnd_Y)
#     return lndTrans


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
