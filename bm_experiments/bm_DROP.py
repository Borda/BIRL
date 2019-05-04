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
        -c ./data_images/pairs-imgs-lnds_histol.csv -d ./data_images \
        -o ./results --visual --unique

.. note:: experiments was tested on Linux Ubuntu based system

.. note:: to check whether uoi have all needed libraries on Linux use `ldd dropreg2d`
     https://askubuntu.com/a/709271/863070
     and set path to the `libdroplib.so` as https://unix.stackexchange.com/a/67783

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import (ImRegBenchmark, COL_IMAGE_MOVE, COL_POINTS_MOVE,
                                COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP)


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('--path_config', type=str, required=True,
                          help='parameters for DROP registration')
    return a_parser


class BmTemplate(ImRegBenchmark):
    """ Benchmark for ImageJ plugin - bUnwarpJ
    no run test while this method requires manual installation of ImageJ

    Running in single thread:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> open('sample_config.yaml', 'w').close()
    >>> main({'nb_workers': 1, 'unique': False, 'visual': True,
    ...       'path_out': path_out, 'path_cover': path_csv,
    ...       'path_config': 'sample_config.yaml'}, BmTemplate)  # doctest: +ELLIPSIS
    '...'
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    >>> os.remove('sample_config.yaml')

    Running in multiple parallel threads:
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_workers': 2, 'unique': False, 'visual': True,
    ...           'path_out': path_out, 'path_cover':
    ...            path_csv, 'path_sample_config': ''}
    >>> benchmark = BmTemplate(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_sample_config']

    def _prepare(self):
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_config')

    def _prepare_img_registration(self, record):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. no preparing before registration experiment')
        return record

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        logging.debug('.. simulate registration: '
                      'copy the source image and landmarks, like regist. failed')
        _, path_im_move, _, path_lnds_move = self._get_paths(record)
        path_reg_dir = self._get_path_reg_dir(record)
        name_img = os.path.basename(record[COL_IMAGE_MOVE])
        cmd_img = 'cp %s %s' % (path_im_move, os.path.join(path_reg_dir, name_img))
        name_lnds = os.path.basename(record[COL_POINTS_MOVE])
        cmd_lnds = 'cp %s %s' % (path_lnds_move, os.path.join(path_reg_dir, name_lnds))
        command = [cmd_img, cmd_lnds]
        return command

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param {str: value} record: dictionary with registration params
        :return {str: str}: paths to ...
        """
        path_reg_dir = self._get_path_reg_dir(record)
        # detect image
        path_img = os.path.join(path_reg_dir, os.path.basename(record[COL_IMAGE_MOVE]))
        # detect landmarks
        path_lnd = os.path.join(path_reg_dir, os.path.basename(record[COL_POINTS_MOVE]))
        # return formatted results
        return {COL_IMAGE_MOVE_WARP: path_img,
                COL_POINTS_MOVE_WARP: path_lnd}

    def _extract_execution_time(self, record):
        """ if needed update the execution time

        :param record: {str: value}, dictionary with registration params
        :return float|None: time in minutes
        """
        return 1. / 60  # running constant time 1 sec.

    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param {str: value} record: dictionary with regist. information
        :return {str: value}: the same or updated regist. info
        """
        logging.debug('.. no cleaning after registration experiment')
        return record


def main(params, cls_benchmark):
    """ run the Main of selected experiment

    :param {str: str|float} params: set of input parameters
    :param cls_benchmark: class of selected benchmark
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = cls_benchmark(params)
    benchmark.run()
    path_expt = benchmark.params['path_exp']
    del benchmark
    logging.info('Done.')
    return path_expt


# RUN by given parameters
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmTemplate)

    if arg_params.get('run_comp_benchmark', False):
        # from bm_experiments import bm_comp_perform
        # bm_comp_perform.main(path_expt)
        logging.info('Here you can call the separate benchmark'
                     ' to measure your computer performances.')
