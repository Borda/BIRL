"""
Benchmark for Elastix.

See references:

* http://elastix.isi.uu.nl/index.php
* https://github.com/SuperElastix/elastix/wiki/Getting-started
* https://blog.yuwu.me/wp-content/uploads/2017/10/elastix_manual_v4.8.pdf

Installation
------------

1. Download compiled executables from https://github.com/SuperElastix/elastix/releases
2. Try to run both executables locally `elastix --help` and `transformix --help`
    * add path to the lib `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/Applications/elastix/lib`
    * define permanent path or copy libraries `cp /elastix/lib/* /usr/local/lib/`

Example
-------

1. Perform sample image registration:
    ```bash
    ~/Applications/elastix/bin/elastix \
        -f ./data_images/images/artificial_reference.jpg \
        -m ./data_images/images/artificial_moving-affine.jpg \
        -out ./results/elastix \
        -p ./configs/elastix_affine.txt
    ```
2. Besides using `transformix` for deforming images, you can also use `transformix`
 to evaluate the transformation at some points. This means that the input points are specified
 in the fixed image domain, since the transformation direction is from fixed to moving image.
 Perform image/points warping:
    ```bash
    ~/Applications/elastix/bin/transformix \
        -tp ./results/elastix/TransformParameters.0.txt \
        -out ./results/elastix \
        -in ./data_images/images/artificial_moving-affine.jpg \
        -def ./data_images/landmarks/artificial_reference.pts
    ```

Usage
-----
Run the basic ANTs registration with original parameters::

    python bm_experiments/bm_elastix.py \
        -t ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -elastix ~/Applications/elastix/bin \
        -cfg ./configs/elastix_affine.txt


.. note:: The origin of VTK coordinate system is in left bottom corner of the image.
 Also the first dimension is horizontal (swapped to matplotlib)

.. note:: For proper confirmation see list of Elastix parameters:
 http://elastix.isi.uu.nl/doxygen/parameter.html

.. note:: If you have any complication with Elastix,
 see https://github.com/SuperElastix/elastix/wiki/FAQ

Klein, Stefan, et al. "Elastix: a toolbox for intensity-based medical image registration."
 IEEE transactions on medical imaging 29.1 (2009): 196-205.
 http://elastix.isi.uu.nl/marius/downloads/2010_j_TMI.pdf

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import shutil

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import load_landmarks, save_landmarks, save_landmarks_pts
from birl.utilities.experiments import exec_commands
from birl.benchmark import ImRegBenchmark
from bm_experiments import bm_comp_perform


class BmElastix(ImRegBenchmark):
    """ Benchmark for Elastix

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_out': path_out,
    ...           'path_table': path_csv,
    ...           'nb_workers': 1,
    ...           'unique': False,
    ...           'path_elastix': '.',
    ...           'path_config': fn_path_conf('elastix_affine.txt')}
    >>> benchmark = BmElastix(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_config']
    #: executable for performing image registration
    EXEC_ELASTIX = 'elastix'
    #: executable for performing image/landmarks transformation
    EXEC_TRANSFX = 'transformix'
    #: default name of warped image (the image extension cam be changed in the config file)
    NAME_IMAGE_WARPED = 'result.*'
    #: default name of warped landmarks
    NAME_LNDS_WARPED = 'outputpoints.txt'
    #: command template for image registration
    COMMAND_REGISTRATION = \
        '%(exec_elastix)s' \
        ' -f %(target)s' \
        ' -m %(source)s' \
        ' -out %(output)s' \
        ' -p %(config)s'
    #: command template for image/landmarks transformation
    COMMAND_TRANSFORMATION = \
        '%(exec_transformix)s' \
        ' -tp %(output)s/TransformParameters.0.txt' \
        ' -out %(output)s' \
        ' -in %(source)s' \
        ' -def %(landmarks)s'

    def _prepare(self):
        """ prepare BM - copy configurations """
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_config')

        def _exec_update(executable):
            is_path = p_elatix and os.path.isdir(p_elatix)
            return os.path.join(p_elatix, executable) if is_path else executable

        p_elatix = self.params.get('path_elastix', '')
        if p_elatix and os.path.isdir(p_elatix):
            logging.info('using local executions from: %s', p_elatix)

        self.exec_elastix = _exec_update(self.EXEC_ELASTIX)
        self.exec_transformix = _exec_update(self.EXEC_TRANSFX)

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(item)

        cmd = self.COMMAND_REGISTRATION % {
            'exec_elastix': self.exec_elastix,
            'target': path_im_ref,
            'source': path_im_move,
            'output': path_dir,
            'config': self.params['path_config'],
        }
        return cmd

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        path_dir = self._get_path_reg_dir(item)
        path_img_ref, path_img_move, path_lnds_ref, path_lnds_move = self._get_paths(item)
        path_img_warp, path_lnds_warp = None, None
        path_log = os.path.join(path_dir, self.NAME_LOG_REGISTRATION)

        name_lnds = os.path.basename(path_lnds_ref)
        path_lnds_local = save_landmarks_pts(os.path.join(path_dir, name_lnds),
                                             load_landmarks(path_lnds_ref))

        # warping the image and points
        cmd = self.COMMAND_TRANSFORMATION % {
            'exec_transformix': self.exec_transformix,
            'source': path_img_move,
            'output': path_dir,
            'landmarks': path_lnds_local,
        }
        exec_commands(cmd, path_logger=path_log, timeout=self.EXECUTE_TIMEOUT)

        # if there is an output image copy it
        path_im_out = glob.glob(os.path.join(path_dir, self.NAME_IMAGE_WARPED))
        if path_im_out:
            path_im_out = sorted(path_im_out)[0]
            _, ext_img = os.path.splitext(path_im_out)
            name_img, _ = os.path.splitext(os.path.basename(path_img_move))
            path_img_warp = os.path.join(path_dir, name_img + ext_img)
            os.rename(path_im_out, path_img_warp)

        path_lnds_out = os.path.join(path_dir, self.NAME_LNDS_WARPED)
        if os.path.isfile(path_lnds_out):
            path_lnds_warp = os.path.join(path_dir, name_lnds)
            lnds = self.parse_warped_points(path_lnds_out)
            save_landmarks(path_lnds_warp, lnds)

        return {self.COL_IMAGE_MOVE_WARP: path_img_warp,
                self.COL_POINTS_REF_WARP: path_lnds_warp}

    def _clear_after_registration(self, item):
        """ clean unnecessarily files after the registration

        :param dict item: dictionary with regist. information
        :return dict: the same or updated regist. info
        """
        logging.debug('.. cleaning after registration experiment, remove `output`')
        path_reg_dir = self._get_path_reg_dir(item)

        for ptn in ('output*', 'result*', '*.txt'):
            for p_file in glob.glob(os.path.join(path_reg_dir, ptn)):
                os.remove(p_file)

        return item

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument('-elastix', '--path_elastix', type=str, required=False,
                                help='path to folder with elastix executables'
                                     ' (if they are not directly callable)')
        arg_parser.add_argument('-cfg', '--path_config', required=True,
                                type=str, help='path to the elastic configuration')
        return arg_parser

    @staticmethod
    def parse_warped_points(path_pts, col_name='OutputPoint'):
        def _parse_lists(cell):
            # get just the string with list
            s_list = cell[cell.index(' = ') + 3:].strip()
            # parse the elements and convert to float
            f_list = list(map(float, s_list[1:-1].strip().split(' ')))
            return f_list

        # load the file as table with custom separator and using the first line
        df = pd.read_csv(path_pts, header=None, sep=';')
        # rename columns according content, it it has following stricture `name = value`
        df.columns = [c[:c.index(' = ')].strip() if '=' in c else n
                      for n, c in zip(df.columns, df.iloc[0])]
        # parse the values for selected column
        vals = df[col_name].apply(_parse_lists).values
        # transform collection of list to matrix
        lnds = np.array(list(vals))
        return lnds


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmElastix.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
