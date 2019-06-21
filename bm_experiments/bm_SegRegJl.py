"""
Benchmark for proprietary Julia package - SegReg
Developed by BIA group - http://cmp.felk.cvut.cz/~kybic/
 https://www.researchgate.net/lab/Biomedical-Imaging-Algorithms-Jan-Kybic

Registration of segmented images and simultaneous registration and segmentation in Julia.

Kybic, J., & Borovec, J. (2018). Fast registration by boundary sampling and linear programming.
 In International Conference On Medical Image Computing & Computer Assisted Intervention (MICCA).
 Granada. Retrieved from ftp://cmp.felk.cvut.cz/pub/cmp/articles/kybic/Kybic-MICCAI2018.pdf

Installation
------------
1. Install the Julia environment https://julialang.org (preferable Long-term support (LTS) release)
2. Contact prof. Kybic <kybic@fel.cvut.cz> and get SegRegJl package
 https://gitlab.fel.cvut.cz/biomedical-imaging-algorithms/segregjl
3. Install some standard Python libraries (mostly already listed in `requirements.txt` of this repo)
 in addition you need to install Qt support as `pip install PySide2`
4. Run Julia and install required packages::

    using Pkg
    Pkg.add(["Images", "ImageFiltering", "Colors", "ColorTypes", "ImageView", "ImageMagick"])
    Pkg.add(["FixedPointNumbers", "AxisArrays", "OffsetArrays", "DataStructures"])
    Pkg.add(["IterativeSolvers", "MathProgBase", "GLPK", "Gurobi", "JuMP"])
    Pkg.add(["Clustering", "MicroLogging", "NLopt", "NearestNeighbors", "PyCall", "PyPlot"])
    Pkg.status()

5. Try to run registration script::

    pwd()  # actual location
    cd("./segregjl")  # move to package folder
    push!(LOAD_PATH, pwd())
    include("testreg.jl")
    testreg.test_register_fast_rigid()
    include("benchmark_register.jl")


Usage
-----
Script paradigm::

    julia benchmark.jl \
        <moving_image> <static_image> \
        <output_dir> <parameters> \
        <static_image_landmarks>

Run the basic Julia script::

    ~/Applications/julia-1.0.4/bin/julia ~/Applications/segregjl/benchmark.jl \
        ./data_images/rat-kidney_/scale-5pc/Rat-Kidney_PanCytokeratin.jpg \
        ./data_images/rat-kidney_/scale-5pc/Rat-Kidney_HE.jpg \
        ./output/ \
        ./configs/segreg.txt \
        ./data_images/rat-kidney_/scale-5pc/Rat-Kidney_HE.csv

Run the SegRegJl benchmark::

    python bm_experiments/bm_SegRegJl.py \
        -t ./data_images/pairs-imgs-lnds_histol.csv \
        -d ./data_images \
        -o ./results \
        -Julia ~/Applications/julia-1.0.4/bin/julia \
        -script ~/Applications/segregjl/benchmark.jl \
        -cfg ./configs/segreg.txt

.. note:: tested for Julia > 1.x

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import shutil

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import load_landmarks, save_landmarks
from birl.utilities.experiments import create_basic_parse, parse_arg_params
from birl.cls_benchmark import ImRegBenchmark, COL_IMAGE_MOVE_WARP, COL_POINTS_REF_WARP
from birl.bm_template import main
from bm_experiments import bm_comp_perform


# TODO

#: file with exported image registration time
NAME_FILE_TIME = 'time.txt'
#: file with warped landmarks after performed registration
NAME_FILE_LANDMARKS = 'points.pts'
#: file with warped image after performed registration
NAME_FILE_IMAGE = 'warped.jpg'


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-julia', '--exec_julia', type=str, required=True,
                          help='path to the Julia executable', default='julia')
    a_parser.add_argument('-script', '--path_jl_script', required=True,
                          type=str, help='path to the Julia script with registration')
    a_parser.add_argument('-cfg', '--path_config', type=str, required=True,
                          help='parameters for SegReg registration')
    return a_parser


class BmSegRegJl(ImRegBenchmark):
    """ Benchmark for R package - RNiftyReg
    no run test while this method requires manual installation of RNiftyReg

    For the app installation details, see module details.

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> path_csv = os.path.join(update_path('data_images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_out': path_out,
    ...           'path_table': path_csv,
    ...           'nb_workers': 2,
    ...           'unique': False,
    ...           'exec_julia': 'julia',
    ...           'path_jl_script': '.',
    ...           'path_config': os.path.join(update_path('configs'), 'segreg.txt')}
    >>> benchmark = BmSegRegJl(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['exec_julia',
                                                        'path_jl_script',
                                                        'path_config']

    def _prepare(self):
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_jl_script')
        self._copy_config_to_expt('path_config')

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        path_im_ref, path_im_move, path_lnds_ref, _ = self._get_paths(item)
        path_dir = self._get_path_reg_dir(item) + os.path.sep

        cmd = ' '.join([
            self.params['exec_julia'],
            self.params['path_jl_script'],
            path_im_move,
            path_im_ref,
            path_dir,
            self.params['path_config'],
            path_lnds_ref,
        ])
        return cmd

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return tuple(str,str,str,str): paths to
        """
        logging.debug('.. warp the registered image and get landmarks')
        path_dir = self._get_path_reg_dir(item)
        _, path_img_move, _, path_lnds_move = self._get_paths(item)
        path_lnds_warp, path_img_warp = None, None

        # ToDo

        # load warped landmarks from TXT
        path_lnds = os.path.join(path_dir, NAME_FILE_LANDMARKS)
        if os.path.isfile(path_lnds):
            points_warp = load_landmarks(path_lnds)
            path_lnds_warp = os.path.join(path_dir, os.path.basename(path_lnds_move))
            save_landmarks(path_lnds_warp, points_warp)
            os.remove(path_lnds)

        path_regist = os.path.join(path_dir, NAME_FILE_IMAGE)
        if os.path.isfile(path_regist):
            name_img_move = os.path.splitext(os.path.basename(path_img_move))[0]
            ext_img_warp = os.path.splitext(NAME_FILE_IMAGE)[-1]
            path_img_warp = os.path.join(path_dir, name_img_move + ext_img_warp)
            os.rename(path_regist, path_img_warp)

        return {COL_IMAGE_MOVE_WARP: path_img_warp,
                COL_POINTS_REF_WARP: path_lnds_warp}

    def _extract_execution_time(self, item):
        """ if needed update the execution time
        :param dict item: dictionary with registration params
        :return float|None: time in minutes
        """
        path_dir = self._get_path_reg_dir(item)

        # ToDo

        path_time = os.path.join(path_dir, NAME_FILE_TIME)
        with open(path_time, 'r') as fp:
            t_exec = float(fp.read()) / 60.
        return t_exec


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmSegRegJl)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
