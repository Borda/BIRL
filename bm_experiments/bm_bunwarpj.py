"""
Benchmark for ImageJ plugin - bUnwarpJ
see: http://imagej.net/BUnwarpJ

INSTALLATION:
1. Enter the application folder in this project
    >> cd <BIRL>/applications
2. Download Fiji - https://fiji.sc/
    >> wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
3. Extract the downloaded application
    >> unzip fiji-linux64.zip
4. Try to run Fiji
    >> Fiji.app/ImageJ-linux64

Run the basic bUnwarpJ registration with original parameters:
>> python bm_experiments/bm_bunwarpj.py \
    -in ../data_images/pairs-imgs-lnds_mix.csv \
    -out ../results \
    -fiji ../applications/Fiji.app/ImageJ-linux64 \
    -config ../configs/ImageJ_bUnwarpJ.txt

The bUnwarpJ is supporting SIFT and MOPS feature extraction as landmarks
see: http://imagej.net/BUnwarpJ#SIFT_and_MOPS_plugin_support
>> python bm_experiments/bm_bunwarpj.py \
    -in ../data_images/pairs-imgs-lnds_mix.csv \
    -out ../results \
    -fiji ../applications/Fiji.app/ImageJ-linux64 \
    -config ../configs/ImageJ_bUnwarpJ.txt \
    -sift ../configs/ImageJ_SIFT.txt

NOTE:
* tested for version ImageJ 2.35

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import shutil

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import benchmark.utilities.data_io as tl_io
import benchmark.utilities.experiments as tl_expt
import benchmark.cls_benchmark as bm

NAME_MACRO_REGISTRATION = 'macro_registration.ijm'
NAME_MACRO_WARP_IMAGE = 'macro_warp_image.ijm'
NAME_MACRO_CONVERT_TRANS = 'macro_convert_transform.ijm'
NAME_TXT_TRANSFORM_RAW = 'transform_raw.txt'

# macro for feature extraction - SIFT
MACRO_SIFT = '''
run("Extract SIFT Correspondences",
    "source_image=%(n_source)s target_image=%(n_target)s %(config_SIFT)s");
'''
# macro for feature extraction - MOPS
MACRO_MOPS = '''
run("Extract MOPS Correspondences",
    "source_image=%(n_source)s target_image=%(n_target)s %(config_MOPS)s");
'''

# macro performing the registration
MACRO_REGISTRATION = '''// Registration
//run("Memory & Threads...", "maximum=6951 parallel=1");

print ("-> images opening...");
open("%(source)s");
open("%(target)s");

time = getTime();
print ("-> start feature extraction...");
%(SIFT)s
%(MOPS)s

print ("-> start registration...");
run("bUnwarpJ",
    "source_image=%(n_source)s target_image=%(n_target)s "
    + " registration=%(config_bUnwarpJ)s save_transformations "
    + " save_direct_transformation=%(output)s/direct_transform.txt "
    + " save_inverse_transformation=%(output)s/invers_transform.txt");
time = getTime() - time;
print ("-> registration finished");
print("TIME: " + time + "ms");

// export the time into a file
file = File.open("%(output)s/TIME.txt");
print(file, time);

run("Close All");
run("Quit");
exit();'''

# macro or warping registered image
MACRO_WARP_IMAGE = '''// Transform image
call("bunwarpj.bUnwarpJ_.elasticTransformImageMacro",
     "%(target)s", "%(source)s",
     "%(output)s/direct_transform.txt", "%(warp)s");
// resave image while bunwarpj macro have an issue
open("%(warp)s");
saveAs("%(warp)s");
run("Quit");
exit();
'''

# macro for converting transform to raw displacement
MACRO_CONVERT_TRANSFORM = '''// Convert transformation
call("bunwarpj.bUnwarpJ_.convertToRawTransformationMacro",
     "%(target)s", "%(source)s",
     "%(output)s/invers_transform.txt", "%(raw)s");
run("Quit");
exit();
'''


def extend_parse(a_parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    a_parser.add_argument('-fiji', '--path_fiji', type=str, required=True,
                          help='path to the Fiji executable')
    a_parser.add_argument('-config', '--path_config_bUnwarpJ', required=True,
                          type=str, help='path to the bUnwarpJ configuration')
    a_parser.add_argument('-sift', '--path_config_IJ_SIFT', required=False,
                          type=str, help='path to the ImageJ SIFT configuration')
    a_parser.add_argument('-mops', '--path_config_IJ_MOPS', required=False,
                          type=str, help='path to the ImageJ MOPS configuration')
    return a_parser


def load_parse_bunwarpj_displacement_axis(fp, size, points):
    """ given pointer in the file aiming to the beginning of displacement
     parse all lines and if in the particular line is a point from list
     get its new position

    :param fp: file pointer
    :param (int, int) size: width, height of the image
    :param points: np.array<nb_points, 2>
    :return list: list of new positions on given axis (x/y) for related points
    """
    width, height = size
    points = np.round(points)
    selected_lines = points[:, 1].tolist()
    pos_new = [0] * len(points)

    # walk thor all lined of this displacement field
    for i in range(height):
        line = fp.readline()
        # if the any point is listed in this line
        if i in selected_lines:
            pos = line.rstrip().split()
            # pos = [float(e) for e in pos if len(e)>0]
            assert len(pos) == width
            # find all points in this line
            for j, point in enumerate(points):
                if point[1] == i:
                    pos_new[j] = float(pos[point[0]])
    return pos_new


def load_parse_bunwarpj_displacements_warp_points(path_file, points):
    """ load and parse displacement field for both X and Y coordinated
    and return new position of selected points

    :param str path_file:
    :param points: np.array<nb_points, 2>
    :return: np.array<nb_points, 2>

    >>> fp = open('./my_transform.txt', 'w')
    >>> _= fp.write('''Width=5
    ... Height=4
    ...
    ... X Trans -----------------------------------
    ... 11 12 13 14 15
    ... 11 12 13 14 15
    ... 11 12 13 14 15
    ... 11 12 13 14 15
    ...
    ... Y Trans -----------------------------------
    ... 20 20 20 20 20
    ... 21 21 21 21 21
    ... 22 22 22 22 22
    ... 23 23 23 23 23''') # py2 has no return, py3 returns nb of characters
    >>> fp.close()
    >>> points = np.array([[1, 1], [4, 0], [2, 3]])
    >>> pts = load_parse_bunwarpj_displacements_warp_points(
    ...                             './my_transform.txt', points)
    >>> pts  # doctest: +NORMALIZE_WHITESPACE
    array([[ 12.,  21.],
           [ 15.,  20.],
           [ 13.,  23.]])
    >>> os.remove('./my_transform.txt')
    """
    if not os.path.isfile(path_file):
        logging.warning('missing transform file "%s"', path_file)
        return None

    fp = open(path_file, 'r')
    # read image sizes
    width = int(fp.readline().split('=')[-1])
    height = int(fp.readline().split('=')[-1])
    logging.debug('loaded image size: %i x %i', width, height)
    size = (width, height)
    if not all(np.max(points, axis=0) <= size):
        logging.warning('some points are outside of the transformation domain')
        return None

    # read inter line
    fp.readline()
    fp.readline()
    # read inter line and Transform notation
    points_x = load_parse_bunwarpj_displacement_axis(fp, size, points)

    # read inter line and Transform notation
    fp.readline(), fp.readline()
    # read Y Trans
    points_y = load_parse_bunwarpj_displacement_axis(fp, size, points)
    fp.close()

    points_new = np.vstack((points_x, points_y)).T
    return points_new


class BmBUnwarpJ(bm.ImRegBenchmark):
    """ Benchmark for ImageJ plugin - bUnwarpJ
    no run test while this method requires manual installation of ImageJ

    >>> path_out = tl_io.create_dir('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(tl_io.update_path('configs'), n)
    >>> params = {'nb_jobs': 1, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': os.path.join(tl_io.update_path('data_images'),
    ...                                      'pairs-imgs-lnds_mix.csv'),
    ...           'path_fiji': '.',
    ...           'path_config_bUnwarpJ': fn_path_conf('ImageJ_bUnwarpJ_histo-1k.txt')}
    >>> benchmark = BmBUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> params['path_config_IJ_SIFT'] = fn_path_conf('ImageJ_SIFT_histo-1k.txt')
    >>> benchmark = BmBUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = bm.ImRegBenchmark.REQUIRED_PARAMS + ['path_fiji', 'path_config_bUnwarpJ']

    def _prepare(self):
        """ prepare BM - copy configurations """
        logging.info('-> copy configuration...')
        path_cofig = self.params['path_config_bUnwarpJ']
        shutil.copy(path_cofig, os.path.join(self.params['path_exp'],
                                             os.path.basename(path_cofig)))
        if 'path_config_IJ_SIFT' in self.params:
            path_cofig = self.params['path_config_IJ_SIFT']
            shutil.copy(path_cofig, os.path.join(self.params['path_exp'],
                                                 os.path.basename(path_cofig)))
        if 'path_config_IJ_MOPS' in self.params:
            path_cofig = self.params['path_config_IJ_MOPS']
            shutil.copy(path_cofig, os.path.join(self.params['path_exp'],
                                                 os.path.basename(path_cofig)))

    def _prepare_registration(self, dict_row):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict dict_row: {str: value}, dictionary with regist. params
        :return dict: {str: value}
        """
        logging.debug('.. generate macros before regist. experiment')
        # set the paths for this experiment
        path_dir = dict_row[bm.COL_REG_DIR]
        path_raw = os.path.join(path_dir, NAME_TXT_TRANSFORM_RAW)
        path_im_ref = dict_row[bm.COL_IMAGE_REF]
        path_im_move = dict_row[bm.COL_IMAGE_MOVE]
        name_im_ref, name_im_move = [os.path.basename(p)
                                     for p in [path_im_ref, path_im_move]]
        path_regist = os.path.join(path_dir, os.path.basename(path_im_move))

        with open(self.params['path_config_bUnwarpJ'], 'r') as fp:
            lines = fp.readlines()
        # merge lines into one string
        config_bunwarpj = ' '.join(l.strip() for l in lines)
        # define set of all possoble paremeters for macros generating
        dict_params = {
            'source': path_im_move, 'target': path_im_ref,
            'n_source': name_im_move, 'n_target': name_im_ref,
            'config_bUnwarpJ': config_bunwarpj, 'output': path_dir,
            'warp': path_regist, 'raw': path_raw}

        # if config for SIFT is defined add SIFT extraction
        if 'path_config_IJ_SIFT' in self.params:
            with open(self.params['path_config_IJ_SIFT'], 'r') as fp:
                lines = fp.readlines()
            dict_params['config_SIFT'] = ' '.join(l.strip() for l in lines)
            dict_params['SIFT'] = MACRO_SIFT % dict_params
        else:
            dict_params['SIFT'] = ''

        # if config for MOPS is defined add MOPS extraction
        if 'path_config_IJ_MOPS' in self.params:
            with open(self.params['path_config_IJ_MOPS'], 'r') as fp:
                lines = fp.readlines()
            dict_params['config_MOPS'] = ' '.join(l.strip() for l in lines)
            dict_params['MOPS'] = MACRO_MOPS % dict_params
        else:
            dict_params['MOPS'] = ''

        # generate the registration macro
        macro_regist = MACRO_REGISTRATION % dict_params
        with open(os.path.join(path_dir, NAME_MACRO_REGISTRATION), 'w') as fp:
            fp.write(macro_regist)

        # generate macro or warping registered image
        macro_warp = MACRO_WARP_IMAGE % dict_params
        with open(os.path.join(path_dir, NAME_MACRO_WARP_IMAGE), 'w') as fp:
            fp.write(macro_warp)

        # generate macro for converting transform to raw displacement
        macro_convert = MACRO_CONVERT_TRANSFORM % dict_params
        with open(os.path.join(path_dir, NAME_MACRO_CONVERT_TRANS), 'w') as fp:
            fp.write(macro_convert)

        return dict_row

    def _generate_regist_command(self, dict_row):
        """ generate the registration command

        :param dict_row: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        path_macro = os.path.join(dict_row[bm.COL_REG_DIR],
                                  NAME_MACRO_REGISTRATION)
        cmd = '%s -batch %s' % (self.params['path_fiji'], path_macro)
        return cmd

    def _evaluate_registration(self, dict_row):
        """ evaluate rests of the experiment and identity the registered image
        and landmarks when the process finished

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. warp the registered image and get landmarks')

        path_dir = dict_row[bm.COL_REG_DIR]
        path_log = os.path.join(path_dir, bm.NAME_LOG_REGIST)
        # warp the image
        path_macro = os.path.join(path_dir, NAME_MACRO_WARP_IMAGE)
        cmd = '%s -batch %s' % (self.params['path_fiji'], path_macro)
        tl_expt.run_command_line(cmd, path_logger=path_log)
        path_img = os.path.join(path_dir,
                                os.path.basename(dict_row[bm.COL_IMAGE_MOVE]))
        # detect image
        if os.path.isfile(path_img):
            dict_row[bm.COL_IMAGE_REF_WARP] = path_img

        # convert the transform do obtain displacement field
        path_macro = os.path.join(path_dir, NAME_MACRO_CONVERT_TRANS)
        cmd = '%s -batch %s' % (self.params['path_fiji'], path_macro)
        tl_expt.run_command_line(cmd, path_logger=path_log)

        # load and parse raw transform to detect landmarks
        path_raw = os.path.join(path_dir, NAME_TXT_TRANSFORM_RAW)
        # points_ref = tl_io.load_landmarks(dict_row[bm.COL_POINTS_REF])
        points_move = tl_io.load_landmarks(dict_row[bm.COL_POINTS_MOVE])
        points_warp = load_parse_bunwarpj_displacements_warp_points(
                                                        path_raw, points_move)
        if points_warp is None:
            return dict_row
        path_lnd = os.path.join(path_dir,
                                os.path.basename(dict_row[bm.COL_POINTS_MOVE]))
        tl_io.save_landmarks_csv(path_lnd, points_warp)
        dict_row[bm.COL_POINTS_REF_WARP] = path_lnd

        return dict_row

    def _clear_after_registration(self, dict_row):
        """ clean unnecessarily files after the registration

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. cleaning: remove raw transformation')
        path_raw = os.path.join(dict_row[bm.COL_REG_DIR], NAME_TXT_TRANSFORM_RAW)
        if os.path.isfile(path_raw):
            os.remove(path_raw)
        return dict_row


def main(arg_params):
    """ run the Main of blank experiment

    :param arg_params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = BmBUnwarpJ(arg_params)
    benchmark.run()
    del benchmark
    logging.info('Done.')


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = tl_expt.create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = tl_expt.parse_params(arg_parser)
    main(arg_params)
