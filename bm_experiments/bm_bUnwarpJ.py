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
>> python bm_experiments/bm_bUnwarpJ.py \
    -c ./data_images/pairs-imgs-lnds_histol.csv \
    -d ./data_images \
    -o ./results \
    -fiji ./applications/Fiji.app/ImageJ-linux64 \
    -config ./configs/ImageJ_bUnwarpJ-pure-image_histol-1k.txt

The bUnwarpJ is supporting SIFT and MOPS feature extraction as landmarks
see: http://imagej.net/BUnwarpJ#SIFT_and_MOPS_plugin_support
>> python bm_experiments/bm_bUnwarpJ.py \
    -c ./data_images/pairs-imgs-lnds_histol.csv \
    -d ./data_images \
    -o ./results \
    -fiji ./applications/Fiji.app/ImageJ-linux64 \
    -config ./configs/ImageJ_bUnwarpJ-landmarks_histol-1k.txt \
    -sift ./configs/ImageJ_SIFT_histol-1k.txt

Disclaimer:
* tested for version ImageJ 1.52i & 2.35

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import time
import logging
import shutil

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import update_path, load_landmarks, save_landmarks
from birl.utilities.experiments import create_basic_parse, parse_arg_params, exec_commands
from birl.cls_benchmark import ImRegBenchmark, NAME_LOG_REGISTRATION
from birl.bm_template import main
from bm_experiments import bm_comp_perform

NAME_MACRO_REGISTRATION = 'macro_registration.ijm'
NAME_MACRO_WARP_IMAGE = 'macro_warp_image.ijm'
PATH_IJ_SCRIPTS = os.path.join(update_path('scripts'), 'ImageJ')
PATH_SCRIPT_WARP_LANDMARKS = os.path.join(PATH_IJ_SCRIPTS, 'apply-bunwarpj-transform.bsh')
PATH_SCRIPT_HIST_MATCHING = os.path.join(PATH_IJ_SCRIPTS, 'histogram-matching.bsh')
PATH_SCRIPT_HIST_MATCH_IJM = os.path.join(PATH_IJ_SCRIPTS, 'histogram-matching-for-macro.bsh')
NAME_LANDMARKS = 'source_landmarks.txt'
NAME_LANDMARKS_WARPED = 'warped_source_landmarks.txt'
COMMAND_WARP_LANDMARKS = '%(path_fiji)s --headless %(path_bsh)s' \
                         ' %(source)s %(target)s' \
                         ' %(output)s/' + NAME_LANDMARKS + \
                         ' %(output)s/' + NAME_LANDMARKS_WARPED + \
                         ' %(output)s/inverse_transform.txt' \
                         ' %(output)s/direct_transform.txt' \
                         ' %(warp)s'
COMMAND_HIST_MATCHING = '%(path_fiji)s --headless %(path_bsh)s' \
                        ' %(target)s %(source)s %(output)s'
# macro for feature extraction - SIFT
MACRO_SIFT = '''
run("Extract SIFT Correspondences",
    "source_image=sourceImage target_image=targetImage %(config_SIFT)s");
'''
# macro for feature extraction - MOPS
MACRO_MOPS = '''
run("Extract MOPS Correspondences",
    "source_image=sourceImage target_image=targetImage %(config_MOPS)s");
'''
# MACRO_HIST_MATCHING = '''
# // open script as text
# bshText = File.openAsString( "%s" );
# // evaluate script
# eval("bsh", bshText );
# ''' % PATH_SCRIPT_HIST_MATCH_IJM
# macro performing the registration
MACRO_REGISTRATION = '''// Registration
//run("Memory & Threads...", "maximum=6951 parallel=1");

print ("-> images opening...");
open("%(source)s");
rename("sourceImage")
open("%(target)s");
rename("targetImage");

time = getTime();
print ("-> start feature extraction...");
%(SIFT)s
%(MOPS)s

print ("-> start registration...");
run("bUnwarpJ",
    "source_image=sourceImage target_image=targetImage "
    + " registration=%(config_bUnwarpJ)s save_transformations "
    + " save_direct_transformation=%(output)s/direct_transform.txt "
    + " save_inverse_transformation=%(output)s/inverse_transform.txt");

time = getTime() - time;
print ("-> registration finished");
print("TIME: " + time + "ms");

// export the time into a file
file = File.open("%(output)s/TIME.txt");
print(file, time);

run("Close All");
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
    a_parser.add_argument('--hist_matching', action='store_true', required=False,
                          default=False, help='apply histogram matching before registration')
    return a_parser


class BmUnwarpJ(ImRegBenchmark):
    """ Benchmark for ImageJ plugin - bUnwarpJ
    no run test while this method requires manual installation of ImageJ

    EXAMPLE
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> params = {'nb_workers': 1, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': os.path.join(update_path('data_images'),
    ...                                      'pairs-imgs-lnds_mix.csv'),
    ...           'path_fiji': '.', 'hist_matching': True,
    ...           'path_config_bUnwarpJ': fn_path_conf('ImageJ_bUnwarpJ-pure-image_histol-1k.txt')}
    >>> benchmark = BmUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> params['path_config_IJ_SIFT'] = fn_path_conf('ImageJ_bUnwarpJ-pure-image_histol-1k.txt')
    >>> benchmark = BmUnwarpJ(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> del benchmark
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_fiji',
                                                        'path_config_bUnwarpJ']

    def _prepare(self):
        """ prepare Benchmark - copy configurations """
        logging.info('-> copy configuration...')

        self._copy_config_to_expt('path_config_bUnwarpJ')
        if 'path_config_IJ_SIFT' in self.params:
            self._copy_config_to_expt('path_config_IJ_SIFT')
        if 'path_config_IJ_MOPS' in self.params:
            self._copy_config_to_expt('path_config_IJ_MOPS')

    def _prepare_registration(self, record):
        """ prepare the experiment folder if it is required,

        * create registration macros

        :param {str: str|float} dict record: dictionary with regist. params
        :return {str: str|float}: the same or updated registration info
        """
        logging.debug('.. generate macros before registration experiment')
        # set the paths for this experiment
        path_dir = self._get_path_reg_dir(record)
        path_im_ref, path_im_move, _, _ = self._get_paths(record)
        path_regist = os.path.join(path_dir, os.path.basename(path_im_move))

        # if histograms matching was selected
        if self.params.get('hist_matching', False):
            path_img_out = os.path.join(path_dir, os.path.basename(path_im_move))
            cmd_hist_match = COMMAND_HIST_MATCHING % {
                'path_fiji': self.params['path_fiji'], 'path_bsh': PATH_SCRIPT_HIST_MATCHING,
                'target': path_im_ref, 'source': path_im_move, 'output': path_img_out
            }
            t_start = time.time()
            path_log = os.path.join(path_dir, NAME_LOG_REGISTRATION)
            exec_commands([cmd_hist_match], path_logger=path_log)
            record['Time hist. matrching'] = (time.time() - t_start) / 60.
            path_im_move = path_img_out

        with open(self.params['path_config_bUnwarpJ'], 'r') as fp:
            lines = fp.readlines()
        # merge lines into one string
        config_bunwarpj = ' '.join(l.strip() for l in lines)
        # define set of all possible parameters for macros generating
        dict_params = {
            'source': path_im_move,
            'target': path_im_ref,
            'config_bUnwarpJ': config_bunwarpj,
            'output': path_dir,
            'warp': path_regist
        }

        dict_params['SIFT'] = 'print("no SIFT performed");'
        # if config for SIFT is defined add SIFT extraction
        if 'path_config_IJ_SIFT' in self.params:
            with open(self.params['path_config_IJ_SIFT'], 'r') as fp:
                lines = fp.readlines()
            dict_params['config_SIFT'] = ' '.join(l.strip() for l in lines)
            dict_params['SIFT'] = MACRO_SIFT % dict_params

        dict_params['MOPS'] = 'print("no MOPS performed");'
        # if config for MOPS is defined add MOPS extraction
        if 'path_config_IJ_MOPS' in self.params:
            with open(self.params['path_config_IJ_MOPS'], 'r') as fp:
                lines = fp.readlines()
            dict_params['config_MOPS'] = ' '.join(l.strip() for l in lines)
            dict_params['MOPS'] = MACRO_MOPS % dict_params

        # dict_params['hist_matching'] = 'print("no histogram-matching performed");'
        # # if histograms matching was selected
        # if self.params.get('hist_matching', False):
        #     dict_params['hist_matching'] = MACRO_HIST_MATCHING

        # generate the registration macro
        macro_regist = MACRO_REGISTRATION % dict_params
        with open(os.path.join(path_dir, NAME_MACRO_REGISTRATION), 'w') as fp:
            fp.write(macro_regist)

        return record

    def _generate_regist_command(self, record):
        """ generate the registration command(s)

        :param {str: str|float} record: dictionary with registration params
        :return str|[str]: the execution commands
        """
        path_macro = os.path.join(self._get_path_reg_dir(record),
                                  NAME_MACRO_REGISTRATION)
        cmd = '%s -batch %s' % (self.params['path_fiji'], path_macro)
        return cmd

    def _extract_warped_image_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to
        """
        logging.debug('.. warp the registered image and get landmarks')
        path_dir = self._get_path_reg_dir(record)
        path_im_ref, path_im_move, _, path_lnds_move = self._get_paths(record)
        path_log = os.path.join(path_dir, NAME_LOG_REGISTRATION)

        # warp moving landmarks to reference frame
        path_regist = os.path.join(path_dir, os.path.basename(path_im_move))
        dict_params = {
            'path_fiji': self.params['path_fiji'],
            'path_bsh': PATH_SCRIPT_WARP_LANDMARKS,
            'source': path_im_move, 'target': path_im_ref,
            'output': path_dir, 'warp': path_regist}
        # export source points to TXT
        pts_source = load_landmarks(path_lnds_move)
        save_landmarks(os.path.join(path_dir, NAME_LANDMARKS), pts_source)
        # execute transformation
        exec_commands(COMMAND_WARP_LANDMARKS % dict_params, path_logger=path_log)
        # load warped landmarks from TXT
        path_lnds = os.path.join(path_dir, NAME_LANDMARKS_WARPED)
        if os.path.isfile(path_lnds):
            points_warp = load_landmarks(path_lnds)
            path_lnds = os.path.join(path_dir, os.path.basename(path_lnds_move))
            save_landmarks(path_lnds, points_warp)
        else:
            path_lnds = None
        return None, path_regist, None, path_lnds

    def _extract_execution_time(self, record):
        """ if needed update the execution time

        :param record: {str: value}, dictionary with registration params
        :return float|None: time in minutes
        """
        path_dir = self._get_path_reg_dir(record)
        path_time = os.path.join(path_dir, 'TIME.txt')
        if not os.path.isfile(path_time):
            return None
        with open(path_time, 'r') as fp:
            exec_time = float(fp.read()) / 1000. / 60.
        # in case used hist. matching were used
        exec_time += record.get('Time hist. matrching', 0)
        return exec_time


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = create_basic_parse()
    arg_parser = extend_parse(arg_parser)
    arg_params = parse_arg_params(arg_parser)
    path_expt = main(arg_params, BmUnwarpJ)

    if arg_params.get('run_comp_benchmark', False):
        logging.info('Running the computer benchmark.')
        bm_comp_perform.main(path_expt)
