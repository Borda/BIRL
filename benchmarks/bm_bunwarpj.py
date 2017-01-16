"""
Benchmark for ImageJ plugin - bUnwarpJ

INSTALLATION:
1. Enter the application folder in this project
    >> cd <ImRegBenchmark>/applications
2. Download Fiji - https://fiji.sc/
    >> wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
3. Extract the downloaded application
    >> unzip fiji-linux64.zip
4. Try to run Fiji
    >> Fiji.app/ImageJ-linux64

EXAMPLE (usage):
>> python bm_registration.py \
    -in ../data/list_pairs_imgs_lnds.csv -out ../output \
    -fiji ../applications/Fiji.app/ImageJ-linux64 \
    -config ../configs/ImageJ_bUnwarpJ.txt

>> python bm_registration.py \
    -in ../data/list_pairs_imgs_lnds.csv -out ../output \
    -fiji ../applications/Fiji.app/ImageJ-linux64 \
    -config ../configs/ImageJ_bUnwarpJ.txt \
    -sift ../configs/ImageJ_SIFT.txt

NOTE:
* tested for version ImageJ 2.35

Copyright (C) 2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

import shutil

import benchmarks.general_utils.io_utils as tl_io
import benchmarks.general_utils.experiments as tl_expt
import benchmarks.bm_registration as bm

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


def extend_parse(parser):
    """ extent the basic arg parses by some extra required parameters

    :return object:
    """
    # SEE: https://docs.python.org/3/library/argparse.html
    parser.add_argument('-fiji', '--path_fiji', type=str, required=True,
                        help='path to the Fiji executable')
    parser.add_argument('-config', '--path_config_bUnwarpJ', required=True,
                        type=str, help='path to the bUnwarpJ configuration')
    parser.add_argument('-sift', '--path_config_IJ_SIFT', required=False,
                        type=str, help='path to the ImageJ SIFT configuration')
    parser.add_argument('-mops', '--path_config_IJ_MOPS', required=False,
                        type=str, help='path to the ImageJ MOPS configuration')
    return parser


class BmBUnwarpJ(bm.BmRegistration):
    """ Benchmark for ImageJ plugin - bUnwarpJ

    no run test while this method requires manual installation of ImageJ
    >>> tl_io.create_dir('output')
    >>> params = {'nb_jobs': 1, 'unique': False, 'path_out': 'output',
    ...           'path_cover': 'data/list_pairs_imgs_lnds.csv',
    ...           'path_fiji': '.',
    ...           'path_config_bUnwarpJ': 'configs/ImageJ_bUnwarpJ.txt'}
    >>> benchmark = BmBUnwarpJ(params)
    >>> params['path_config_IJ_SIFT'] = 'configs/ImageJ_SIFT.txt'
    >>> benchmark = BmBUnwarpJ(params)
    >>> del benchmark
    >>> shutil.rmtree('output/BmBUnwarpJ', ignore_errors=True)
    """

    def _check_required_params(self):
        """ check some extra required parameters for this benchmark """
        super(BmBUnwarpJ, self)._check_required_params()
        for param in ['path_fiji', 'path_config_bUnwarpJ']:
            assert param in self.params

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

    def _prepare_single_regist(self, dict_row):
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
            dict_params['MOPS'] = MACRO_SIFT % dict_params
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

    def _evaluate_single_regist(self, dict_row):
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
        if os.path.exists(path_img):
            dict_row[bm.COL_IMAGE_REF_WARP] = path_img

        # convert the transform do obtain displacement field
        path_macro = os.path.join(path_dir, NAME_MACRO_CONVERT_TRANS)
        cmd = '%s -batch %s' % (self.params['path_fiji'], path_macro)
        tl_expt.run_command_line(cmd, path_logger=path_log)

        # load and parse raw transform to detect landmarks
        path_raw = os.path.join(path_dir, NAME_TXT_TRANSFORM_RAW)
        # points_ref = tl_io.load_landmarks(dict_row[bm.COL_POINTS_REF])
        points_move = tl_io.load_landmarks(dict_row[bm.COL_POINTS_MOVE])
        points_warp = tl_io.load_parse_bunwarpj_displacements_warp_points(
                                                        path_raw, points_move)
        path_lnd = os.path.join(path_dir,
                                os.path.basename(dict_row[bm.COL_POINTS_MOVE]))
        tl_io.save_landmarks_csv(path_lnd, points_warp)
        dict_row[bm.COL_POINTS_REF_WARP] = path_lnd

        return dict_row

    def _clean_single_regist(self, dict_row):
        """ clean unnecessarily files after the registration

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. cleaning: remove raw transformation')
        os.remove(os.path.join(dict_row[bm.COL_REG_DIR],
                               NAME_TXT_TRANSFORM_RAW))
        return dict_row


def main(params):
    """ run the Main of blank experiment

    :param params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = BmBUnwarpJ(params)
    benchmark.run()
    del benchmark
    logging.info('Done.')


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = tl_expt.create_basic_parse()
    parser = extend_parse(parser)
    params = tl_expt.parse_params(parser)
    main(params)
