"""
General benchmark template for all registration methods.
It also serves for evaluating the input registration pairs
(while no registration is performed, there is only the initial deformation)

EXAMPLE (usage):
>> python bm_registration.py \
    -in ../data/list_pairs_imgs_lnds.csv -out ../output --unique

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import logging
import multiprocessing as mproc

import shutil
import tqdm
import numpy as np
import pandas as pd

import benchmarks.general_utils.io_utils as tl_io
import benchmarks.general_utils.experiments as tl_expt
import benchmarks.general_utils.visualisation as tl_visu
from benchmarks.general_utils.cls_experiment import Experiment

NB_THREADS = int(mproc.cpu_count() * .8)
# some needed files
NAME_CSV_REGIST = 'registration.csv'
NAME_CSV_RESULTS = 'results.csv'
NAME_TXT_RESULTS = 'results.txt'
NAME_LOG_REGIST = 'registration.log'
NAME_IMAGE_REGIST_POINTS = 'registration_landmarks.png'
NAME_IMAGE_REGIST_VISUAL = 'registration_visual.png'
# columns names in cover and also registration table
COL_IMAGE_REF = 'Reference image'
COL_IMAGE_MOVE = 'Moving image'
COL_IMAGE_REF_WARP = 'Reference image, Warped'
COL_IMAGE_MOVE_WARP = 'Moving image, Warped'
COL_POINTS_REF = 'Reference landmarks'
COL_POINTS_MOVE = 'Moving landmarks'
COL_POINTS_REF_WARP = 'Reference landmarks, Warped'
COL_POINTS_MOVE_WARP = 'Moving landmarks, Warped'
COL_REG_DIR = 'Regist. folder'

COVER_COLUMNS = [COL_IMAGE_REF, COL_IMAGE_MOVE,
                 COL_POINTS_REF, COL_POINTS_MOVE]


# fixing ImportError: No module named 'copy_reg' for Python3
if sys.version_info.major == 2:
    import types
    import copy_reg

    def _reduce_method(m):
        # SOLVING issue: cPickle.PicklingError:
        #   Can't pickle <type 'instancemethod'>:
        #       attribute lookup __builtin__.instancemethod failed
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _reduce_method)


class BmRegistration(Experiment):
    """
    General benchmark class for all registration methods.
    It also serves for evaluating the input registration pairs.

    The benchmark has following steps:
    1. check all necessary pathers and required parameters
    2. load cover file and set all paths as absolute
    3. run individual registration experiment in sequence or in parallel
       (nb_jobs > 1); if the particular experiment folder exist (assume
       completed experiment) and skip it
        a) create experiment folder and init experiment
        b) generate execution command
        c) run the command (an option to lock it in single thread)
        d) evaluate experiment, set the expected outputs and visualisation
        e) clean all extra files if any
    4. visualise results abd evaluate registration results

    Running in single thread:
    >>> tl_io.create_dir('output')
    >>> params = {'nb_jobs': 1, 'unique': False, 'path_out': 'output',
    ...           'path_cover': 'data/list_pairs_imgs_lnds.csv'}
    >>> benchmark = BmRegistration(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> shutil.rmtree('output/BmRegistration', ignore_errors=True)
    """

    def __init__(self, params):
        """ initialise benchmark

        :param dict params:  {str: value}
        """
        assert 'unique' in params
        super(BmRegistration, self).__init__(params, params['unique'])
        logging.info(self.__doc__)

    def _check_required_params(self):
        """ check some extra required parameters for this benchmark """
        logging.debug('.. check if the BM have all required parameters')
        super(BmRegistration, self)._check_required_params()
        for param in ['path_cover', 'path_out', 'nb_jobs']:
            assert param in self.params

    def _load_data(self):
        """ loading data, the cover file with all registration pairs """
        logging.info('-> loading data...')
        # loading the csv cover file
        assert os.path.exists(self.params['path_cover']), \
            'path to csv cover not defined'
        self.df_cover = pd.DataFrame().from_csv(self.params['path_cover'],
                                                index_col=None)
        assert all(col in self.df_cover.columns for col in COVER_COLUMNS), \
            'Some required columns are mIssing in the cover file.'
        for col in COVER_COLUMNS:
            # try to find the correct location, calls by test and running
            self.df_cover[col] = self.df_cover[col].apply(
                tl_io.try_find_upper_folders)
            # extend the complete path
            self.df_cover[col] = self.df_cover[col].apply(os.path.abspath)

    def _perform(self):
        """ perform complete benchmark experiment """
        logging.info('-> perform set of experiments...')

        # load existing result of create new entity
        self.path_csv_regist = os.path.join(self.params['path_exp'],
                                            NAME_CSV_REGIST)
        if os.path.exists(self.path_csv_regist):
            logging.info('loading existing csv: "%s"', self.path_csv_regist)
            self.df_regist = pd.DataFrame.from_csv(self.path_csv_regist)
        else:
            self.df_regist = pd.DataFrame()

        # run the experiment in parallel of single thread
        if self.params['nb_jobs'] > 1:
            self.__perform_parallel(self._perform_registration, self.df_cover,
                                    self.path_csv_regist, 'regist. experiments',
                                    self.params['nb_jobs'])
        else:
            self.__perform_one_thread(self._perform_registration,
                                      self.df_cover, self.path_csv_regist,
                                      'registration experiments')

    def __perform_parallel(self, method, in_table, path_csv=None, name='',
                           nb_jobs=NB_THREADS):
        """ running several registration experiments in parallel

        :param method: executed self method which have as input row
        :param DataFrame in_table: table to be iterated by rows
        :param str path_csv: path to temporary saves
        :param str name: name of the process
        :param int nb_jobs: number of experiment running in parallel
        """
        logging.info('-> running %s in parallel, (%i threads)', name, nb_jobs)
        tqdm_bar = tqdm.tqdm(total=len(in_table))
        iter_table = ((idx, dict(row)) for idx, row, in in_table.iterrows())
        mproc_pool = mproc.Pool(nb_jobs)
        for res in mproc_pool.imap(method, iter_table):
            if res is not None:
                self.df_regist = self.df_regist.append(res, ignore_index=True)
                if path_csv is not None:
                    self.df_regist.to_csv(path_csv)
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()

    def __perform_one_thread(self, method, in_table, path_csv=None, name=''):
        """ running only one registration experiment in the time

        :param method: executed self method which have as input row
        :param DataFrame in_table: table to be iterated by rows
        :param str path_csv: path to temporary saves
        :param str name: name of the process
        """
        logging.info('-> running %s in single thread', name)
        tqdm_bar = tqdm.tqdm(total=len(in_table))
        iter_table = ((idx, dict(row)) for idx, row, in in_table.iterrows())
        for res in map(method, iter_table):
            if res is not None:
                self.df_regist = self.df_regist.append(res, ignore_index=True)
                if path_csv is not None:
                    self.df_regist.to_csv(path_csv)
            tqdm_bar.update()

    def _perform_registration(self, df_row):
        """ run single registration experiment with all sub-stages

        :param (int, dict) df_row: tow from iterated table
        """
        idx, dict_row = df_row
        logging.debug('-> perform single registration #%d...', idx)
        # create folder for this particular experiment
        dict_row[COL_REG_DIR] = os.path.join(self.params['path_exp'], str(idx))
        if os.path.exists(dict_row[COL_REG_DIR]):
            logging.warning('particular regist. experiment already exists: '
                            '"%s"', repr(idx))
            return None
        tl_io.create_dir(dict_row[COL_REG_DIR])

        dict_row = self._prepare_single_regist(dict_row)
        str_cmd = self._generate_regist_command(dict_row)
        path_log = os.path.join(dict_row[COL_REG_DIR], NAME_LOG_REGIST)
        # todo, add lock to single thread, create pool with possible thread ids
        # (USE taskset [native], numactl [need install])

        time_start = time.time()
        cmd_result = tl_expt.run_command_line(str_cmd, path_log)
        dict_row['Time [s]'] = time.time() - time_start
        # if the experiment failed, return back
        if not cmd_result:
            return None

        dict_row = self._evaluate_single_regist(dict_row)
        dict_row = self._clean_single_regist(dict_row)
        return dict_row

    def _summarise(self):
        """ summarise complete benchmark experiment """
        logging.info('-> summarise experiment...')
        # load df_regist and compute stat
        self.__perform_one_thread(self.__compute_landmarks_statistic,
                                  self.df_regist, name='compute inaccuracy')
        if self.params['nb_jobs'] > 1:
            # add visualisations
            self.__perform_parallel(self._visualise_regist_results,
                                    self.df_regist, name='compute inaccuracy',
                                    nb_jobs=self.params['nb_jobs'])
        else:
            # add visualisations
            self.__perform_one_thread(self._visualise_regist_results,
                                      self.df_regist, name='visualise results')
        # export stat to csv
        self.df_regist.to_csv(self.path_csv_regist)
        # export simple stat to txt
        self.__expert_summary_txt()

    def __compute_landmarks_statistic(self, df_row):
        """ after successful registration load initial nad estimated landmarks
        afterwords compute various statistic for init, and final alignment

        :param (int, dict) df_row: tow from iterated table
        """
        idx, dict_row = df_row
        # load initial landmarks
        points_ref = tl_io.load_landmarks(dict_row[COL_POINTS_REF])
        points_move = tl_io.load_landmarks(dict_row[COL_POINTS_MOVE])
        # compute statistic
        self.__compute_landmarks_inaccuracy(idx, points_ref, points_move,
                                            'init')
        # load transformed landmarks
        if COL_POINTS_REF_WARP in dict_row:
            points_target = points_ref
            points_estim = tl_io.load_landmarks(dict_row[COL_POINTS_REF_WARP])
        elif COL_POINTS_MOVE_WARP in dict_row:
            points_target = points_move
            points_estim = tl_io.load_landmarks(dict_row[COL_POINTS_MOVE_WARP])
        else:
            logging.error('not allowed scenario: no output landmarks')
        # compute statistic
        self.__compute_landmarks_inaccuracy(idx, points_target, points_estim,
                                            'final')

    def __compute_landmarks_inaccuracy(self, idx, points1, points2, state=''):
        """ compute statistic on two points sets

        :param int idx: index of tha particular record
        :param points1: np.array<nb_points, dim>
        :param points2: np.array<nb_points, dim>
        :param str state: whether it was before of after registration
        """
        dist, stat = tl_expt.compute_points_dist_statistic(points1, points2)
        # update particular idx
        for name in stat:
            col_name = '%s [px] (%s)' % (name, state)
            self.df_regist.set_value(idx, col=col_name, value=stat[name])

    def _visualise_regist_results(self, df_row):
        """ visualise the registration results according what landmarks were
        estimated - in registration or moving frame

        :param (int, dict) df_row: tow from iterated table
        """
        idx, dict_row = df_row
        points_ref = tl_io.load_landmarks(dict_row[COL_POINTS_REF])
        points_move = tl_io.load_landmarks(dict_row[COL_POINTS_MOVE])
        assert COL_IMAGE_REF_WARP in dict_row, 'missing registered image'
        # visualise particular experiment by idx
        if COL_POINTS_REF_WARP in dict_row:
            image_ref = tl_io.load_image(dict_row[COL_IMAGE_REF])
            image_warp = tl_io.load_image(dict_row[COL_IMAGE_REF_WARP])
            points_warp = tl_io.load_landmarks(dict_row[COL_POINTS_REF_WARP])
            # draw image with landmarks
            image = tl_visu.draw_image_points(image_warp, points_warp)
            tl_io.save_image(os.path.join(dict_row[COL_REG_DIR],
                                          NAME_IMAGE_REGIST_POINTS), image)
            fig = tl_visu.draw_regist_landmarks_ref(image_ref, image_warp,
                                        points_ref, points_move, points_warp)
        elif COL_POINTS_MOVE_WARP in dict_row:
            image_move = tl_io.load_image(dict_row[COL_IMAGE_MOVE])
            # image_warp = tl_io.load_image(row['Moving image, Transf.'])
            points_warp = tl_io.load_landmarks(dict_row[COL_POINTS_MOVE_WARP])
            fig = tl_visu.draw_regist_landmartks_move(image_move,
                                     points_ref, points_move, points_warp)
        else:
            logging.error('not allowed scenario: no output image or landmarks')
        path_fig = os.path.join(dict_row[COL_REG_DIR], NAME_IMAGE_REGIST_VISUAL)
        tl_visu.export_figure(path_fig, fig)

    def __expert_summary_txt(self):
        """ export the summary as CSV and TXT """
        path_txt = os.path.join(self.params['path_exp'], NAME_TXT_RESULTS)
        costume_prec = np.arange(0., 1., 0.05)
        df_summary = self.df_regist.describe(percentiles=costume_prec).T
        df_summary['median'] = self.df_regist.median()
        df_summary.sort_index(inplace=True)
        with open(path_txt, 'w') as fp:
            fp.write(tl_expt.string_dict(self.params, 'CONFIGURATION:'))
            fp.write('\n' * 3 + 'RESULTS:\n')
            fp.write('completed regist. experiments: %i' % len(self.df_regist))
            fp.write('\n' * 2)
            fp.write(repr(df_summary[['mean', 'std', 'median', 'min', 'max']]))
            fp.write('\n' * 2)
            fp.write(repr(df_summary[['5%', '25%', '50%', '75%', '95%']]))
        path_csv = os.path.join(self.params['path_exp'], NAME_CSV_RESULTS)
        df_summary.to_csv(path_csv)

    def _prepare_single_regist(self, dict_row):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict dict_row: {str: value}, dictionary with regist. params
        :return dict: {str: value}
        """
        logging.debug('.. no preparing before regist. experiment')
        return dict_row

    def _generate_regist_command(self, dict_row):
        """ generate the registration command

        :param dict_row: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        target_img = os.path.join(dict_row[COL_REG_DIR],
                                  os.path.basename(dict_row[COL_IMAGE_MOVE]))
        target_lnd = os.path.join(dict_row[COL_REG_DIR],
                                  os.path.basename(dict_row[COL_POINTS_MOVE]))
        cmds = ['cp %s %s' % (os.path.abspath(dict_row[COL_IMAGE_MOVE]),
                              target_img),
                'cp %s %s' % (os.path.abspath(dict_row[COL_POINTS_MOVE]),
                              target_lnd)]
        cmd = ' && '.join(cmds)
        return cmd

    def _evaluate_single_regist(self, dict_row):
        """ evaluate rests of the experiment and identity the registered image
        and landmarks when the process finished

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. simulate registration: '
                      'copy the original image and landmarks')
        # detect image
        path_img = os.path.join(dict_row[COL_REG_DIR],
                                os.path.basename(dict_row[COL_IMAGE_MOVE]))
        if os.path.exists(path_img):
            dict_row[COL_IMAGE_REF_WARP] = path_img
        # detect landmarks
        path_lnd = os.path.join(dict_row[COL_REG_DIR],
                                os.path.basename(dict_row[COL_POINTS_MOVE]))
        if os.path.exists(path_lnd):
            dict_row[COL_POINTS_REF_WARP] = path_lnd

        return dict_row

    def _clean_single_regist(self, dict_row):
        """ clean unnecessarily files after the registration

        :param dict_row: {str: value}, dictionary with regist. params
        :return: {str: value}
        """
        logging.debug('.. no cleaning after regist. experiment')
        return dict_row


def main(params):
    """ run the Main of blank experiment

    :param params: {str: value} set of input parameters
    """
    logging.info('running...')
    logging.info(__doc__)
    benchmark = BmRegistration(params)
    benchmark.run()
    del benchmark
    logging.info('Done.')


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = tl_expt.create_basic_parse()
    params = tl_expt.parse_params(parser)
    main(params)
