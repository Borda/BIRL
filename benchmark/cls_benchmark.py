"""
General benchmark template for all registration methods.
It also serves for evaluating the input registration pairs
(while no registration is performed, there is only the initial deformation)

EXAMPLE (usage):
>> mkdir results
>> python benchmarks/bm_registration.py \
    -c data_images/pairs-imgs-lnds_mix.csv -o results --unique

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import time
import logging
import multiprocessing as mproc

import tqdm
import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import benchmark.utilities.data_io as tl_io
import benchmark.utilities.experiments as tl_expt
import benchmark.utilities.visualisation as tl_visu
from benchmark.utilities.cls_experiment import Experiment

NB_THREADS = mproc.cpu_count()
NB_THREADS_USED = max(1, int(NB_THREADS * .8))
# some needed files
NAME_CSV_REGISTRATION_PAIRS = 'registration-results.csv'
NAME_CSV_RESULTS = 'results-summary.csv'
NAME_TXT_RESULTS = 'results-summary.txt'
NAME_LOG_REGISTRATION = 'registration.log'
NAME_IMAGE_MOVE_WARP_POINTS = 'image_warped_landmarks_warped.jpg'
NAME_IMAGE_REF_POINTS_WARP = 'image_ref_landmarks_warped.jpg'
NAME_IMAGE_WARPED_VISUAL = 'registration_visual_landmarks.jpg'
# columns names in cover and also registration table
COL_IMAGE_REF = 'Target image'
COL_IMAGE_MOVE = 'Source image'
# reference image warped to the moving frame
COL_IMAGE_REF_WARP = 'Warped target image'
# moving image warped to the reference frame
COL_IMAGE_MOVE_WARP = 'Warped source image'
COL_POINTS_REF = 'Target landmarks'
COL_POINTS_MOVE = 'Source landmarks'
# reference landmarks warped to the moving frame
COL_POINTS_REF_WARP = 'Warped target landmarks'
# moving landmarks warped to the reference frame
COL_POINTS_MOVE_WARP = 'Warped source landmarks'
# registration folder for each particular experiment
COL_REG_DIR = 'Registration folder'
COL_TIME = 'Execution time [minutes]'

# list of columns in cover csv
COVER_COLUMNS = (COL_IMAGE_REF, COL_IMAGE_MOVE,
                 COL_POINTS_REF, COL_POINTS_MOVE)


# fixing ImportError: No module named 'copy_reg' for Python3
if sys.version_info.major == 2:
    import types
    import copy_reg

    def _reduce_method(m):
        # SOLVING issue: cPickle.PicklingError:
        #   Can't pickle <type 'instancemethod'>:
        #       attribute lookup __builtin__.instancemethod failed
        if m.im_self is None:
            tp = m.im_class
        else:
            tp = m.im_self
        return getattr, (tp, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _reduce_method)


class ImRegBenchmark(Experiment):
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
    >>> path_out = tl_io.create_dir('temp_results')
    >>> path_csv = os.path.join(tl_io.update_path('data_images'),
    ...                         'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_jobs': 1, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': path_csv}
    >>> benchmark = ImRegBenchmark(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)

    Running in 2 threads:
    >>> path_out = tl_io.create_dir('temp_results')
    >>> path_csv = os.path.join(tl_io.update_path('data_images'),
    ...                         'pairs-imgs-lnds_mix.csv')
    >>> params = {'nb_jobs': 2, 'unique': False,
    ...           'path_out': path_out,
    ...           'path_cover': path_csv}
    >>> benchmark = ImRegBenchmark(params)
    >>> benchmark.run()
    True
    >>> del benchmark
    >>> import shutil
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    REQUIRED_PARAMS = ['path_cover', 'path_out', 'nb_jobs']

    def __init__(self, params):
        """ initialise benchmark

        :param dict params:  {str: value}
        """
        assert 'unique' in params, 'missing "unique" among %s' \
                                   % repr(params.keys())
        super(ImRegBenchmark, self).__init__(params, params['unique'])
        logging.info(self.__doc__)

    def _check_required_params(self):
        """ check some extra required parameters for this benchmark """
        logging.debug('.. check if the BM have all required parameters')
        super(ImRegBenchmark, self)._check_required_params()
        for n in self.REQUIRED_PARAMS:
            assert n in self.params, 'missing "%s" among %s' \
                                     % (n, repr(self.params.keys()))

    def _update_path(self, path, tp='data'):
        """ update te path to the datset or output

        :param str path: original path
        :param str tp: type of update - data | output | general
        :return str: updated path
        """
        if tp == 'data' and 'path_dataset' in self.params:
            path = os.path.join(self.params['path_dataset'], path)
        elif tp == 'expt' and 'path_exp' in self.params:
            path = os.path.join(self.params['path_exp'], path)
        else:
            path = tl_io.update_path(path, absolute=True)
        path = os.path.abspath(path)
        return path

    def _get_paths(self, row):
        """ expand the relative paths to absolute

        :param row: row from cover ffile with relative pathes
        :return (str, str, str, str): path to reference and moving image
            and reference and moving landmarks
        """
        paths = [self._update_path(row[col]) for col in COVER_COLUMNS]
        return paths

    def _get_path_reg_dir(self, record):
        return self._update_path(str(record[COL_REG_DIR]), 'expt')

    def _load_data(self):
        """ loading data, the cover file with all registration pairs """
        logging.info('-> loading data...')
        # loading the csv cover file
        assert os.path.isfile(self.params['path_cover']), \
            'path to csv cover is not defined - %s' % self.params['path_cover']
        self._df_cover = pd.read_csv(self.params['path_cover'], index_col=None)
        assert all(col in self._df_cover.columns for col in COVER_COLUMNS), \
            'Some required columns are missing in the cover file.'

    def _run(self):
        """ perform complete benchmark experiment """
        logging.info('-> perform set of experiments...')

        # load existing result of create new entity
        self._path_csv_regist = os.path.join(self.params['path_exp'],
                                             NAME_CSV_REGISTRATION_PAIRS)
        if os.path.isfile(self._path_csv_regist):
            logging.info('loading existing csv: "%s"', self._path_csv_regist)
            self._df_experiments = pd.read_csv(self._path_csv_regist,
                                               index_col=None)
            if 'ID' in self._df_experiments.columns:
                self._df_experiments.set_index('ID', inplace=True)
        else:
            self._df_experiments = pd.DataFrame()

        # run the experiment in parallel of single thread
        self.__execute_method(self._perform_registration, self._df_cover,
                              self._path_csv_regist, 'registration experiments')

    def __execute_method(self, method, in_table, path_csv=None, name='',
                         use_output=True):
        """ execute a method in sequence or parallel

        :param func method: used method
        :param iter in_table: iterate over table
        :param str path_csv: path to the output temporal csv
        :param str name: name of the running process
        :param bool use_output: append output to experiment DF
        :return:
        """
        # run the experiment in parallel of single thread
        if self.params.get('nb_jobs', 0) > 1:
            self.nb_jobs = self.params['nb_jobs']
            self.__execute_parallel(method, in_table, path_csv,
                                    name, use_output=use_output)
        else:
            self.__execute_serial(method, in_table, path_csv, name,
                                  use_output=use_output)

    def __export_df_experiments(self, path_csv=None):
        """ export the DataFrame with registration results

        :param str | None path_csv: path to output CSV file
        """
        if path_csv is not None:
            if 'ID' in self._df_experiments.columns:
                self._df_experiments.set_index('ID').to_csv(path_csv)
            else:
                self._df_experiments.to_csv(path_csv, index=None)

    def __execute_parallel(self, method, in_table, path_csv=None, name='',
                           use_output=True):
        """ running several registration experiments in parallel

        :param method: executed self method which have as input row
        :param DataFrame in_table: table to be iterated by rows
        :param str path_csv: path to temporary saves
        :param str name: name of the process
        :param int nb_jobs: number of experiment running in parallel
        :param bool use_output: append output to experiment DF
        """
        if not hasattr(self, 'nb_jobs'):
            self.nb_jobs = NB_THREADS
        logging.info('-> running %s in parallel, (%i threads)',
                     name, self.nb_jobs)
        tqdm_bar = tqdm.tqdm(total=len(in_table),
                             desc='%s (@ %i threads)' % (name, self.nb_jobs))
        iter_table = ((idx, dict(row)) for idx, row, in in_table.iterrows())
        mproc_pool = mproc.Pool(self.nb_jobs)
        for res in mproc_pool.imap(method, iter_table):
            tqdm_bar.update()
            if res is None or not use_output:
                continue
            self._df_experiments = self._df_experiments.append(res, ignore_index=True)
            # if len(self._df_experiments) == 0:
            #     logging.warning('no particular expt. results')
            #     continue
            self.__export_df_experiments(path_csv)
        mproc_pool.close()
        mproc_pool.join()

    def __execute_serial(self, method, in_table, path_csv=None, name='',
                         use_output=True):
        """ running only one registration experiment in the time

        :param method: executed self method which have as input row
        :param DataFrame in_table: table to be iterated by rows
        :param str path_csv: path to temporary saves
        :param str name: name of the process
        :param bool use_output: append output to experiment DF
        """
        logging.info('-> running %s in single thread', name)
        tqdm_bar = tqdm.tqdm(total=len(in_table))
        iter_table = ((idx, dict(row)) for idx, row, in in_table.iterrows())
        for res in map(method, iter_table):
            tqdm_bar.update()
            if res is None or not use_output:
                continue
            self._df_experiments = self._df_experiments.append(res,
                                                               ignore_index=True)
            self.__export_df_experiments(path_csv)

    def __check_exist_regist(self, idx, path_dir_reg):
        """ check whether the particular experiment already exists and have results

        if the folder with experiment already exist and it is also part
        of the loaded finished experiments, sometimes the oder may mean
        failed experiment

        :param int idx: index of particular
        :param str path_dir_reg:
        :return bool:
        """
        b_df_col = ('ID' in self._df_experiments.columns and idx in self._df_experiments['ID'])
        b_df_idx = idx in self._df_experiments.index
        check = os.path.exists(path_dir_reg) and (b_df_col or b_df_idx)
        if check:
            logging.warning('particular registration experiment already exists:'
                            ' "%s"', repr(idx))
        return check

    def _perform_registration(self, df_row):
        """ run single registration experiment with all sub-stages

        :param (int, dict) df_row: tow from iterated table
        """
        idx, row = df_row
        logging.debug('-> perform single registration #%d...', idx)
        # create folder for this particular experiment
        row['ID'] = idx
        row[COL_REG_DIR] = str(idx)
        path_dir_reg = self._get_path_reg_dir(row)
        # check whether the particular experiment already exists and have result
        if self.__check_exist_regist(idx, path_dir_reg):
            return None
        tl_io.create_dir(path_dir_reg)

        row = self._prepare_registration(row)
        str_cmd = self._generate_regist_command(row)
        path_log = os.path.join(path_dir_reg, NAME_LOG_REGISTRATION)
        # TODO, add lock to single thread, create pool with possible thread ids
        # (USE taskset [native], numactl [need install])

        # measure execution time
        time_start = time.time()
        cmd_result = tl_expt.run_command_line(str_cmd, path_log)
        row[COL_TIME] = (time.time() - time_start) / 60.
        # if the experiment failed, return back None
        if not cmd_result:
            return None

        row = self._parse_regist_results(row)
        row = self._clear_after_registration(row)
        return row

    def _summarise(self):
        """ summarise complete benchmark experiment """
        logging.info('-> summarise experiment...')
        # load _df_experiments and compute stat
        self.__execute_serial(self.__compute_landmarks_statistic,
                              self._df_experiments, name='compute inaccuracy')
        # add visualisations
        if self.params.get('visual', False):
            self.__execute_method(self._visualise_registration,
                                  self._df_experiments, use_output=False,
                                  name='visualise results')
        # export stat to csv
        if self._df_experiments.empty:
            logging.warning('no experimental results were collected')
            return
        self.__export_df_experiments(self._path_csv_regist)
        # export simple stat to txt
        self.__export_summary_txt()

    def __compute_landmarks_statistic(self, df_row):
        """ after successful registration load initial nad estimated landmarks
        afterwords compute various statistic for init, and final alignment

        :param (int, dict) df_row: tow from iterated table
        """
        idx, row = df_row
        path_img_ref, _, path_lnds_ref, path_lnds_move = self._get_paths(row)
        # load initial landmarks
        points_ref = tl_io.load_landmarks(path_lnds_ref)
        points_move = tl_io.load_landmarks(path_lnds_move)
        points_warped = []
        img_size = tl_io.load_image(path_img_ref).shape[:2]
        # compute statistic
        self.__compute_landmarks_inaccuracy(idx, points_ref, points_move,
                                            'init', img_size)
        # load transformed landmarks
        if COL_POINTS_REF_WARP in row:
            points_target = points_move
            path_landmarks = self._update_path(row[COL_POINTS_REF_WARP], 'expt')
        elif COL_POINTS_MOVE_WARP in row:
            points_target = points_ref
            path_landmarks = self._update_path(row[COL_POINTS_MOVE_WARP], 'expt')
        else:
            logging.error('Statistic: no output landmarks')
            points_target, path_landmarks = [], None
        # load landmarks
        if isinstance(path_landmarks, str):
            points_warped = tl_io.load_landmarks(path_landmarks)

        # compute statistic
        img_size = tl_io.load_image(path_img_ref).shape[:2]
        if list(points_target) and list(points_warped):
            self.__compute_landmarks_inaccuracy(idx, points_target,
                                                points_warped, 'final', img_size)

    def __compute_landmarks_inaccuracy(self, idx, points1, points2, state='',
                                       img_size=None):
        """ compute statistic on two points sets

        :param int idx: index of tha particular record
        :param points1: np.array<nb_points, dim>
        :param points2: np.array<nb_points, dim>
        :param str state: whether it was before of after registration
        :param (int, int) img_size: target image size
        """
        _, stat = tl_expt.compute_points_dist_statistic(points1, points2)
        img_diag = None
        if img_size is not None:
            img_diag = np.sqrt(np.sum(np.array(img_size) ** 2))
        # update particular idx
        for name in stat:
            col_name = 'TRE %s (%s)' % (name, state)
            self._df_experiments.at[idx, col_name] = stat[name]
            if img_diag is not None:
                self._df_experiments.at[idx, 'Image diagonal [px]'] = img_diag
                col_name = 'rTRE %s (%s)' % (name, state)
                self._df_experiments.at[idx, col_name] = stat[name] / img_diag

    def __visual_image_move_warp_lnds_move_warp(self, record, image_ref,
                                                points_ref, points_move):
        """ visualise the case with warped moving image and landmarks
        to the reference frame so they are simple to overlap

        :param {} record: row with the experiment
        :param ndarray image_ref:
        :param ndarray points_move:
        :param ndarray points_ref:
        :return obj | None:
        """
        if COL_IMAGE_MOVE_WARP not in record \
                or not isinstance(record[COL_IMAGE_MOVE_WARP], str):
            logging.warning('Missing registered image "%s"', COL_IMAGE_MOVE_WARP)
            return None
        if COL_POINTS_MOVE_WARP not in record \
                or not isinstance(record[COL_POINTS_MOVE_WARP], str):
            logging.warning('Missing registered landmarks "%s"', COL_POINTS_MOVE_WARP)
            return None
        path_image_warp = self._update_path(record[COL_IMAGE_MOVE_WARP], 'expt')
        image_warp = tl_io.load_image(path_image_warp)
        path_points_warp = self._update_path(record[COL_POINTS_MOVE_WARP], 'expt')
        points_warp = tl_io.load_landmarks(path_points_warp)
        # draw image with landmarks
        image = tl_visu.draw_image_points(image_warp, points_warp)
        tl_io.save_image(os.path.join(self._get_path_reg_dir(record),
                                      NAME_IMAGE_MOVE_WARP_POINTS), image)
        # visualise the landmarks move during registration
        fig = tl_visu.draw_images_warped_landmarks(image_ref, image_warp,
                                                   points_move, points_ref,
                                                   points_warp)
        return fig

    def __visual_image_ref_warp_lnds_move_warp(self, record, image_ref,
                                               points_ref, points_move):
        """ visualise the case with warped reference landmarks to the move frame

        :param {} record: row with the experiment
        :param ndarray image_ref: reference image
        :param ndarray points_ref: landmarks in reference frame
        :param ndarray points_move: landmarks in moving frame
        :return obj:
        """
        image_move = tl_io.load_image(self._update_path(record[COL_IMAGE_MOVE]))
        # image_warp = tl_io.load_image(row['Moving image, Transf.'])
        path_points_warp = self._update_path(record[COL_POINTS_REF_WARP], 'expt')
        points_warp = tl_io.load_landmarks(path_points_warp)
        # draw image with landmarks
        image = tl_visu.draw_image_points(image_move, points_warp)
        tl_io.save_image(os.path.join(self._get_path_reg_dir(record),
                                      NAME_IMAGE_REF_POINTS_WARP), image)
        # visualise the landmarks move during registration
        fig = tl_visu.draw_images_warped_landmarks(image_ref, image_move,
                                                   points_ref, points_move,
                                                   points_warp)
        return fig

    def _visualise_registration(self, df_row):
        """ visualise the registration results according what landmarks were
        estimated - in registration or moving frame

        :param (int, dict) df_row: tow from iterated table
        """
        _, row = df_row
        path_img_ref, _, path_lnds_ref, path_lnds_move = self._get_paths(row)
        image_ref = tl_io.load_image(path_img_ref)
        points_ref = tl_io.load_landmarks(path_lnds_ref)
        points_move = tl_io.load_landmarks(path_lnds_move)
        # visualise particular experiment by idx
        if COL_POINTS_MOVE_WARP in row:
            fig = self.__visual_image_move_warp_lnds_move_warp(
                row, image_ref, points_ref, points_move)
            if fig is None:
                return None
        elif COL_POINTS_REF_WARP in row:
            fig = self.__visual_image_ref_warp_lnds_move_warp(
                row, image_ref, points_ref, points_move)
        else:
            logging.error('Visualisation: no output image or landmarks')
            fig, _ = tl_visu.create_figure((1, 1))
        path_fig = os.path.join(self._get_path_reg_dir(row),
                                NAME_IMAGE_WARPED_VISUAL)
        tl_visu.export_figure(path_fig, fig)
        return path_fig

    def __export_summary_txt(self):
        """ export the summary as CSV and TXT """
        path_txt = os.path.join(self.params['path_exp'], NAME_TXT_RESULTS)
        costume_prec = np.arange(0., 1., 0.05)
        if self._df_experiments.empty:
            logging.error('No registration results found.')
            return
        if 'ID' in self._df_experiments.columns:
            self._df_experiments.set_index('ID', inplace=True)
        df_summary = self._df_experiments.describe(percentiles=costume_prec).T
        df_summary['median'] = self._df_experiments.median()
        df_summary.sort_index(inplace=True)
        with open(path_txt, 'w') as fp:
            fp.write(tl_expt.string_dict(self.params, 'CONFIGURATION:'))
            fp.write('\n' * 3 + 'RESULTS:\n')
            fp.write('completed regist. experiments: %i' % len(self._df_experiments))
            fp.write('\n' * 2)
            fp.write(repr(df_summary[['mean', 'std', 'median', 'min', 'max']]))
            fp.write('\n' * 2)
            fp.write(repr(df_summary[['5%', '25%', '50%', '75%', '95%']]))
        path_csv = os.path.join(self.params['path_exp'], NAME_CSV_RESULTS)
        df_summary.to_csv(path_csv)

    @classmethod
    def _prepare_registration(self, record):
        """ prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict record: {str: value}, dictionary with regist. params
        :return dict: {str: value}
        """
        logging.debug('.. no preparing before regist. experiment')
        return record

    def _generate_regist_command(self, record):
        """ generate the registration command

        :param record: {str: value}, dictionary with regist. params
        :return: str, the execution string
        """
        logging.debug('.. simulate registration: '
                      'copy the target image and landmarks, simulate ideal case')
        path_reg_dir = self._get_path_reg_dir(record)
        name_img = os.path.basename(record[COL_IMAGE_REF])
        name_lnds = os.path.basename(record[COL_POINTS_MOVE])
        cmd_img = 'cp %s %s' % (self._update_path(record[COL_IMAGE_REF]),
                                os.path.join(path_reg_dir, name_img))
        cmd_lnds = 'cp %s %s' % (self._update_path(record[COL_POINTS_MOVE]),
                                 os.path.join(path_reg_dir, name_lnds))
        command = [cmd_img, cmd_lnds]
        return command

    @classmethod
    def _extract_warped_images_landmarks(self, record):
        """ get registration results - warped registered images and landmarks

        :param record: {str: value}, dictionary with registration params
        :return (str, str, str, str): paths to img_ref_warp, img_move_warp,
                                                lnds_ref_warp, lnds_move_warp
        """
        # detect image
        path_img = os.path.join(record[COL_REG_DIR],
                                os.path.basename(record[COL_IMAGE_REF]))
        # detect landmarks
        path_lnd = os.path.join(record[COL_REG_DIR],
                                os.path.basename(record[COL_POINTS_MOVE]))
        return None, path_img, path_lnd, None

    def _parse_regist_results(self, record):
        """ evaluate rests of the experiment and identity the registered image
        and landmarks when the process finished

        :param record: {str: value}, dictionary with registration params
        :return: {str: value}
        """
        paths = self._extract_warped_images_landmarks(record)
        columns = (COL_IMAGE_REF_WARP, COL_IMAGE_MOVE_WARP,
                   COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP)

        for path, col in zip(paths, columns):
            # detect image and landmarks
            if path is not None and os.path.isfile(self._update_path(path, 'expt')):
                record[col] = path

        return record

    @classmethod
    def _clear_after_registration(self, record):
        """ clean unnecessarily files after the registration

        :param record: {str: value}, dictionary with registration params
        :return: {str: value}
        """
        logging.debug('.. no cleaning after registration experiment')
        return record
