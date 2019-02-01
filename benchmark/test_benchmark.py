"""
Testing default benchmarks in single thred and parallel configuration
Check whether it generates correct outputs and resulting values

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import unittest
import shutil
import argparse
try:  # python 3
    from unittest.mock import patch
except ImportError:  # python 2
    from mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_raises, assert_array_almost_equal

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.data_io import update_path
from benchmark.utilities.experiments import (parse_arg_params, try_decorator)
from benchmark.cls_benchmark import ImRegBenchmark
from benchmark.cls_benchmark import (NAME_CSV_RESULTS, NAME_TXT_RESULTS,
                                     NAME_CSV_REGISTRATION_PAIRS, COVER_COLUMNS,
                                     COL_IMAGE_MOVE_WARP,
                                     COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP)
from benchmark.bm_template import BmTemplate

PATH_DATA = update_path('data_images')
PATH_CSV_COVER_MIX = os.path.join(PATH_DATA, 'pairs-imgs-lnds_mix.csv')
PATH_CSV_COVER_ANHIR = os.path.join(PATH_DATA, 'pairs-imgs-lnds_anhir.csv')
logging.basicConfig(level=logging.INFO)


class TestBmRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path_base = os.path.dirname(update_path('requirements.txt'))
        cls.path_out = os.path.join(path_base, 'output')
        shutil.rmtree(cls.path_out, ignore_errors=True)
        os.mkdir(cls.path_out)

    @classmethod
    def test_benchmark_invalid_inputs(self):
        # test missing some parameters
        params = {'path_cover': 'x', 'path_out': 'x',
                  'nb_jobs': 0, 'unique': False}
        # try a missing params
        for miss in ['path_cover', 'path_out', 'nb_jobs', 'unique']:
            params_miss = params.copy()
            del params_miss[miss]
            assert_raises(AssertionError, ImRegBenchmark, params_miss)
        # not defined output folder
        assert_raises(Exception, ImRegBenchmark, params)

    def test_benchmark_parallel(self):
        """ test run in parallel (2 threads) """
        params = {
            'path_cover': PATH_CSV_COVER_ANHIR,
            'path_dataset': PATH_DATA,
            'path_out': self.path_out,
            'nb_jobs': 2,
            'visual': True,
            'unique': False,
        }
        self.benchmark = ImRegBenchmark(params)
        self.benchmark.run()
        self.check_benchmark_results(final_means=[0., 0.],
                                     final_stds=[0., 0.])
        del self.benchmark

    def test_benchmark_simple(self):
        """ test run in parallel (2 threads) """
        params = {
            'path_cover': PATH_CSV_COVER_ANHIR,
            'path_dataset': PATH_DATA,
            'path_out': self.path_out,
            'nb_jobs': 1,
            'visual': True,
            'unique': False,
        }
        self.benchmark = ImRegBenchmark(params)
        self.benchmark.run()
        self.check_benchmark_results(final_means=[0., 0.],
                                     final_stds=[0., 0.])
        del self.benchmark

    def test_benchmark_template(self):
        """ test run in single thread """
        params = {
            'path_cover': PATH_CSV_COVER_MIX,
            'path_out': self.path_out,
            'nb_jobs': 2,
            'unique': False,
            'visual': True,
            'an_executable': None,
        }
        self.benchmark = BmTemplate(params)
        self.benchmark.run()
        self.check_benchmark_results(final_means=[28., 68., 73., 76.],
                                     final_stds=[13., 28., 28., 34.])
        del self.benchmark

    def check_benchmark_results(self, final_means, final_stds):
        """ check whether the benchmark folder contains all required files
        and compute statistic correctly """
        bm_name = self.benchmark.__class__.__name__
        path_bm = os.path.join(self.path_out, bm_name)
        self.assertTrue(os.path.exists(path_bm), msg='missing benchmark: %s' % bm_name)
        # required output files
        for file_name in [NAME_CSV_REGISTRATION_PAIRS,
                          NAME_CSV_RESULTS,
                          NAME_TXT_RESULTS]:
            self.assertTrue(os.path.isfile(os.path.join(path_bm, file_name)),
                            msg='Missing "%s" file in the BM experiment' % file_name)

        # load registration file
        path_csv = os.path.join(path_bm, NAME_CSV_REGISTRATION_PAIRS)
        df_regist = pd.read_csv(path_csv, index_col=0)

        # only two records in the benchmark
        self.assertEqual(len(df_regist), len(self.benchmark._df_cover),
                         msg='Found only %i records instead of %i'
                             % (len(df_regist), len(self.benchmark._df_cover)))

        # test presence of particular columns
        for col in list(COVER_COLUMNS) + [COL_IMAGE_MOVE_WARP]:
            self.assertIn(col, df_regist.columns,
                          msg='Missing column "%s" in result table' % col)
        cols_lnds_warp = [col in df_regist.columns
                          for col in [COL_POINTS_REF_WARP, COL_POINTS_MOVE_WARP]]
        self.assertTrue(any(cols_lnds_warp), msg='Missing any column of warped landmarks')
        col_lnds_warp = COL_POINTS_REF_WARP if cols_lnds_warp[0] else COL_POINTS_MOVE_WARP
        # check existence of all mentioned files
        for _, row in df_regist.iterrows():
            self.assertTrue(os.path.isfile(os.path.join(path_bm, row[COL_IMAGE_MOVE_WARP])),
                            msg='Missing image "%s"' % row[COL_IMAGE_MOVE_WARP])
            self.assertTrue(os.path.isfile(os.path.join(path_bm, row[col_lnds_warp])),
                            msg='Missing landmarks "%s"' % row[col_lnds_warp])

        # check existence of statistical results
        for stat_name in ['Mean', 'STD', 'Median', 'Min', 'Max']:
            self.assertTrue(any(stat_name in col for col in df_regist.columns),
                            msg='Missing statistics "%s"' % stat_name)

        # test specific results
        assert_array_almost_equal(sorted(df_regist['TRE Mean (final)'].values),
                                  np.array(final_means), decimal=0)
        assert_array_almost_equal(sorted(df_regist['TRE STD (final)'].values),
                                  np.array(final_stds), decimal=0)

    @try_decorator
    def test_try_wrap(self):
        print('%i' % '42')

    def test_argparse(self):
        with patch('argparse._sys.argv', ['script.py']):
            parse_arg_params(argparse.ArgumentParser())
