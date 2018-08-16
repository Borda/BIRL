"""
Testing default benchmarks in single thred and parallel configuration
Check whether it generates correct outputs and resulting values

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import unittest
import shutil

import numpy as np
import pandas as pd
from numpy.testing import assert_raises, assert_array_almost_equal

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.data_io import update_path
import benchmark.cls_benchmark as bm

PATH_CSV_COVER = os.path.join(update_path('data_images'),
                              'list_pairs_imgs_lnds.csv')


class TestBmRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path_base = os.path.dirname(update_path('requirements.txt'))
        cls.path_out = os.path.join(path_base, 'output')
        if not os.path.exists(cls.path_out):
            os.mkdir(cls.path_out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path_out, ignore_errors=True)

    @classmethod
    def setUp(self):
        # remove previous benchmark folder
        shutil.rmtree(os.path.join(self.path_out, 'ImRegBenchmark'),
                      ignore_errors=True)

    @classmethod
    def test_benchmark_invalid_inputs(self):
        # test missing some parameters
        params = {'path_cover': 'x', 'path_out': 'x',
                  'nb_jobs': 0, 'unique': False}
        # try a missing params
        for miss in ['path_cover', 'path_out', 'nb_jobs', 'unique']:
            params_miss = params.copy()
            del params_miss[miss]
            assert_raises(AssertionError, bm.ImRegBenchmark, params_miss)
        # not defined output folder
        assert_raises(Exception, bm.ImRegBenchmark, params)

    @classmethod
    def test_benchmark_parallel(self):
        """ test run in parallel (2 threads) """
        params = {
            'path_cover': PATH_CSV_COVER,
            'path_out': self.path_out,
            'nb_jobs': 2,
            'unique': False,
        }
        self.benchmark = bm.ImRegBenchmark(params)
        self.benchmark.run()
        self.check_benchmark_results()
        del self.benchmark

    @classmethod
    def test_benchmark_simple(self):
        """ test run in single thread """
        params = {
            'path_cover': PATH_CSV_COVER,
            'path_out': self.path_out,
            'nb_jobs': 1,
            'unique': False,
        }
        self.benchmark = bm.ImRegBenchmark(params)
        self.benchmark.run()
        self.check_benchmark_results()
        del self.benchmark

    @classmethod
    def check_benchmark_results(self):
        """ check whether the benchmark folder contains all required files
        and compute statistic correctly """
        path_bm = os.path.join(self.path_out, self.benchmark.__class__.__name__)
        assert os.path.exists(path_bm), 'missing benchmark: %s' % \
                                        self.benchmark.__class__.__name__
        # required output files
        for file_name in [bm.NAME_CSV_REGIST,
                          bm.NAME_CSV_RESULTS,
                          bm.NAME_TXT_RESULTS]:
            assert os.path.exists(os.path.join(path_bm, file_name)), \
                'missing "%s" file in the benchmark experiment' % file_name
        # load registration file
        df_regist = pd.read_csv(os.path.join(path_bm, bm.NAME_CSV_REGIST), index_col=0)
        # only two records in the benchmark
        assert len(df_regist) == len(self.benchmark._df_cover), \
            'found only %i records instead of %i' % \
            (len(df_regist), len(self.benchmark._df_cover))
        # check existence of all mentioned files
        for _, row in df_regist.iterrows():
            for col in bm.COVER_COLUMNS + \
                    [bm.COL_IMAGE_REF_WARP, bm.COL_POINTS_REF_WARP]:
                assert os.path.exists(row[col]), \
                    'missing column "%s" in result table' % col
        # check existence of statistical results
        for stat_name in ['Mean', 'STD', 'Median', 'Min', 'Max']:
            assert any(stat_name in col for col in df_regist.columns), \
                'missing statistis "%s"' % stat_name
        # test specific results
        assert_array_almost_equal(sorted(df_regist['Mean [px] (init)'].values),
                                  np.array([28.055, 68.209, 73.175, 76.439]),
                                  decimal=3)
        assert_array_almost_equal(sorted(df_regist['STD [px] (init)'].values),
                                  np.array([12.765, 28.12, 28.255, 34.417]),
                                  decimal=3)
