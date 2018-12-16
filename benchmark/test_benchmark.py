"""
Testing default benchmarks in single thred and parallel configuration
Check whether it generates correct outputs and resulting values

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

import os
import sys
import logging
import unittest
import shutil

import numpy as np
import pandas as pd
from numpy.testing import assert_raises, assert_array_almost_equal

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from benchmark.utilities.data_io import update_path
from benchmark.cls_benchmark import ImRegBenchmark
from benchmark.cls_benchmark import (NAME_CSV_RESULTS, NAME_TXT_RESULTS,
                                     NAME_CSV_REGISTRATION_PAIRS, COVER_COLUMNS,
                                     COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP)
from benchmark.bm_template import BmTemplate

PATH_CSV_COVER = os.path.join(update_path('data_images'),
                              'pairs-imgs-lnds_mix.csv')
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
            'path_cover': PATH_CSV_COVER,
            'path_out': self.path_out,
            'nb_jobs': 2,
            'unique': False,
        }
        self.benchmark = ImRegBenchmark(params)
        self.benchmark.run()
        self.check_benchmark_results(final_means=[0., 0., 0., 0.],
                                     final_stds=[0., 0., 0., 0.])
        del self.benchmark

    def test_benchmark_simple(self):
        """ test run in single thread """
        params = {
            'path_cover': PATH_CSV_COVER,
            'path_out': self.path_out,
            'nb_jobs': 1,
            'unique': False,
            'an_executable': None,
        }
        self.benchmark = BmTemplate(params)
        self.benchmark.run()
        self.check_benchmark_results(final_means=[28.05, 68.21, 73.18, 76.44],
                                     final_stds=[12.77, 28.12, 28.26, 34.42])
        del self.benchmark

    def check_benchmark_results(self, final_means, final_stds):
        """ check whether the benchmark folder contains all required files
        and compute statistic correctly """
        path_bm = os.path.join(self.path_out, self.benchmark.__class__.__name__)
        assert os.path.exists(path_bm), 'missing benchmark: %s' % \
                                        self.benchmark.__class__.__name__
        # required output files
        for file_name in [NAME_CSV_REGISTRATION_PAIRS,
                          NAME_CSV_RESULTS,
                          NAME_TXT_RESULTS]:
            assert os.path.isfile(os.path.join(path_bm, file_name)), \
                'Missing "%s" file in the benchmark experiment' % file_name
        # load registration file
        path_csv = os.path.join(path_bm, NAME_CSV_REGISTRATION_PAIRS)
        df_regist = pd.read_csv(path_csv, index_col=0)
        # only two records in the benchmark
        assert len(df_regist) == len(self.benchmark._df_cover), \
            'Found only %i records instead of %i' % \
            (len(df_regist), len(self.benchmark._df_cover))
        # check existence of all mentioned files
        for _, row in df_regist.iterrows():
            for col in list(COVER_COLUMNS) + \
                    [COL_IMAGE_MOVE_WARP, COL_POINTS_MOVE_WARP]:
                assert os.path.exists(row[col]), \
                    'Missing column "%s" in result table' % col
        # check existence of statistical results
        for stat_name in ['Mean', 'STD', 'Median', 'Min', 'Max']:
            assert any(stat_name in col for col in df_regist.columns), \
                'Missing statistics "%s"' % stat_name
        # test specific results
        assert_array_almost_equal(sorted(df_regist['TRE Mean (final)'].values),
                                  np.array(final_means), decimal=2)
        assert_array_almost_equal(sorted(df_regist['TRE STD (final)'].values),
                                  np.array(final_stds), decimal=2)
