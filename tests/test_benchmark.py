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
from birl.utilities.data_io import update_path, save_config_yaml
from birl.utilities.dataset import args_expand_parse_images
from birl.utilities.experiments import parse_arg_params, try_decorator
from birl.benchmark import ImRegBenchmark
from birl.bm_template import BmTemplate

PATH_ROOT = os.path.dirname(update_path('birl'))
PATH_DATA = update_path('data_images')
PATH_CSV_COVER_MIX = os.path.join(PATH_DATA, 'pairs-imgs-lnds_mix.csv')
PATH_CSV_COVER_ANHIR = os.path.join(PATH_DATA, 'pairs-imgs-lnds_histol.csv')
# logging.basicConfig(level=logging.INFO)


class TestBmRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        cls.path_out = os.path.join(PATH_ROOT, 'output-testing')
        shutil.rmtree(cls.path_out, ignore_errors=True)
        os.mkdir(cls.path_out)

    def _remove_default_experiment(self, bm_name):
        path_expt = os.path.join(self.path_out, bm_name)
        shutil.rmtree(path_expt, ignore_errors=True)

    @classmethod
    def test_benchmark_invalid_inputs(self):
        # test missing some parameters
        params = {
            'path_table': 'x',
            'path_out': 'x',
            'nb_workers': 0,
            'unique': False,
        }
        # try a missing params
        for miss in ['path_table', 'path_out', 'unique']:
            params_miss = params.copy()
            del params_miss[miss]
            assert_raises(AssertionError, ImRegBenchmark, params_miss)
        # not defined output folder
        assert_raises(Exception, ImRegBenchmark, params)

    def test_benchmark_failing(self):
        """ test run in parallel with failing experiment """
        params = {
            'path_table': PATH_CSV_COVER_MIX,
            'path_dataset': PATH_DATA,
            'path_out': self.path_out,
            'preprocessing': 'nothing',
            'nb_workers': 4,
            'visual': True,
            'unique': True,
        }
        benchmark = ImRegBenchmark(params)
        benchmark.run()
        # no landmarks was copy and also no experiment results was produced
        list_csv = [len([csv for csv in files if os.path.splitext(csv)[-1] == '.csv'])
                    for _, _, files in os.walk(benchmark.params['path_exp'])]
        self.assertEqual(sum(list_csv), 0)
        del benchmark

    def test_benchmark_parallel(self):
        """ test run in parallel (2 threads) """
        self._remove_default_experiment(ImRegBenchmark.__name__)
        params = {
            'path_table': PATH_CSV_COVER_MIX,
            'path_out': self.path_out,
            'preprocessing': ['gray', 'matching-rgb'],
            'nb_workers': 2,
            'visual': True,
            'unique': False,
        }
        benchmark = ImRegBenchmark(params)
        # run it for the first time, complete experiment
        benchmark.run()
        # rerun experiment simulated repeating unfinished benchmarks
        benchmark.run()
        self.check_benchmark_results(benchmark,
                                     final_means=[0., 0., 0., 0., 0.],
                                     final_stds=[0., 0., 0., 0., 0.])
        del benchmark

    def test_benchmark_simple(self):
        """ test run in sequence (1 thread) """
        self._remove_default_experiment(ImRegBenchmark.__name__)
        params = {
            'path_table': PATH_CSV_COVER_ANHIR,
            'path_dataset': PATH_DATA,
            'path_out': self.path_out,
            'preprocessing': ['matching-hsv', 'gray'],
            'nb_workers': 1,
            'visual': True,
            'unique': False,
        }
        benchmark = ImRegBenchmark(params)
        benchmark.run()
        self.check_benchmark_results(benchmark,
                                     final_means=[0., 0.],
                                     final_stds=[0., 0.])
        del benchmark

    def test_benchmark_template(self):
        """ test run in single thread """
        path_config = os.path.join(self.path_out, 'sample_config.yaml')
        save_config_yaml(path_config, {})
        params = {
            'path_table': PATH_CSV_COVER_MIX,
            'path_out': self.path_out,
            'path_config': path_config,
            'nb_workers': 2,
            'unique': False,
            'visual': True,
        }
        benchmark = BmTemplate(params)
        benchmark.run()
        self.check_benchmark_results(benchmark,
                                     final_means=[28., 68., 73., 76., 95.],
                                     final_stds=[1., 13., 28., 28., 34.])
        os.remove(path_config)
        del benchmark

    def check_benchmark_results(self, benchmark, final_means, final_stds):
        """ check whether the benchmark folder contains all required files
        and compute statistic correctly """
        bm_name = benchmark.__class__.__name__
        path_bm = os.path.join(self.path_out, bm_name)
        self.assertTrue(os.path.exists(path_bm), msg='Missing benchmark: %s' % bm_name)
        # required output files
        for file_name in [benchmark.NAME_CSV_REGISTRATION_PAIRS,
                          benchmark.NAME_RESULTS_CSV,
                          benchmark.NAME_RESULTS_TXT]:
            self.assertTrue(os.path.isfile(os.path.join(path_bm, file_name)),
                            msg='Missing "%s" file in the BM experiment' % file_name)

        # load registration file
        path_csv = os.path.join(path_bm, benchmark.NAME_CSV_REGISTRATION_PAIRS)
        df_regist = pd.read_csv(path_csv, index_col=0)

        # only two items in the benchmark
        self.assertEqual(len(df_regist), len(benchmark._df_overview),
                         msg='Found only %i records instead of %i'
                             % (len(df_regist), len(benchmark._df_overview)))

        # test presence of particular columns
        for col in list(benchmark.COVER_COLUMNS) + [benchmark.COL_IMAGE_MOVE_WARP]:
            self.assertIn(col, df_regist.columns,
                          msg='Missing column "%s" in result table' % col)
        cols_lnds_warp = [col in df_regist.columns
                          for col in [benchmark.COL_POINTS_REF_WARP, benchmark.COL_POINTS_MOVE_WARP]]
        self.assertTrue(any(cols_lnds_warp), msg='Missing any column of warped landmarks')
        col_lnds_warp = benchmark.COL_POINTS_REF_WARP if cols_lnds_warp[0] \
            else benchmark.COL_POINTS_MOVE_WARP
        # check existence of all mentioned files
        for _, row in df_regist.iterrows():
            self.assertTrue(os.path.isfile(os.path.join(path_bm, row[benchmark.COL_IMAGE_MOVE_WARP])),
                            msg='Missing image "%s"' % row[benchmark.COL_IMAGE_MOVE_WARP])
            self.assertTrue(os.path.isfile(os.path.join(path_bm, row[col_lnds_warp])),
                            msg='Missing landmarks "%s"' % row[col_lnds_warp])

        # check existence of statistical results
        for stat_name in ['Mean', 'STD', 'Median', 'Min', 'Max']:
            self.assertTrue(any(stat_name in col for col in df_regist.columns),
                            msg='Missing statistics "%s"' % stat_name)

        # test specific results
        assert_array_almost_equal(sorted(df_regist['TRE Mean'].values),
                                  np.array(final_means), decimal=0)
        assert_array_almost_equal(sorted(df_regist['TRE STD'].values),
                                  np.array(final_stds), decimal=0)

    def test_try_wrap(self):
        self.assertIsNone(try_wrap())

    def test_argparse(self):
        with patch('argparse._sys.argv', ['script.py']):
            args = parse_arg_params(argparse.ArgumentParser())
            self.assertIsInstance(args, dict)

    def test_argparse_images(self):
        with patch('argparse._sys.argv', ['script.py', '-i', 'an_image.png']):
            args = args_expand_parse_images(argparse.ArgumentParser())
            self.assertIsInstance(args, dict)

    def test_fail_visual(self):
        fig = ImRegBenchmark._visual_image_move_warp_lnds_move_warp(
            {ImRegBenchmark.COL_POINTS_MOVE_WARP: 'abc'})
        self.assertIsNone(fig)
        fig = ImRegBenchmark._visual_image_move_warp_lnds_ref_warp(
            {ImRegBenchmark.COL_POINTS_REF_WARP: 'abc'})
        self.assertIsNone(fig)
        fig = ImRegBenchmark.visualise_registration((0, {}))
        self.assertIsNone(fig)


@try_decorator
def try_wrap():
    return '%i' % '42'
