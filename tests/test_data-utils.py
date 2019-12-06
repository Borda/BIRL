"""
Testing default benchmarks in single thred and parallel configuration
Check whether it generates correct outputs and resulting values

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import unittest

from parameterized import parameterized

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.utilities.data_io import update_path, load_image
from birl.utilities.dataset import image_histogram_matching, CONVERT_RGB

PATH_ROOT = os.path.dirname(update_path('birl'))
PATH_DATA = update_path('data_images')
PATH_IMAGE_REF = os.path.join(PATH_DATA, 'rat-kidney_', 'scale-5pc', 'Rat-Kidney_HE.jpg')
PATH_IMAGE_SRC = os.path.join(PATH_DATA, 'rat-kidney_', 'scale-5pc', 'Rat-Kidney_PanCytokeratin.jpg')


class TestHistogramMatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_ref = load_image(PATH_IMAGE_REF)
        cls.img_src = load_image(PATH_IMAGE_SRC)

    @parameterized.expand(list(CONVERT_RGB.keys()))
    def test_hist_matching(self, clr_space):
        """ test run in parallel with failing experiment """
        img = image_histogram_matching(self.img_src, self.img_ref, use_color=clr_space)
        self.assertAlmostEqual(self.img_src.shape, img.shape)
