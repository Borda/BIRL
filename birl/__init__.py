import birl.utilities

birl.utilities

__version__ = '0.2.2'
__author__ = 'Jiri Borovec'
__author_email__ = 'jiri.borovec@fel.cvut.cz'
__license__ = 'BSD 3-clause'
__home__ = 'https://borda.github.io/BIRL',
__copyright__ = 'Copyright (c) 2014-2019, Jiri Borovec.'
__doc__ = """
# BIRL: Benchmark on Image Registration methods with Landmark validation

The project aims at automatic evaluation of state-of-the-art image registration
 methods based on landmark annotation for given image dataset. In particular,
 this project is the main evaluation framework for ANHIR challenge.

## Main Features
* **automatic** execution of image registration on a sequence of image pairs
* integrated **evaluation** of registration performances
* integrated **visualization** of performed registration
* running several image registration experiment in **parallel**
* **resuming** unfinished sequence of registration benchmark
* handling around dataset and **creating own experiments**
* rerun evaluation and visualisation for finished experiments

## References
Borovec, J., Munoz-Barrutia, A., & Kybic, J. (2018). Benchmarking of image
 registration methods for differently stained histological slides.
 In IEEE International Conference on Image Processing (ICIP) (pp. 3368-3372),
 Athens. DOI: 10.1109/ICIP.2018.8451040
"""
