"""
Image registration using Elastix package.

Installing (building) the package:

* https://simpleelastix.readthedocs.io
* https://github.com/rcasero/pysto/wiki/Build-and-install-SimpleElastix-for-python-3.x

    git clone https://github.com/SuperElastix/SimpleElastix
    cd SimpleElastix && mkdir build && cd build
    cmake -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 \
        -DPYTHON_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libpython3.6.so \
        -DPYTHON_INCLUDE_DIR:PATH=/usr/include/python3.6 \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        ../SuperBuild
    make -j4
    cd SimpleITK-build/Wrapping/Python
    python Packaging/setup.py install

.. seealso:: https://simpleelastix.readthedocs.io/GettingStarted.html

Sample call of this script::

    python scripts/Python/run_SimpleElastix.py \
        ./data-images/images/artificial_reference.jpg \
        ./data-images/images/artificial_moving-affine.jpg \
        ./data-images/landmarks/artificial_reference.pts \
        ./results/elastix

.. seealso:: https://simpleelastix.readthedocs.io/PointBasedRegistration.html?highlight=warp#transforming-point-sets

"""

import os
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image


def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))


if len(sys.argv) < 5:
    print("Usage: {0} <fixedImageFilter> <movingImageFile> <fixedLandmarksFile> <outputFolder>"
          .format(sys.argv[0]))
    sys.exit(1)

NAME_TRANSFORM = 'transformation.txt'
path_fixed = sys.argv[1]
path_moving = sys.argv[2]
path_lnds = sys.argv[3]
path_out = sys.argv[4]

img_fixed = sitk.ReadImage(path_fixed, sitk.sitkFloat32)
# img_fixed = sitk.DiscreteGaussian(img_fixed, 2.0)
img_moving = sitk.ReadImage(path_moving, sitk.sitkFloat32)
# img_moving = sitk.DiscreteGaussian(img_moving, 2.0)
path_image = os.path.join(path_out, os.path.basename(path_moving))
path_landmarks = os.path.join(path_out, 'warped-fixed-landmarks.pts')
path_transf = os.path.join(path_out, NAME_TRANSFORM)

elastix = sitk.ElastixImageFilter()
elastix.SetFixedImage(img_fixed)
elastix.SetMovingImage(img_moving)

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastix.SetParameterMap(parameterMapVector)

elastix.Execute()
img_warped = elastix.GetResultImage()

Image.fromarray(sitk.GetArrayViewFromImage(img_warped).astype(np.uint8)).save(path_image)

# warp landmarks
transformix = sitk.TransformixImageFilter()
transformix.SetTransformParameterMap(elastix.GetTransformParameterMap())
transformix.SetFixedPointSetFileName(path_lnds)
# The transformed points will be written to a file named outputpoints.txt
transformix.SetOutputDirectory(path_out)
transformix.Execute()
