"""
Image registration using ITK package.

Sample call of this script::

    python scripts/Python/run_SimpleITK.py \
        ./data-images/images/artificial_reference.jpg \
        ./data-images/images/artificial_moving-affine.jpg \
        ./data-images/landmarks/artificial_moving-affine.pts \
        ./results/elastix

.. seealso:: https://simpleitk.readthedocs.io/en/master/Examples/index.html

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
NAME_LANDMARKS = 'warped-landmarks.pts'
path_fixed = sys.argv[1]
path_moving = sys.argv[2]
path_lnds = sys.argv[3]
path_out = sys.argv[4]

img_fixed = sitk.ReadImage(path_fixed, sitk.sitkFloat32)
# img_fixed = sitk.DiscreteGaussian(img_fixed, 2.0)
img_moving = sitk.ReadImage(path_moving, sitk.sitkFloat32)
# img_moving = sitk.DiscreteGaussian(img_moving, 2.0)
path_image = os.path.join(path_out, os.path.basename(path_moving))
path_landmarks = os.path.join(path_out, os.path.basename(path_lnds))
path_transf = os.path.join(path_out, NAME_TRANSFORM)

number_bins = 24
sampling_percentage = 0.10

reg = sitk.ImageRegistrationMethod()
reg.SetMetricAsJointHistogramMutualInformation()
reg.SetOptimizerAsGradientDescentLineSearch(learningRate=0.01,
                                            numberOfIterations=200,
                                            convergenceMinimumValue=1e-5,
                                            convergenceWindowSize=25)
reg.SetInitialTransform(sitk.AffineTransform(img_fixed.GetDimension()))
reg.SetInterpolator(sitk.sitkLinear)

reg.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(reg))

out_transform = reg.Execute(img_fixed, img_moving)
sitk.WriteTransform(out_transform, path_transf)

# Warp image
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(img_fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(out_transform)

img_warped = resampler.Execute(img_moving)
Image.fromarray(sitk.GetArrayViewFromImage(img_warped).astype(np.uint8)).save(path_image)

# Load points
with open(path_lnds, 'r') as fp:
    points = fp.readlines()[2:]

# Transform and warp landmarks
with open(os.path.join(path_out, NAME_LANDMARKS), 'w') as fp:
    points_w = ['point', str(len(points))]
    for pts in points:
        pts = list(map(float, pts.split(' ')))
        pts_w = out_transform.TransformPoint(pts)
        points_w.append('%.3f %.3f' % pts_w)
    fp.writelines('\n'.join(points_w))
