'''
Execute Image registration with ANTsPy
https://github.com/ANTsX/ANTsPy/blob/master/tutorials/10minTutorial.ipynb

>> python ./scripts/Python/run_ANTsPy.py \
    ./data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.jpg \
    ./data_images/rat-kidney_/scale-5pc/Rat_Kidney_PanCytokeratin.jpg \
    ./data_images/rat-kidney_/scale-5pc/Rat_Kidney_PanCytokeratin.csv \
    ./results

Copyright (C) 2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
'''

import os
import sys
import time

import ants
import pandas as pd
from skimage.io import imread, imsave
from skimage.color import rgb2gray

# setting paths
paths = {
    'fixed': os.path.abspath(os.path.expanduser(sys.argv[1])),
    'moving': os.path.abspath(os.path.expanduser(sys.argv[2])),
    'lnds': os.path.abspath(os.path.expanduser(sys.argv[3])),
    'out': os.path.abspath(os.path.expanduser(sys.argv[4]))
}
paths['warped'] = os.path.join(paths['out'], 'warped-image.jpg')
paths['pts'] = os.path.join(paths['out'], 'warped-landmarks.csv')
paths['time'] = os.path.join(paths['out'], 'time.txt')

t_start = time.time()
# loading images and landmarks
fixed = ants.from_numpy(rgb2gray(imread(paths['fixed'])))
moving = ants.from_numpy(rgb2gray(imread(paths['moving'])))
lnds = pd.read_csv(paths['lnds'])[['X', 'Y']]
# transform landmarks coordinates
lnds.columns = ['y', 'x']

# perform image registration
mytx = ants.registration(fixed=fixed,
                         moving=moving,
                         type_of_transform='ElasticSyN')
print('Transform: %r' % mytx)
t_elapsed = time.time() - t_start
print('Time: %r seconds' % t_elapsed)
warped_moving = ants.apply_transforms(fixed=fixed,
                                      moving=moving,
                                      transformlist=mytx['fwdtransforms'])
warped_points = ants.apply_transforms_to_points(dim=2,
                                                points=lnds,
                                                transformlist=mytx['invtransforms'])

# # Visualisation
# import matplotlib.pyplot as plt
# plt.imshow(warped_moving.numpy(), cmap=plt.cm.Greys_r)
# plt.plot(warped_points['y'], warped_points['x'], '.')
# plt.show()

# Exporting results
with open(paths['time'], 'w') as fp:
    fp.write(str(t_elapsed))
imsave(paths['warped'], warped_moving.numpy())
lnds = warped_points[['y', 'x']]
# transform landmarks coordinates back
lnds.columns = ['X', 'Y']
lnds.to_csv(paths['pts'])
print('finished')
