import os
import subprocess

import matplotlib
import numpy as np
import pandas as pd


CMD_TRY_MATPLOTLIB = 'python -c "from matplotlib import pyplot; pyplot.close(pyplot.figure())"'
# in case you are running on machine without display, e.g. server
if not os.environ.get('DISPLAY', '') and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
# _tkinter.TclError: couldn't connect to display "localhost:10.0"
elif subprocess.call(CMD_TRY_MATPLOTLIB, stdout=None, stderr=None, shell=True):
    print('Problem with display. Using non-interactive Agg backend')
    matplotlib.use('Agg')

# parse the numpy versions
np_version = [int(i) for i in np.version.full_version.split('.')]
# comparing strings does not work for version lower 1.10
if np_version >= [1, 14]:
    # np.set_printoptions(sign='legacy')
    np.set_printoptions(legacy='1.13')

# default display size was changed in pandas v0.23
if 'display.max_columns' in pd.core.config._registered_options:
    pd.set_option('display.max_columns', 20)
