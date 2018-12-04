"""
Zip scaled datasets
The paths and all other constants are set to run on CMP grid for ANHIR dataset

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import subprocess

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bm_dataset.generate_dataset_cover import (DATASET_TISSUE_SCALE, NAME_DIR_SCALE,
                                               DATASET_SCALES, GENERATED_SCALES)

ZIP_COMMAND = 'cd /datagrid/Medical/dataset_ANHIR/images/ ' \
              '&& zip -R dataset_%s.zip %s'


def create_command(scale):
    names = []
    shift = DATASET_SCALES[scale]
    for tissue in DATASET_TISSUE_SCALE:
        sc = DATASET_TISSUE_SCALE[tissue]
        sc = GENERATED_SCALES[min(GENERATED_SCALES.index(sc) + shift,
                                  len(GENERATED_SCALES))]
        name = os.path.join(tissue, NAME_DIR_SCALE % sc, '*')
        names.append(name)
    cmd = ZIP_COMMAND % (scale, ' '.join(names))
    print(cmd)
    return cmd


if __name__ == '__main__':
    # zip all datasets scales
    for scale in DATASET_SCALES:
        cmd = create_command(scale)
        subprocess.call(cmd, shell=True)
