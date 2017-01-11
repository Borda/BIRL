"""
Useful function for managing Input/Output

Copyright (C) 2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging

import numpy as np
import pandas as pd
from PIL import Image

LANDMARK_COORDS = ['X', 'Y']


def create_dir(path_dir):
    """ create a folder if it not exists

    :param str path_dir:
    """
    if not os.path.exists(path_dir):
        os.makedirs(path_dir, mode=0o775)
    else:
        logging.warning('Folder already exists: %s', path_dir)


def load_landmarks(path_file):
    """ load landmarks in csv and txt format

    :param str path_file: path to the input file
    :return: np.array<np_points, dim>
    """
    assert os.path.exists(path_file), str(path_file)
    ext = os.path.splitext(path_file)[-1]
    if ext == '.csv':
        return load_landmarks_csv(path_file)
    elif ext == '.txt':
        return load_landmarks_txt(path_file)
    else:
        logging.error('not supported landmarks file: %s',
                      os.path.basename(path_file))


def load_landmarks_txt(path_file):
    """ load file with landmarks in txt format

    :param str path_file: path to the input file
    :return: np.array<np_points, dim>

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_landmarks_txt('./sample_landmarks.txt', points)
    >>> load_landmarks_txt('./sample_landmarks.txt')
    array([[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]])
    >>> os.remove('./sample_landmarks.txt')
    """
    assert os.path.exists(path_file)
    with open(path_file, 'r') as fp:
        lines = fp.readlines()
    nb_points = int(lines[1])
    points = [[float(n) for n in line.split()] for line in lines[2:]]
    assert nb_points == len(points)
    return np.array(points, dtype=np.float)


def load_landmarks_csv(path_file):
    """ load file with landmarks in cdv format

    :param str path_file: path to the input file
    :return: np.array<np_points, dim>

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_landmarks_csv('./sample_landmarks.csv', points)
    >>> load_landmarks_csv('./sample_landmarks.csv')
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> os.remove('./sample_landmarks.csv')
    """
    assert os.path.exists(path_file)
    df = pd.DataFrame.from_csv(path_file)
    points = df[LANDMARK_COORDS].values
    return points


def save_landmarks(path_file, landmarks):
    """ save landmarks into a specific file

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.exists(os.path.dirname(path_file))
    path_file = os.path.splitext(path_file)[0]
    save_landmarks_csv(path_file + '.csv', landmarks)
    save_landmarks_txt(path_file + '.txt', landmarks)


def save_landmarks_txt(path_file, landmarks):
    """ save landmarks into a txt file

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.exists(os.path.dirname(path_file))
    lines = ['point', str(len(landmarks))]
    lines += [' '.join(str(i) for i in point) for point in landmarks]
    with open(path_file, 'w') as fp:
        fp.write(os.linesep.join(lines))


def save_landmarks_csv(path_file, landmarks):
    """ save landmarks into a csv file

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.exists(os.path.dirname(path_file))
    assert os.path.splitext(path_file)[-1] == '.csv'
    df = pd.DataFrame(landmarks, columns=LANDMARK_COORDS)
    df.to_csv(path_file)


def try_find_upper_folders(path_file, depth=5):
    """ try to bubble up the until max depth of find such path

    :param str path_file: original path to the file
    :param int depth: max depth of up bubbling
    :return str: resulting path file
    """
    while not os.path.exists(path_file) and depth > 0:
        path_file = os.path.join('..', path_file)
        depth -= 1
    return path_file


def load_image(path_image):
    """ load the image in value range (0, 1)

    :param str path_image:
    :return: np.array<height, width, n>
    """
    assert os.path.exists(path_image)
    image = np.array(Image.open(path_image))
    if image.max() > 1.:
        image = (image / 255.)
    return image
