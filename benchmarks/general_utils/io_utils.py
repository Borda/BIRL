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

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_landmarks('./sample_landmarks.csv', points)
    >>> points1 = load_landmarks('./sample_landmarks.csv')
    >>> points2 = load_landmarks('./sample_landmarks.txt')
    >>> np.array_equal(points1, points2)
    True
    >>> os.remove('./sample_landmarks.csv')
    >>> os.remove('./sample_landmarks.txt')
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
    :return: np.array<height, width, ch>

    >>> img = np.random.random((50, 50, 3))
    >>> save_image('./test_image.jpg', img)
    >>> img2 = load_image('./test_image.jpg')
    >>> img2.max() <= 1.
    True
    >>> os.remove('./test_image.jpg')
    """
    assert os.path.exists(path_image), 'missing image "%s"' % (path_image)
    image = np.array(Image.open(path_image))
    while image.max() > 1.:
        image = (image / 255.)
    return image


def convert_ndarray_2_image(image):
    """ convert ndarray to PIL image if it not already

    :param image: np.ndarray
    :return: Image

    >>> img = np.random.random((50, 50, 3))
    >>> image = convert_ndarray_2_image(img)
    >>> isinstance(image, Image.Image)
    True
    """
    if isinstance(image, np.ndarray):
        if image.max() <= 1.:
            image = (image * 255)
        image = Image.fromarray(np.round(image).astype(np.uint8))
    return image


def save_image(path_image, image):
    """ save the image into given path

    :param str path_image: path to the image
    :param image: np.array<height, width, ch>
    """
    image = convert_ndarray_2_image(image)
    path_dir = os.path.dirname(path_image)
    if os.path.exists(path_dir):
        image.save(path_image)
    else:
        logging.error('upper folder does not exists: "%s"', path_dir)


def load_parse_bunwarpj_displacement_axis(fp, size, points):
    """ given pointer in the file aiming to the beginning of displacement
     parse all lines and if in the particular line is a point from list
     get its new position

    :param fp: file pointer
    :param (int, int) size: width, height of the image
    :param points: np.array<nb_points, 2>
    :return list: list of new positions on given axis (x/y) for related points
    """
    width, height = size
    points = np.round(points)
    selected_lines = points[:, 1].tolist()
    pos_new = [0] * len(points)

    # walk thor all lined of this displacement field
    for i in range(height):
        line = fp.readline()
        # if the any point is listed in this line
        if i in selected_lines :
            pos = line.rstrip().split()
            # pos = [float(e) for e in pos if len(e)>0]
            assert len(pos) == width
            # find all points in this line
            for j, point in enumerate(points):
                if point[1] == i:
                    pos_new[j] = float(pos[point[0]])
    return pos_new


def load_parse_bunwarpj_displacements_warp_points(path_file, points):
    """ load and parse displacement field for both X and Y coordinated
    and return new position of selected points

    :param str path_file:
    :param points: np.array<nb_points, 2>
    :return: np.array<nb_points, 2>

    >>> fp = open('./my_transform.txt', 'w')
    >>> fp.write('''Width=5
    ... Height=4
    ...
    ... X Trans -----------------------------------
    ... 11 12 13 14 15
    ... 11 12 13 14 15
    ... 11 12 13 14 15
    ... 11 12 13 14 15
    ...
    ... Y Trans -----------------------------------
    ... 20 20 20 20 20
    ... 21 21 21 21 21
    ... 22 22 22 22 22
    ... 23 23 23 23 23''')  # doctest: +ELLIPSIS
    >>> fp.close()
    >>> points = np.array([[1, 1], [4, 0], [2, 3]])
    >>> load_parse_bunwarpj_displacements_warp_points('./my_transform.txt',
    ...                                               points)
    array([[ 12.,  21.],
           [ 15.,  20.],
           [ 13.,  23.]])
    >>> os.remove('./my_transform.txt')
    """
    assert os.path.exists(path_file)
    assert os.path.isfile(path_file)

    fp = open(path_file, 'r')
    # read image sizes
    width = int(fp.readline().split('=')[-1])
    height = int(fp.readline().split('=')[-1])
    logging.debug('loaded image size: %i x %i', width, height)
    size = (width, height)
    assert all(np.max(points, axis=0) <= size), \
        'some points are outside of the image'

    # read inter line
    fp.readline(), fp.readline()
    # read inter line and Transform notation
    points_x = load_parse_bunwarpj_displacement_axis(fp, size, points)

    # read inter line and Transform notation
    fp.readline(), fp.readline()
    # read Y Trans
    points_y = load_parse_bunwarpj_displacement_axis (fp, size, points)
    fp.close()

    points_new = np.vstack((points_x, points_y)).T
    return points_new
