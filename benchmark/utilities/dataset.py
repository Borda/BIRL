"""
Some functionality related to dataset

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity

TISSUE_CONTENT = 0.05


def detect_binary_blocks(vec_bin):
    """ detect the binary object by beginning, end and length in !d signal

    :param [bool] vec_bin: binary vector with 1 for an object
    :return [int], [int], [int]:

    >>> vec = np.array([1] * 15 + [0] * 5 + [1] * 20)
    >>> detect_binary_blocks(vec)
    ([0, 20], [15, 39], [14, 19])
    """
    begins, ends, lengths = [], [], []

    # in case that it starts with an object
    if vec_bin[0]:
        begins.append(0)

    length = 0
    # iterate over whole array, skip the first one
    for i in range(1, len(vec_bin)):
        if vec_bin[i] > vec_bin[i - 1]:
            begins.append(i)
        elif vec_bin[i] < vec_bin[i - 1]:
            ends.append(i)
            lengths.append(length)
            length = 0
        elif vec_bin[i] == 1:
            length += 1

    # in case that it ends with an object
    if vec_bin[-1]:
        ends.append(len(vec_bin) - 1)
        lengths.append(length)

    return begins, ends, lengths


def find_split_objects(hist, nb_objects=2, threshold=TISSUE_CONTENT):
    """ find the N largest objects and set split as middle distance among them

    :param [float] hist: input vector
    :param int nb_objects: number of desired objects
    :param float threshold: threshold for input vector
    :return [int]:

    >>> vec = np.array([1] * 15 + [0] * 5 + [1] * 20)
    >>> find_split_objects(vec)
    [17]
    """
    hist_bin = hist > threshold
    begins, ends, lengths = detect_binary_blocks(hist_bin)

    if len(lengths) < nb_objects:
        logging.error('not enough objects')
        return []

    # select only the number of largest objects
    obj_sorted = sorted(zip(lengths, range(len(lengths))), reverse=True)
    obj_select = sorted([o[1] for o in obj_sorted][:nb_objects])

    # compute the mean in the gup
    splits = [np.mean((ends[obj_select[i]], begins[obj_select[i + 1]]))
              for i in range(len(obj_select) - 1)]
    splits = list(map(int, splits))

    return splits


def find_largest_object(hist, threshold=0.05):
    """ find the largest objects and give its beginning end end

    :param [float] hist: input vector
    :param float threshold: threshold for input vector
    :return [int]:

    >>> vec = np.array([1] * 15 + [0] * 5 + [1] * 20)
    >>> find_largest_object(vec)
    (20, 39)
    """
    hist_bin = hist > threshold
    begins, ends, lengths = detect_binary_blocks(hist_bin)

    assert len(lengths) > 0, 'no object found'

    # select only the number of largest objects
    obj_sorted = sorted(zip(lengths, range(len(lengths))), reverse=True)
    obj_select = obj_sorted[0][1]

    return begins[obj_select], ends[obj_select]


def project_object_edge(img, cut_dimension):
    """ scale the image, binarise with Otsu and project to one dimension

    :param ndarray img:
    :param int cut_dimension:
    :return [float]:

    >>> img = np.zeros((20, 10, 3))
    >>> img[2:6, 1:7, :] = 1
    >>> img[10:17, 4:6, :] = 1
    >>> project_object_edge(img, 0).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    """
    assert img.ndim == 3, 'unsupported image shape %s' % repr(img.shape)
    img_gray = np.mean(img, axis=-1)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    p_low, p_high = np.percentile(img_gray, (1, 95))
    img_gray = rescale_intensity(img_gray, in_range=(p_low, p_high))
    img_bin = img_gray > threshold_otsu(img_gray)
    img_edge = np.mean(img_bin, axis=1 - cut_dimension)
    return img_edge


def load_large_image(img_path):
    """ loading very large images

    Note, for the loading we have to use matplotlib while ImageMagic nor other
     lib (opencv, skimage, Pillow) is able to load larger images then 32k.

    :param str img_path:
    :return ndarray:
    """
    img = plt.imread(img_path)
    return img


def save_large_image(img_path, img):
    """ saving large images more then 50k x 50k

    :param str img_path:
    :param ndarray img:
    """
    if np.max(img) <= 1.:
        img = (img * 255)
    if np.max(img) > 255:
        img = (img / 255)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    plt.imsave(img_path, img)
