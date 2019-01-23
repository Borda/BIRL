"""
Some functionality related to dataset

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import re
import glob
import logging

from scipy import spatial
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity

TISSUE_CONTENT = 0.01
# supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
# https://github.com/opencv/opencv/issues/6729
# https://www.life2coding.com/save-opencv-images-jpeg-quality-png-compression
IMAGE_COMPRESSION_OPTIONS = (cv.IMWRITE_JPEG_QUALITY, 98) + \
                            (cv.IMWRITE_PNG_COMPRESSION, 9)
REEXP_FOLDER_SCALE = r'\S*scale-(\d+)pc'
# ERROR:root:error: Image size (... pixels) exceeds limit of ... pixels,
# could be decompression bomb DOS attack.
# SEE: https://gitlab.mister-muffin.de/josch/img2pdf/issues/42
Image.MAX_IMAGE_PIXELS = None


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


def find_largest_object(hist, threshold=TISSUE_CONTENT):
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

    assert lengths, 'no object found'

    # select only the number of largest objects
    obj_sorted = sorted(zip(lengths, range(len(lengths))), reverse=True)
    obj_select = obj_sorted[0][1]

    return begins[obj_select], ends[obj_select]


def project_object_edge(img, dimension):
    """ scale the image, binarise with Othu and project to one dimension

    :param ndarray img:
    :param int dimension: select dimension for projection
    :return [float]:

    >>> img = np.zeros((20, 10, 3))
    >>> img[2:6, 1:7, :] = 1
    >>> img[10:17, 4:6, :] = 1
    >>> project_object_edge(img, 0).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    """
    assert dimension in (0, 1), 'not supported dimension %i' % dimension
    assert img.ndim == 3, 'unsupported image shape %r' % img.shape
    img_gray = np.mean(img, axis=-1)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    p_low, p_high = np.percentile(img_gray, (1, 95))
    img_gray = rescale_intensity(img_gray, in_range=(p_low, p_high))
    img_bin = img_gray > threshold_otsu(img_gray)
    img_edge = np.mean(img_bin, axis=1 - dimension)
    return img_edge


def load_large_image(img_path):
    """ loading very large images

    Note, for the loading we have to use matplotlib while ImageMagic nor other
     lib (opencv, skimage, Pillow) is able to load larger images then 64k or 32k.

    :param str img_path: path to the image
    :return ndarray: image
    """
    assert os.path.isfile(img_path), 'missing image: %s' % img_path
    img = plt.imread(img_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    if np.max(img) <= 1.5:
        np.clip(img, a_min=0, a_max=1, out=img)
        # this command split should reduce mount of required memory
        np.multiply(img, 255, out=img)
        img = img.astype(np.uint8, copy=False)
    return img


def save_large_image(img_path, img):
    """ saving large images more then 50k x 50k

    Note, for the saving we have to use openCV while other
    lib (matplotlib, Pillow, ITK) is not able to save larger images then 32k.

    :param str img_path: path to the new image
    :param ndarray img: image

    >>> img = np.zeros((2500, 3200, 4), dtype=np.uint8)
    >>> img[:, :, 0] = 255
    >>> img[:, :, 1] = 127
    >>> img_path = './sample-image.jpg'
    >>> save_large_image(img_path, img)
    >>> img2 = load_large_image(img_path)
    >>> img2[0, 0].tolist()
    [255, 127, 0]
    >>> img.shape[:2] == img2.shape[:2]
    True
    >>> os.remove(img_path)
    >>> img_path = './sample-image.png'
    >>> save_large_image(img_path, img.astype(np.uint16) * 255)
    >>> img3 = load_large_image(img_path)
    >>> img.shape[:2] == img3.shape[:2]
    True
    >>> img3[0, 0].tolist()
    [255, 127, 0]
    >>> save_large_image(img_path, img2 / 255. * 1.15)  # test overwrite message
    >>> os.remove(img_path)
    """
    # drop transparency
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    # for some reasons with linear interpolation some the range overflow (0, 1)
    if np.max(img) <= 1.5:
        np.clip(img, a_min=0, a_max=1, out=img)
        # this command split should reduce mount of required memory
        np.multiply(img, 255, out=img)
        img = img.astype(np.uint8, copy=False)
    # some tiff images have higher ranger int16
    elif np.max(img) > 255:
        img = img / 255
    # for images as integer clip the value range as (0, 255)
    if img.dtype != np.uint8:
        np.clip(img, a_min=0, a_max=255, out=img)
        img = img.astype(np.uint8, copy=False)
    if os.path.isfile(img_path):
        logging.debug('WARNING: this image will be overwritten: %s', img_path)
    # why cv2 imwrite changes the color of pics
    # https://stackoverflow.com/questions/42406338
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(img_path, img, IMAGE_COMPRESSION_OPTIONS)


def generate_pairing(count, step_hide=None):
    """ generate registration pairs with an option of hidden landmarks

    :param int count: total number of samples
    :param int|None step_hide: hide every N sample
    :return [(int, int)], [bool]: registration pairs

    >>> generate_pairing(4, None)  # doctest: +NORMALIZE_WHITESPACE
    ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
     [True, True, True, True, True, True])
    >>> generate_pairing(4, step_hide=3)  # doctest: +NORMALIZE_WHITESPACE
    ([(0, 1), (0, 2), (1, 2), (3, 1), (3, 2)],
     [False, False, True, False, False])
    """
    idxs_all = list(range(count))
    idxs_hide = idxs_all[::step_hide] if step_hide is not None else []
    # prune image on diagonal and missing both landmarks (target and source)
    idxs_pairs = [(i, j) for i in idxs_all for j in idxs_all
                  if i != j and j not in idxs_hide]
    # prune symmetric image pairs
    idxs_pairs = [(i, j) for k, (i, j) in enumerate(idxs_pairs)
                  if (j, i) not in idxs_pairs[:k]]
    public = [not (i in idxs_hide or j in idxs_hide) for i, j in idxs_pairs]
    return idxs_pairs, public


def parse_path_scale(path):
    """ from given path with annotation parse scale

    :param str path: path to the user folder
    :return int:

    >>> parse_path_scale('scale-.1pc')
    nan
    >>> parse_path_scale('user-JB_scale-50pc')
    50
    >>> parse_path_scale('scale-10pc')
    10
    """
    name = os.path.basename(path)
    obj = re.match(REEXP_FOLDER_SCALE, name)
    if obj is None:
        return np.nan
    scale = int(obj.groups()[0])
    return scale


def compute_convex_hull(landmarks):
    """ compute convex hull around landmarks

    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.spatial.ConvexHull.html
    https://stackoverflow.com/questions/21727199

    :param ndarray landmarks:
    :return ndarray:

    >>> np.random.seed(0)
    >>> pts = np.random.randint(15, 30, (10, 2))
    >>> compute_convex_hull(pts)
    array([[27, 20],
           [27, 25],
           [22, 24],
           [16, 21],
           [15, 18],
           [26, 18]])
    """
    chull = spatial.ConvexHull(landmarks)
    chull_points = landmarks[chull.vertices]
    return chull_points


def inside_polygon(polygon, point):
    """ check if a point is strictly inside the polygon

    :param ndarray|[] polygon: polygon contour
    :param ()|[] point: sample point
    :return bool: inside

    >>> poly = [[1, 1], [1, 3], [3, 3], [3, 1]]
    >>> inside_polygon(poly, [0, 0])
    False
    >>> inside_polygon(poly, [1, 1])
    False
    >>> inside_polygon(poly, [2, 2])
    True
    """
    path = Path(polygon)
    return path.contains_points([point])[0]


def list_sub_folders(path_folder, name='*'):
    """ list all sub folders with particular name pattern

    :param str path_folder: path to a particular folder
    :param str name: name pattern
    :return [str]:
    """
    sub_dirs = sorted([p for p in glob.glob(os.path.join(path_folder, name))
                       if os.path.isdir(p)])
    return sub_dirs
