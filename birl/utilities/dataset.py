"""
Some functionality related to dataset

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import re
import glob
import logging

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from scipy import spatial, optimize
from matplotlib.path import Path
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity
from cv2 import (IMWRITE_JPEG_QUALITY, IMWRITE_PNG_COMPRESSION, GaussianBlur,
                 cvtColor, COLOR_RGBA2RGB, COLOR_RGB2BGR, imwrite)

#: threshold of tissue/background presence on potential cutting line
TISSUE_CONTENT = 0.01
#: supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
# https://github.com/opencv/opencv/issues/6729
# https://www.life2coding.com/save-opencv-images-jpeg-quality-png-compression
IMAGE_COMPRESSION_OPTIONS = (IMWRITE_JPEG_QUALITY, 98) + \
                            (IMWRITE_PNG_COMPRESSION, 9)
#: template for detecting/parsing scale from folder name
REEXP_FOLDER_SCALE = r'\S*scale-(\d+)pc'
# ERROR:root:error: Image size (... pixels) exceeds limit of ... pixels,
# could be decompression bomb DOS attack.
# SEE: https://gitlab.mister-muffin.de/josch/img2pdf/issues/42
Image.MAX_IMAGE_PIXELS = None
#: maximal image size for visualisations, larger images will be downscaled
MAX_IMAGE_SIZE = 5000


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
    img_gray = GaussianBlur(img_gray, (5, 5), 0)
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
        img = cvtColor(img, COLOR_RGBA2RGB)
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
    img = cvtColor(img, COLOR_RGB2BGR)
    imwrite(img_path, img, IMAGE_COMPRESSION_OPTIONS)


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


def parse_path_scale(path_folder):
    """ from given path with annotation parse scale

    :param str path_folder: path to the scale folder
    :return int: scale

    >>> parse_path_scale('scale-.1pc')
    nan
    >>> parse_path_scale('user-JB_scale-50pc')
    50
    >>> parse_path_scale('scale-10pc')
    10
    """
    folder = os.path.basename(path_folder)
    obj = re.match(REEXP_FOLDER_SCALE, folder)
    if obj is None:
        return np.nan
    scale = int(obj.groups()[0])
    return scale


def line_angle_2d(point_begin, point_end, deg=True):
    """ Compute direction of line with given two points

    the zero is horizontal in direction [1, 0]

    :param [float] point_begin: starting line point
    :param [float] point_end: ending line point
    :param bool deg: return angle in degrees
    :return float:

    >>> [line_angle_2d([0, 0], p) for p in ((1, 0), (0, 1), (-1, 0), (0, -1))]
    [0.0, 90.0, 180.0, -90.0]
    >>> line_angle_2d([1, 1], [2, 3])  # doctest: +ELLIPSIS
    63.43...
    >>> line_angle_2d([1, 2], [-2, -3])  # doctest: +ELLIPSIS
    -120.96...
    """
    diff = np.asarray(point_end) - np.asarray(point_begin)
    angle = -np.arctan2(diff[0], diff[1]) + np.pi / 2.
    if deg:
        angle = angle / np.pi * 180.
    angle = norm_angle(angle, deg)
    return angle


def norm_angle(angle, deg=True):
    """ Normalise to be in range (-180, 180) degrees

    :param float angle: inut angle
    :param bool deg: use degrees
    :return float: norma angle
    """
    alpha = 180. if deg else np.pi
    while angle > alpha:
        angle = -2 * alpha + angle
    while angle < -alpha:
        angle = 2 * alpha + angle
    return angle


def is_point_inside_perpendicular(point_begin, point_end, point_test):
    """ If point is left from line and perpendicularly in between line segment

    Note that negative response does not mean that that the point is on tight side

    :param [float] point_begin: starting line point
    :param [float] point_end: ending line point
    :param [float] point_test: testing point
    :return int: gives +1 if it is above, -1 if bellow and 0 elsewhere

    >>> is_point_inside_perpendicular([1, 1], [3, 1], [2, 2])
    1
    >>> is_point_inside_perpendicular([1, 1], [3, 1], [2, 0])
    -1
    >>> is_point_inside_perpendicular([1, 1], [3, 1], [4, 2])
    0
    """
    angle_line = line_angle_2d(point_begin, point_end, deg=True)
    # compute angle of end - test compare to begin - end
    angle_a = norm_angle(line_angle_2d(point_end, point_test, deg=True) - angle_line, deg=True)
    # compute angle of begin - test compare to begin - end
    angle_b = norm_angle(line_angle_2d(point_begin, point_test, deg=True) - angle_line, deg=True)
    if (angle_a >= 90) and (angle_b <= 90):
        state = 1
    elif (angle_a <= -90) and (angle_b >= -90):
        state = -1
    else:
        state = 0
    return state


def is_point_in_quadrant_left(point_begin, point_end, point_test):
    """ If point is left quadrant from line end point

    Note that negative response does not mean that that the point is on tight side

    :param [float] point_begin: starting line point
    :param [float] point_end: ending line point
    :param [float] point_test: testing point
    :return int: gives +1 if it is above, -1 if bellow and 0 elsewhere

    >>> is_point_in_quadrant_left([1, 1], [3, 1], [2, 2])
    1
    >>> is_point_in_quadrant_left([3, 1], [1, 1], [2, 0])
    1
    >>> is_point_in_quadrant_left([1, 1], [3, 1], [2, 0])
    -1
    >>> is_point_in_quadrant_left([1, 1], [3, 1], [4, 2])
    0
    """
    angle_line = line_angle_2d(point_begin, point_end, deg=True)
    # compute angle of end - test compare to begin - end
    angle_pt = norm_angle(line_angle_2d(point_end, point_test, deg=True) - angle_line, deg=True)
    if (180 >= angle_pt >= 90) or angle_pt == -180:
        state = 1
    elif (-180 <= angle_pt <= -90) or angle_pt == 180:
        state = -1
    else:
        state = 0
    return state


def is_point_above_line(point_begin, point_end, point_test):
    """ If point is left from line

    :param [float] point_begin: starting line point
    :param [float] point_end: ending line point
    :param [float] point_test: testing point
    :return bool: left from line

    >>> is_point_above_line([1, 1], [2, 2], [3, 4])
    True
    """
    # compute angle of end - test compare to begin - end
    angle_line = line_angle_2d(point_begin, point_end, deg=True)
    angle_test = line_angle_2d(point_end, point_test, deg=True)
    state = 0 <= norm_angle(angle_test - angle_line, deg=True) <= 180
    return state


def compute_half_polygon(landmarks, idx_start=0, idx_end=-1):
    """ compute half polygon path

    :param int idx_start: index of starting point
    :param int idx_end: index of ending point
    :param ndarray landmarks: set of points
    :return ndarray:

    >>> pts = [(-1, 1), (0, 0), (0, 2), (1, 1), (1, -0.5), (2, 0)]
    >>> compute_half_polygon(pts, idx_start=0, idx_end=-1)
    [[-1.0, 1.0], [0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]
    >>> compute_half_polygon(pts[:2], idx_start=-1, idx_end=0)
    [[-1, 1], [0, 0]]
    >>> pts = [[0, 2], [1, 5], [2, 4], [2, 5], [4, 4], [4, 6], [4, 8], [5, 8], [5, 8]]
    >>> compute_half_polygon(pts)
    [[0, 2], [1, 5], [2, 5], [4, 6], [4, 8], [5, 8]]
    """
    # the three or less are always minimal polygon
    if len(landmarks) < 3:
        return np.array(landmarks).tolist()
    # normalise indexes to be larger then 0
    while idx_start < 0:
        idx_start = len(landmarks) + idx_start
    while idx_end < 0:
        idx_end = len(landmarks) + idx_end
    # select points
    pt_begin, pt_end = landmarks[idx_start], landmarks[idx_end]
    del idx_start, idx_end
    # only unique points
    points = np.vstack({tuple(lnd) for lnd in landmarks})
    dists = spatial.distance.cdist(points, points, metric='euclidean')
    poly = [pt_begin]

    def _in(pt0, pts):
        return any([np.array_equal(pt, pt0) for pt in pts])

    def _disturbed(poly, pt_new, pt_test):
        last = is_point_in_quadrant_left(poly[-1], pt_new, pt_test) == 1
        path = sum(is_point_inside_perpendicular(pt0, pt1, pt_test)
                   for pt0, pt1 in zip(poly, poly[1:] + [pt_new])) < 0
        return last and not path

    # iterated until you add the lst point to chain
    while not np.array_equal(poly[-1], pt_end):
        idx_last = np.argmax([np.array_equal(pt, poly[-1]) for pt in points])
        # walk over ordered list by distance starting with the closest
        pt_order = np.argsort(dists[idx_last])
        # iterate over all possible candidates not in chain already
        for pt0 in (pt for pt in points[pt_order] if not _in(pt, poly)):
            # find a point which does not have any point on the left perpendic
            if any(_disturbed(poly, pt0, pt) for pt in points if not _in(pt, poly + [pt0])):
                continue
            else:
                poly.append(pt0)
                break
    poly = np.array(poly).tolist()
    return poly


def get_close_diag_corners(points):
    """ finds points closes to the top left and bottom right corner

    :param ndarray points: set of points
    :return (ndarray, ndarray): begin and end of imaginary diagonal

    >>> np.random.seed(0)
    >>> points = np.random.randint(1, 9, (20, 2))
    >>> get_close_diag_corners(points)
    (array([1, 2]), array([7, 8]), (12, 10))
    """
    pt_min = np.min(points, axis=0)
    pt_max = np.max(points, axis=0)
    dists = spatial.distance.cdist([pt_min, pt_max], points)
    idx_begin = np.argmin(dists[0])
    idx_end = np.argmin(dists[1])
    pt_begin = points[np.argmin(dists[0])]
    pt_end = points[np.argmin(dists[1])]
    return pt_begin, pt_end, (idx_begin, idx_end)


def simplify_polygon(points, tol_degree=5):
    """ simplify path, drop point on the same line

    :param ndarray points: point in polygon
    :param float tol_degree: tolerance on change in orientation
    :return [[float]]:

    >>> pts = [[1, 2], [2, 4], [1, 5], [2, 8], [3, 8], [5, 8], [7, 8], [8, 7],
    ...     [8, 5], [8, 3], [8, 1], [7, 1], [6, 1], [4, 1], [3, 1], [3, 2], [2, 2]]
    >>> simplify_polygon(pts)
    [[1, 2], [2, 4], [1, 5], [2, 8], [7, 8], [8, 7], [8, 1], [3, 1], [3, 2]]
    """
    if len(points) < 3:
        return points
    path = [points[0]]
    for i in range(1, len(points)):
        angle0 = line_angle_2d(path[-1], points[i], deg=True)
        angle1 = line_angle_2d(points[i], points[(i + 1) % len(points)], deg=True)
        if abs(norm_angle(angle0 - angle1, deg=True)) > tol_degree:
            path.append(points[i])
    return np.array(path).tolist()


def compute_bounding_polygon(landmarks):
    """ get the polygon where all point lies inside

    :param ndarray landmarks: set of points
    :return ndarray: pints of polygon

    >>> np.random.seed(0)
    >>> points = np.random.randint(1, 9, (45, 2))
    >>> compute_bounding_polygon(points)  # doctest: +NORMALIZE_WHITESPACE
    [[1, 2], [2, 4], [1, 5], [2, 8], [7, 8], [8, 7], [8, 1], [3, 1], [3, 2]]
    """
    # the three or less are always mimimal polygon
    if len(landmarks) <= 3:
        return np.array(landmarks)
    points = np.array(landmarks)
    # split to two half by diagonal from [min, min] to [max, max]
    points = points[points[:, 0].argsort()]
    pt_begin, pt_end, _ = get_close_diag_corners(points)
    is_above = np.array([is_point_above_line(pt_begin, pt_end, pt) for pt in points])

    poly = []
    # compute one curve starting with [min, min] until [max, max] is added
    pts_above = [pt_begin] + [pt for i, pt in enumerate(points) if is_above[i]] + [pt_end]
    poly += compute_half_polygon(pts_above, idx_start=0, idx_end=-1)[:-1]
    # analogy got second curve from [max, max] to [min, min]
    pts_bellow = [pt_begin] + [pt for i, pt in enumerate(points) if not is_above[i]] + [pt_end]
    poly += compute_half_polygon(pts_bellow, idx_start=-1, idx_end=0)[:-1]
    return simplify_polygon(poly)


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

    >>> from birl.utilities.data_io import update_path
    >>> paths = list_sub_folders(update_path('data_images'))
    >>> list(map(os.path.basename, paths))
    ['images', 'landmarks', 'lesions_', 'rat-kidney_']
    """
    sub_dirs = sorted([p for p in glob.glob(os.path.join(path_folder, name))
                       if os.path.isdir(p)])
    return sub_dirs


def common_landmarks(points1, points2, threshold=0.5):
    """ find common landmarks in two sets

    :param ndarray|[[float]] points1: first point set
    :param ndarray|[[float]] points2: second point set
    :param float threshold: threshold for assignment
    :return [bool]:

    >>> np.random.seed(0)
    >>> common = np.random.random((5, 2))
    >>> pts1 = np.vstack([common, np.random.random((10, 2))])
    >>> pts2 = np.vstack([common, np.random.random((15, 2))])
    >>> common_landmarks(pts1, pts2, threshold=0.1)
    array([[ 0,  0],
           [ 1,  1],
           [ 2,  2],
           [ 3,  3],
           [ 4,  4],
           [14, 15]])
    """
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    dist = spatial.distance.cdist(points1, points2, 'euclidean')
    ind_row, ind_col = optimize.linear_sum_assignment(dist)
    dist_sel = dist[ind_row, ind_col]
    pairs = [(i, j) for (i, j, d) in zip(ind_row, ind_col, dist_sel)
             if d < threshold]
    return np.array(pairs, dtype=int)


def args_expand_images(parser, nb_workers=1, overwrite=True):
    """ expand the parser by standard parameters related to images:
        * image paths
        * allow overwrite (optional)
        * number of jobs

    :param obj parser: existing parser
    :param int nb_workers: number threads by default
    :param bool overwrite: allow overwrite images
    :return obj:

    >>> import argparse
    >>> args_expand_images(argparse.ArgumentParser())  # doctest: +ELLIPSIS
    ArgumentParser(...)
    """
    parser.add_argument('-i', '--path_images', type=str, required=True,
                        help='path (pattern) to the input image')
    parser.add_argument('--nb_workers', type=int, required=False, default=nb_workers,
                        help='number of processes running in parallel')
    if overwrite:
        parser.add_argument('--overwrite', action='store_true', required=False,
                            default=False, help='allow overwrite existing images')
    return parser


def args_expand_parse_images(parser, nb_workers=1, overwrite=True):
    """ expand the parser by standard parameters related to images:
        * image paths
        * allow overwrite (optional)
        * number of jobs

    :param obj parser: existing parser
    :param int nb_workers: number threads by default
    :param bool overwrite: allow overwrite images
    :return {}:
    """
    parser = args_expand_images(parser, nb_workers, overwrite)
    args = vars(parser.parse_args())
    args['path_images'] = os.path.expanduser(args['path_images'])
    return args


def estimate_scaling(images, max_size=MAX_IMAGE_SIZE):
    """ find scaling for given set of images and maximal image size

    :param [ndarray] images: input images
    :param float max_size: max image size in any dimension
    :return float: scaling in range (0, 1)

    >>> estimate_scaling([np.zeros((12000, 300, 3))])  # doctest: +ELLIPSIS
    0.4...
    >>> estimate_scaling([np.zeros((1200, 800, 3))])
    1.0
    """
    sizes = [img.shape[:2] for img in images if img is not None]
    if not sizes:
        return 1.
    max_dim = np.max(sizes)
    scale = np.round(float(max_size) / max_dim, 1) if max_dim > max_size else 1.
    return scale


def scale_large_images_landmarks(images, landmarks):
    """ scale images and landmarks up to maximal image size

    :param [ndarray] images: list of images
    :param [ndarray] landmarks: list of landmarks
    :return ([ndarray], [ndarray]): lists of images and landmarks

    >>> scale_large_images_landmarks([np.zeros((8000, 500, 3), dtype=np.uint8)], [None, None])  # doctest: +ELLIPSIS
    ([array(...)], [None, None])
    """
    if not images:
        return images, landmarks
    scale = estimate_scaling(images)
    if scale < 1.:
        logging.debug('One or more images ra larger then recommended size for visualisation,'
                      ' an resize with factor %f will be applied', scale)
    # using float16 as image raise TypeError: src data type = 23 is not supported
    images = [cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
              if img is not None else None for img in images]
    landmarks = [lnds * scale if lnds is not None else None for lnds in landmarks]
    return images, landmarks
