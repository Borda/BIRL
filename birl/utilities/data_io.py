"""
Useful function for managing Input/Output

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging
import warnings
from functools import wraps

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import gray2rgb

#: landmarks coordinates, loading from CSV file
LANDMARK_COORDS = ['X', 'Y']
# PIL.Image.DecompressionBombError: could be decompression bomb DOS attack.
# SEE: https://gitlab.mister-muffin.de/josch/img2pdf/issues/42
Image.MAX_IMAGE_PIXELS = None


def create_folder(path_folder, ok_existing=True):
    """ create a folder if it not exists

    :param str path_folder: path to creating folder
    :param bool ok_existing: suppres warning for missing
    :return str|None: path to created folder

    >>> p_dir = create_folder('./sample-folder', ok_existing=True)
    >>> create_folder('./sample-folder', ok_existing=False)
    False
    >>> import shutil
    >>> shutil.rmtree(p_dir)
    """
    path_folder = os.path.abspath(path_folder)
    if not os.path.isdir(path_folder):
        try:
            os.makedirs(path_folder, mode=0o775)
        except Exception:
            logging.exception('Something went wrong (probably parallel access),'
                              ' the status of "%s" is %s', path_folder,
                              os.path.isdir(path_folder))
            path_folder = None
    elif not ok_existing:
        logging.warning('Folder already exists: %s', path_folder)
        path_folder = False

    return path_folder


def load_landmarks(path_file):
    """ load landmarks in csv and txt format

    :param str path_file: path to the input file
    :return: np.array<np_points, dim>

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_landmarks('./sample_landmarks.csv', points)
    >>> pts1 = load_landmarks('./sample_landmarks.csv')
    >>> pts2 = load_landmarks('./sample_landmarks.txt')
    >>> np.array_equal(pts1, pts2)
    True
    >>> os.remove('./sample_landmarks.csv')
    >>> os.remove('./sample_landmarks.txt')

    Wrong loading
    >>> load_landmarks('./sample_landmarks.file')
    >>> open('./sample_landmarks.file', 'w').close()
    >>> load_landmarks('./sample_landmarks.file')
    >>> os.remove('./sample_landmarks.file')
    """
    if not os.path.isfile(path_file):
        logging.warning('missing landmarks "%s"', path_file)
        return None
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
    >>> pts = load_landmarks_txt('./sample_landmarks.txt')
    >>> pts  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]])
    >>> os.remove('./sample_landmarks.txt')

    Empty landmarks
    >>> open('./sample_landmarks.txt', 'w').close()
    >>> load_landmarks_txt('./sample_landmarks.txt').size
    0
    >>> os.remove('./sample_landmarks.txt')
    """
    assert os.path.isfile(path_file), 'missing file "%s"' % path_file
    with open(path_file, 'r') as fp:
        data = fp.read()
        lines = data.split('\n')
        # lines = [re.sub("(\\r|)\\n$", '', line) for line in lines]
    if len(lines) < 2:
        logging.warning('invalid format: file has less then 2 lines, "%r"', lines)
        return np.zeros((0, 2))
    nb_points = int(lines[1])
    points = [[float(n) for n in line.split()]
              for line in lines[2:] if line]
    assert nb_points == len(points), 'number of declared (%i) and found (%i) ' \
                                     'does not match' % (nb_points, len(points))
    return np.array(points, dtype=np.float)


def load_landmarks_csv(path_file):
    """ load file with landmarks in cdv format

    :param str path_file: path to the input file
    :return: np.array<np_points, dim>

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_landmarks_csv('./sample_landmarks.csv', points)
    >>> pts = load_landmarks_csv('./sample_landmarks.csv')
    >>> pts  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[1, 2],
           [3, 4],
           [5, 6]]...)
    >>> os.remove('./sample_landmarks.csv')
    """
    assert os.path.isfile(path_file), 'missing file "%s"' % path_file
    df = pd.read_csv(path_file, index_col=0)
    points = df[LANDMARK_COORDS].values
    return points


def save_landmarks(path_file, landmarks):
    """ save landmarks into a specific file

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0]
    save_landmarks_csv(path_file + '.csv', landmarks)
    save_landmarks_txt(path_file + '.txt', landmarks)


def save_landmarks_txt(path_file, landmarks):
    """ save landmarks into a txt file

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    lines = ['point', str(len(landmarks))]
    lines += [' '.join(str(i) for i in point) for point in landmarks]
    with open(path_file, 'w') as fp:
        fp.write('\n'.join(lines))


def save_landmarks_csv(path_file, landmarks):
    """ save landmarks into a csv file

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    assert os.path.splitext(path_file)[-1] == '.csv', \
        'wrong file extension "%s"' % os.path.basename(path_file)
    df = pd.DataFrame(landmarks, columns=LANDMARK_COORDS)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(path_file)


def update_path(path_file, lim_depth=5, absolute=True):
    """ bubble in the folder tree up intil it found desired file
    otherwise return original one

    :param str path_file: original path
    :param int lim_depth: lax depth of going up
    :param bool absolute: return absolute path
    :return str:

    >>> os.path.exists(update_path('./birl', absolute=False))
    True
    >>> os.path.exists(update_path('/', absolute=False))
    True
    >>> os.path.exists(update_path('~', absolute=False))
    True
    """
    if path_file.startswith('/'):
        return path_file
    elif path_file.startswith('~'):
        path_file = os.path.expanduser(path_file)

    tmp_path = path_file[2:] if path_file.startswith('./') else path_file
    for _ in range(lim_depth):
        if os.path.exists(tmp_path):
            path_file = tmp_path
            break
        tmp_path = os.path.join('..', tmp_path)

    if absolute:
        path_file = os.path.abspath(path_file)
    return path_file


def io_image_decorate(func):
    """ costume decorator to suppers debug messages from the PIL function
    to suppress PIl debug logging
    - DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13

    :param func:
    :return:
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = func(*args, **kwargs)
        logging.getLogger().setLevel(log_level)
        return response
    return wrap


@io_image_decorate
def image_size(path_image, decimal=1):
    """ get image size (without loading image raster)

    :param str path_image: path to the image
    :param int decimal: rounding digits
    :return (int, int), float: image size and diagonal

    >>> img = np.random.random((50, 75, 3))
    >>> save_image('./test_image.jpg', img)
    >>> image_size('./test_image.jpg', decimal=0)
    ((50, 75), 90.0)
    >>> os.remove('./test_image.jpg')
    """
    assert os.path.isfile(path_image), 'missing image: %s' % path_image
    img_size = Image.open(path_image).size[::-1]
    img_diag = np.sqrt(np.sum(np.array(img_size) ** 2))
    return img_size, np.round(img_diag, decimal)


@io_image_decorate
def load_image(path_image, force_rgb=True):
    """ load the image in value range (0, 1)

    :param str path_image: path to the image
    :param bool force_rgb: convert RGB image
    :return: np.array<height, width, ch>

    >>> img = np.random.random((50, 50))
    >>> save_image('./test_image.jpg', img)
    >>> img2 = load_image('./test_image.jpg')
    >>> img2.max() <= 1.
    True
    >>> os.remove('./test_image.jpg')
    """
    assert os.path.isfile(path_image), 'missing image "%s"' % path_image
    image = np.array(Image.open(path_image))
    while image.max() > 1.5:
        image = image / 255.
    if force_rgb and (image.ndim == 2 or image.shape[2] == 1):
        image = image[:, :, 0] if image.ndim == 3 else image
        image = gray2rgb(image)
    return image.astype(np.float32)


def convert_ndarray2image(image):
    """ convert ndarray to PIL image if it not already

    :param image: np.ndarray
    :return: Image

    >>> img = np.random.random((50, 50, 3))
    >>> image = convert_ndarray2image(img)
    >>> isinstance(image, Image.Image)
    True
    """
    if isinstance(image, np.ndarray):
        if np.max(image) <= 1.5:
            image = image * 255
        np.clip(image, a_min=0, a_max=255, out=image)
        image = Image.fromarray(image.astype(np.uint8))
    return image


@io_image_decorate
def save_image(path_image, image):
    """ save the image into given path

    :param str path_image: path to the image
    :param image: np.array<height, width, ch>

    Wrong path
    >>> save_image('./missing-path/any-image.png', np.zeros((10, 20)))
    False
    """
    path_dir = os.path.dirname(path_image)
    if not os.path.isdir(path_dir):
        logging.error('upper folder does not exists: "%s"', path_dir)
        return False
    image = convert_ndarray2image(image)
    image.save(path_image)
