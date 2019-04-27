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
from skimage.color import gray2rgb, rgb2hsv, hsv2rgb, rgb2lab, lab2rgb, lch2lab, lab2lch

#: landmarks coordinates, loading from CSV file
LANDMARK_COORDS = ['X', 'Y']
# PIL.Image.DecompressionBombError: could be decompression bomb DOS attack.
# SEE: https://gitlab.mister-muffin.de/josch/img2pdf/issues/42
Image.MAX_IMAGE_PIXELS = None
CONVERT_RGB = {
    'hsv': (rgb2hsv, hsv2rgb),
    'lab': (rgb2lab, lab2rgb),
    'lch': (lambda img: lab2lch(rgb2lab(img)),
            lambda img: lab2rgb(lch2lab(img))),
}


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


def image_histogram_matching(source, reference, use_color='hsv'):
    """ adjust image histogram between two images

    https://www.researchgate.net/post/Histogram_matching_for_color_images
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/histogram_matching.py
    https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    https://github.com/mapbox/rio-hist/issues/3

    :param ndarray source: image to be transformed
    :param ndarray reference: reference image
    :return ndarray: transformed image

    >>> path_imgs = os.path.join(update_path('data_images'), 'rat-kidney_', 'scale-5pc')
    >>> img1 = load_image(os.path.join(path_imgs, 'Rat-Kidney_HE.jpg'))
    >>> img2 = load_image(os.path.join(path_imgs, 'Rat-Kidney_PanCytokeratin.jpg'))
    >>> image_histogram_matching(img1[..., 0], img2[..., 0]).shape == img1.shape[:2]
    True
    >>> image_histogram_matching(img1, img2).shape == img1.shape
    True
    """
    # in case gray images normalise dimensionality
    def _normalise_image(img):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        # if img.max() < 1.5:
        #     img = np.clip(np.round(img * 255), 0, 255)  # .astype(np.uint8)
        # assert img.max() > 1.5, 'expected range of source image is (0, 255)'
        return img

    source = _normalise_image(source)
    reference = _normalise_image(reference)
    assert source.ndim == reference.ndim, 'the image dimensionality has to be equal'

    if source.ndim == 2:
        matched = histogram_match_cumulative_cdf(source, reference)
    elif source.ndim == 3:
        matched = np.empty(source.shape, dtype=source.dtype)

        conv_from_rgb, conv_to_rgb = CONVERT_RGB.get(use_color, (None, None))
        if conv_from_rgb:
            source = conv_from_rgb(source)
            reference = conv_from_rgb(reference)
        for ch in range(source.shape[-1]):
            matched[..., ch] = histogram_match_cumulative_cdf(source[..., ch],
                                                              reference[..., ch])
        if conv_to_rgb:
            matched = conv_to_rgb(matched)
    else:
        logging.warning('unsupported image dimensions: %r', source.shape)
        matched = source

    return matched


def histogram_match_cumulative_cdf(source, reference, norm_img_size=1024):
    """ Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    :param source:
    :param template:
    :return:

    >>> np.random.seed(0)
    >>> img = histogram_match_cumulative_cdf(np.random.randint(0, 18, (10, 12)),
    ...                                      np.random.randint(128, 145, (15, 13)))
    >>> img.astype(int)
    array([[139, 142, 129, 132, 132, 135, 137, 133, 135, 139, 130, 135],
           [135, 141, 144, 134, 140, 136, 137, 143, 134, 142, 142, 129],
           [132, 144, 141, 135, 129, 130, 137, 129, 138, 132, 139, 130],
           [129, 129, 133, 134, 135, 136, 144, 142, 133, 137, 138, 130],
           [130, 135, 137, 132, 135, 139, 141, 129, 141, 132, 139, 138],
           [139, 133, 135, 133, 142, 132, 139, 133, 136, 141, 142, 132],
           [142, 140, 143, 144, 134, 137, 132, 129, 134, 129, 144, 133],
           [130, 143, 132, 130, 138, 140, 143, 135, 137, 129, 138, 139],
           [130, 130, 132, 132, 141, 132, 144, 141, 137, 130, 133, 138],
           [139, 136, 139, 130, 143, 129, 129, 135, 141, 138, 136, 140]])
    """
    source = np.round(source * 255) if source.max() < 1.5 else source
    source = source.astype(int)
    out_float = reference.max() < 1.5
    reference = np.round(reference * 255) if reference.max() < 1.5 else reference
    reference = reference.astype(int)

    # use smaller image
    step = int(np.max(np.array([source.shape, reference.shape])) / norm_img_size)
    step = max(1, step)
    src_counts = np.bincount(source[::step, ::step].ravel())
    # src_values = np.arange(0, len(src_counts))
    ref_counts = np.bincount(reference[::step, ::step].ravel())
    ref_values = np.arange(0, len(ref_counts))
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / float(source.size)
    ref_quantiles = np.cumsum(ref_counts) / float(reference.size)

    interp_a_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    matched = np.round(interp_a_values)[source]

    if out_float:
        matched = matched.astype(float) / 255.
    return matched
