"""
Useful function for managing Input/Output

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import logging
import warnings
from functools import wraps

import yaml
import cv2 as cv
import nibabel
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from skimage.color import (
    rgb2gray, gray2rgb, rgb2hsv, hsv2rgb, rgb2lab, lab2rgb, lch2lab, lab2lch)

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
    :param bool ok_existing: suppress warning for missing
    :return str|None: path to created folder

    >>> p_dir = create_folder('./sample-folder', ok_existing=True)
    >>> create_folder('./sample-folder', ok_existing=False)
    False
    >>> os.rmdir(p_dir)
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
    :return ndarray: np.array<np_points, dim> of landmarks points

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_landmarks('./sample_landmarks.csv', points)
    >>> pts1 = load_landmarks('./sample_landmarks.csv')
    >>> pts2 = load_landmarks('./sample_landmarks.pts')
    >>> np.array_equal(pts1, pts2)
    True
    >>> os.remove('./sample_landmarks.csv')
    >>> os.remove('./sample_landmarks.pts')

    >>> # Wrong loading
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
    elif ext == '.pts':
        return load_landmarks_pts(path_file)
    else:
        logging.error('not supported landmarks file: %s',
                      os.path.basename(path_file))


def load_landmarks_pts(path_file):
    """ load file with landmarks in txt format

    :param str path_file: path to the input file
    :return ndarray: np.array<np_points, dim> of landmarks points

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> p_lnds = save_landmarks_pts('./sample_landmarks.csv', points)
    >>> p_lnds
    './sample_landmarks.pts'
    >>> pts = load_landmarks_pts(p_lnds)
    >>> pts  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]])
    >>> os.remove('./sample_landmarks.pts')

    >>> # Empty landmarks
    >>> open('./sample_landmarks.pts', 'w').close()
    >>> load_landmarks_pts('./sample_landmarks.pts').size
    0
    >>> os.remove('./sample_landmarks.pts')
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
    :return ndarray: np.array<np_points, dim> of landmarks points

    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> p_lnds = save_landmarks_csv('./sample_landmarks', points)
    >>> p_lnds
    './sample_landmarks.csv'
    >>> pts = load_landmarks_csv(p_lnds)
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

    both used formats csv/pts is using the same coordinate frame,
    the origin (0, 0) is located in top left corner of the image

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0]
    landmarks = landmarks.values if isinstance(landmarks, pd.DataFrame) else landmarks
    save_landmarks_csv(path_file + '.csv', landmarks)
    save_landmarks_pts(path_file + '.pts', landmarks)


def save_landmarks_pts(path_file, landmarks):
    """ save landmarks into a txt file

    we are using VTK pointdata legacy format, ITK compatible::

        <index, point>
        <number of points>
        point1-x point1-y [point1-z]
        point2-x point2-y [point2-z]

    .. ref:: https://simpleelastix.readthedocs.io/PointBasedRegistration.html

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    :return str: file path
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0] + '.pts'
    lines = ['point', str(len(landmarks))]
    lines += [' '.join(str(i) for i in point) for point in landmarks]
    with open(path_file, 'w') as fp:
        fp.write('\n'.join(lines))
    return path_file


def save_landmarks_csv(path_file, landmarks):
    """ save landmarks into a csv file

    we are using simple format::

        ,X,Y
        0,point1-x,point1-y
        1,point2-x,point2-y

    :param str path_file: path to the output file
    :param landmarks: np.array<np_points, dim>
    :return str: file path
    """
    assert os.path.isdir(os.path.dirname(path_file)), \
        'missing folder "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0] + '.csv'
    df = pd.DataFrame(landmarks, columns=LANDMARK_COORDS)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(path_file)
    return path_file


def update_path(a_path, pre_path=None, lim_depth=5, absolute=True):
    """ bubble in the folder tree up until it found desired file
    otherwise return original one

    :param str a_path: original path
    :param str|None base_path: special case when you want to add something before
    :param int lim_depth: max depth of going up in the folder tree
    :param bool absolute: format as absolute path
    :return str: updated path if it exists otherwise the original one

    >>> os.path.exists(update_path('./birl', absolute=False))
    True
    >>> os.path.exists(update_path('/', absolute=False))
    True
    >>> os.path.exists(update_path('~', absolute=False))
    True
    """
    path_ = str(a_path)
    if path_.startswith('/'):
        return path_
    elif path_.startswith('~'):
        path_ = os.path.expanduser(path_)
    # special case when you want to add something before
    elif pre_path:
        path_ = os.path.join(pre_path, path_)

    tmp_path = path_[2:] if path_.startswith('./') else path_
    for _ in range(lim_depth):
        if os.path.exists(tmp_path):
            path_ = tmp_path
            break
        tmp_path = os.path.join('..', tmp_path)

    if absolute:
        path_ = os.path.abspath(path_)
    return path_


def io_image_decorate(func):
    """ costume decorator to suppers debug messages from the PIL function
    to suppress PIl debug logging
    - DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13

    :param func: decorated function
    :return func: output of the decor. function
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
def image_sizes(path_image, decimal=1):
    """ get image size (without loading image raster)

    :param str path_image: path to the image
    :param int decimal: rounding digits
    :return tuple(tuple(int,int),float): image size (height, width) and diagonal

    >>> img = np.random.random((50, 75, 3))
    >>> save_image('./test_image.jpg', img)
    >>> image_sizes('./test_image.jpg', decimal=0)
    ((50, 75), 90.0)
    >>> os.remove('./test_image.jpg')
    """
    assert os.path.isfile(path_image), 'missing image: %s' % path_image
    width, height = Image.open(path_image).size
    img_diag = np.sqrt(np.sum(np.array([height, width]) ** 2))
    return (height, width), np.round(img_diag, decimal)


@io_image_decorate
def load_image(path_image, force_rgb=True):
    """ load the image in value range (0, 1)

    :param str path_image: path to the image
    :param bool force_rgb: convert RGB image
    :return ndarray: np.array<height, width, ch>

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

    :param ndarray image: input image
    :return Image: output image

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

    >>> # Wrong path
    >>> save_image('./missing-path/any-image.png', np.zeros((10, 20)))
    False
    """
    path_dir = os.path.dirname(path_image)
    if not os.path.isdir(path_dir):
        logging.error('upper folder does not exists: "%s"', path_dir)
        return False
    image = convert_ndarray2image(image)
    image.save(path_image)


@io_image_decorate
def convert_image_to_nifti(path_image, path_out):
    """ converting normal image to Nifty Image

    :param str path_image: input image
    :param str path_out: path to output folder
    :return str: resulted image

    >>> path_img = './sample-image.png'
    >>> save_image(path_img, np.zeros((100, 200, 3)))
    >>> path_img2 = convert_image_to_nifti(path_img, '.')
    >>> os.path.isfile(path_img2)
    True
    >>> path_img3 = convert_image_from_nifti(path_img2, '.')
    >>> os.path.isfile(path_img3)
    True
    >>> list(map(os.remove, [path_img, path_img2, path_img3]))  # doctest: +ELLIPSIS
    [...]
    """
    img_name = os.path.splitext(os.path.basename(path_image))[0]
    path_img_out = os.path.join(path_out, img_name + '.nii')
    logging.debug('Convert image to Nifti format "%s" ->  "%s"', path_image, path_img_out)

    # img = Image.open(path_image).convert('LA')
    img = load_image(path_image)
    nim = nibabel.Nifti1Pair(img, np.eye(4))
    del img
    nibabel.save(nim, path_img_out)

    return path_img_out


@io_image_decorate
def convert_image_to_nifti_gray(path_image, path_out):
    """ converting normal image to Nifty Image

    :param str path_image: input image
    :param str path_out: path to output folder
    :return str: resulted image

    >>> path_img = './sample-image.png'
    >>> save_image(path_img, np.zeros((100, 200, 3)))
    >>> path_img2 = convert_image_to_nifti_gray(path_img, '.')
    >>> os.path.isfile(path_img2)
    True
    >>> path_img3 = convert_image_from_nifti(path_img2, '.')
    >>> os.path.isfile(path_img3)
    True
    >>> list(map(os.remove, [path_img, path_img2, path_img3]))  # doctest: +ELLIPSIS
    [...]
    """
    img_name = os.path.splitext(os.path.basename(path_image))[0]
    path_img_out = os.path.join(path_out, img_name + '.nii')
    logging.debug('Convert image to Nifti format "%s" ->  "%s"', path_image, path_img_out)

    # img = Image.open(path_image).convert('LA')
    img = rgb2gray(load_image(path_image))
    nim = nibabel.Nifti1Pair(np.swapaxes(img, 1, 0), np.eye(4))
    del img
    nibabel.save(nim, path_img_out)

    return path_img_out


@io_image_decorate
def convert_image_from_nifti(path_image, path_out):
    """ converting Nifti to standard image

    :param str path_image: path to input image
    :param str path_out: destination directory
    :return str: path to new image
    """
    img_name = os.path.splitext(os.path.basename(path_image))[0]
    path_img_out = os.path.join(path_out, img_name + '.jpg')
    logging.debug('Convert Nifti to image format "%s" ->  "%s"', path_image, path_img_out)
    nim = nibabel.load(path_image)

    if len(nim.get_data().shape) > 2:  # colour
        img = nim.get_data()
    else:  # gray
        img = np.swapaxes(nim.get_data(), 1, 0)

    if img.max() > 1.5:
        img = img / 255.

    save_image(path_img_out, img)
    return path_img_out


def image_resize(img, scale=1., v_range=255, dtype=int):
    """ rescale image with other optional formating

    :param ndarray img: input image
    :param float scale: the new image size is im_size * scale
    :param int|float v_range: desired output image range 1. or 255
    :param dtype: output image type
    :return ndarray: image

    >>> np.random.seed(0)
    >>> img = image_resize(np.random.random((250, 300, 3)), scale=2, v_range=255)
    >>> np.array(img.shape, dtype=int)
    array([500, 600,   3])
    >>> img.max()
    255
    """
    if scale is None or scale == 1:
        return img
    # scale the image accordingly
    interp = cv.INTER_CUBIC if scale > 1 else cv.INTER_LINEAR
    img = cv.resize(img, None, fx=scale, fy=scale, interpolation=interp)

    v_range = 255 if v_range > 1.5 else 1.
    # if resulting image is in range to 1 and desired is in 255
    if np.max(img) < 1.5 < v_range:
        np.multiply(img, 255, out=img)
        np.round(img, out=img)

    # convert image datatype
    if dtype is not None:
        img = img.astype(dtype)

    # clip image values in certain range
    np.clip(img, a_min=0, a_max=v_range, out=img)
    return img


@io_image_decorate
def convert_image_from_mhd(path_image, path_out_dir=None, img_ext='.png', scaling=None):
    """ convert standard image to MHD format

    .. ref:: https://www.programcreek.com/python/example/96382/SimpleITK.WriteImage

    :param str path_image: path to the input image
    :param str path_out_dir: path to output directory, if None use the input dir
    :param str img_ext: image extension like PNG or JPEG
    :param float|None scaling: image down-scaling,
        resulting image will be larger by this factor
    :return str: path to exported image

    >>> path_img = os.path.join(update_path('data_images'), 'images',
    ...                         'artificial_reference.jpg')
    >>> path_img = convert_image_to_mhd(path_img, scaling=1.5)
    >>> convert_image_from_mhd(path_img, scaling=1.5)  # doctest: +ELLIPSIS
    '...artificial_reference.png'
    """
    path_image = update_path(path_image)
    assert os.path.isfile(path_image), 'missing image: %s' % path_image
    # Reads the image using SimpleITK
    itk_image = sitk.ReadImage(path_image)

    # Convert the image to a numpy array first and then shuffle the dimensions
    # to get axis in the order z,y,x
    img = sitk.GetArrayFromImage(itk_image)
    # Scaling image if requested
    img = image_resize(img, scaling, v_range=255)

    # define output/destination path
    img_name = os.path.splitext(os.path.basename(path_image))[0]
    if not path_out_dir:
        path_out_dir = os.path.dirname(path_image)
    path_image = os.path.join(path_out_dir, img_name + img_ext)
    save_image(path_image, img)
    return path_image


@io_image_decorate
def convert_image_to_mhd(path_image, path_out_dir=None, to_gray=True, overwrite=True, scaling=None):
    """ converting standard image to MHD (Nifty format)

    .. ref:: https://stackoverflow.com/questions/37290631

    :param str path_image: path to the input image
    :param str path_out_dir: path to output directory, if None use the input dir
    :param bool overwrite: allow overwrite existing image
    :param float|None scaling: image up-scaling
        resulting image will be smaller by this factor
    :return str: path to exported image

    >>> path_img = os.path.join(update_path('data_images'), 'images',
    ...                         'artificial_moving-affine.jpg')
    >>> convert_image_to_mhd(path_img, scaling=2)  # doctest: +ELLIPSIS
    '...artificial_moving-affine.mhd'
    """
    path_image = update_path(path_image)
    # define output/destination path
    img_name = os.path.splitext(os.path.basename(path_image))[0]
    if not path_out_dir:
        path_out_dir = os.path.dirname(path_image)
    path_image_new = os.path.join(path_out_dir, img_name + '.mhd')
    # in case the image exists and you are not allowed to overwrite it
    if os.path.isfile(path_image_new) and not overwrite:
        logging.debug('skip converting since the image exists and no-overwrite: %s',
                      path_image_new)
        return path_image_new

    img = load_image(path_image)
    # if required and RGB on input convert to gray-scale
    if to_gray and img.ndim == 3 and img.shape[2] in (3, 4):
        img = rgb2gray(img)
    # Scaling image if requested
    scaling = 1. / scaling
    # the MHD usually require pixel value range (0, 255)
    img = image_resize(img, scaling, v_range=255)

    logging.debug('exporting image of size: %r', img.shape)
    image = sitk.GetImageFromArray(img.astype(np.uint8), isVector=False)

    # do not use text in MHD, othwerwise it crash DROP method
    sitk.WriteImage(image, path_image_new, False)
    return path_image_new


def image_histogram_matching(source, reference, use_color='hsv', norm_img_size=4096):
    """ adjust image histogram between two images

    Optionally transform the image to more continues color space.
    The source and target image does not need to be the same size, but RGB/gray.

    See cor related information:

    * https://www.researchgate.net/post/Histogram_matching_for_color_images
    * https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/histogram_matching.py
    * https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    * https://github.com/mapbox/rio-hist/issues/3

    :param ndarray source: 2D image to be transformed
    :param ndarray reference: reference 2D image
    :param str use_color: using color space for hist matching
    :param int norm_img_size: subsample image to this max size
    :return ndarray: transformed image

    >>> path_imgs = os.path.join(update_path('data_images'), 'rat-kidney_', 'scale-5pc')
    >>> img1 = load_image(os.path.join(path_imgs, 'Rat-Kidney_HE.jpg'))
    >>> img2 = load_image(os.path.join(path_imgs, 'Rat-Kidney_PanCytokeratin.jpg'))
    >>> image_histogram_matching(img1, img2).shape == img1.shape
    True
    >>> img = image_histogram_matching(img1[..., 0], np.expand_dims(img2[..., 0], 2))
    >>> img.shape == img1.shape[:2]
    True
    >>> image_histogram_matching(np.random.random((10, 20, 30, 5)),
    ...                          np.random.random((30, 10, 20, 5))).ndim
    4
    """
    # in case gray images normalise dimensionality
    def _normalise_image(img):
        # normalise gray-scale images
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        return img

    source = _normalise_image(source)
    reference = _normalise_image(reference)
    assert source.ndim == reference.ndim, 'the image dimensionality has to be equal'

    if source.ndim == 2:
        matched = histogram_match_cumulative_cdf(source, reference, norm_img_size=norm_img_size)
    elif source.ndim == 3:
        matched = np.empty(source.shape, dtype=source.dtype)

        conv_from_rgb, conv_to_rgb = CONVERT_RGB.get(use_color, (None, None))
        if conv_from_rgb:
            source = conv_from_rgb(source)
            reference = conv_from_rgb(reference)
        for ch in range(source.shape[-1]):
            matched[..., ch] = histogram_match_cumulative_cdf(
                source[..., ch], reference[..., ch], norm_img_size=norm_img_size)
        if conv_to_rgb:
            matched = conv_to_rgb(matched)
    else:
        logging.warning('unsupported image dimensions: %r', source.shape)
        matched = source

    return matched


def histogram_match_cumulative_cdf(source, reference, norm_img_size=1024):
    """ Adjust the pixel values of a gray-scale image such that its histogram
    matches that of a target image

    :param ndarray source: 2D image to be transformed, np.array<height1, width1>
    :param ndarray reference: reference 2D image, np.array<height2, width2>
    :return ndarray: transformed image, np.array<height1, width1>

    >>> np.random.seed(0)
    >>> img = histogram_match_cumulative_cdf(np.random.randint(128, 145, (150, 200)),
    ...                                      np.random.randint(0, 18, (200, 180)))
    >>> img.astype(int)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[13, 16,  0, ..., 12,  2,  5],
           [17,  9,  1, ..., 16,  9,  0],
           [11, 12, 14, ...,  8,  5,  4],
           ...,
           [12,  6,  3, ..., 15,  0,  3],
           [11, 17,  2, ..., 12, 12,  5],
           [ 6, 12,  3, ...,  8,  0,  1]])
    >>> np.bincount(img.ravel()).astype(int)  # doctest: +NORMALIZE_WHITESPACE
    array([1705, 1706, 1728, 1842, 1794, 1866, 1771,    0, 1717, 1752, 1757,
           1723, 1823, 1833, 1749, 1718, 1769, 1747])
    >>> img_source = np.random.randint(50, 245, (2500, 3000)).astype(float)
    >>> img_source[-1, -1] = 255
    >>> img = histogram_match_cumulative_cdf(img_source / 255., img)
    >>> np.array(img.shape, dtype=int)
    array([2500, 3000])
    """
    # use smaller image
    step_src = max(1, int(np.max(source.shape) / norm_img_size))
    step_ref = max(1, int(np.max(reference.shape) / norm_img_size))

    # determine if we need remember that output should be float valued
    out_float = source.max() < 1.5
    # if the image is flout in range (0, 1) extend it
    source = np.round(source * 255) if source.max() < 1.5 else source
    # here we need convert to int values
    source = source.astype(np.int16)
    # work with just a small image
    src_small = source[::step_src, ::step_src]

    # here we need work with just a small image
    ref_small = reference[::step_ref, ::step_ref]
    # if the image is flout in range (0, 1) extend it
    ref_small = np.round(ref_small * 255) if reference.max() < 1.5 else ref_small
    # here we need convert to int values
    ref_small = ref_small.astype(np.int16)

    src_counts = np.bincount(src_small.ravel())
    # src_values = np.arange(0, len(src_counts))
    ref_counts = np.bincount(ref_small.ravel())
    ref_values = np.arange(0, len(ref_counts))
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / float(source.size)
    ref_quantiles = np.cumsum(ref_counts) / float(reference.size)

    interp_values = np.round(np.interp(src_quantiles, ref_quantiles, ref_values))
    # in case that it overflows, due to sampling step may skip some high values
    if source.max() >= len(interp_values):
        logging.warning('source image max value %i overflow generated LUT of size %i',
                        source.max(), len(interp_values))
        # then clip the source image values to fit ot the range
        source[source >= len(interp_values)] = len(interp_values) - 1
    matched = np.round(interp_values)[source].astype(np.int16)

    if out_float:
        matched = matched.astype(float) / 255.
    return matched


def load_config_yaml(path_config):
    """ loading the

    :param str path_config:
    :return dict:

    >>> p_conf = './testing-congif.yaml'
    >>> save_config_yaml(p_conf, {'a': 2})
    >>> load_config_yaml(p_conf)
    {'a': 2}
    >>> os.remove(p_conf)
    """
    with open(path_config, 'r') as fp:
        config = yaml.safe_load(fp)
    return config


def save_config_yaml(path_config, config):
    """ exporting configuration as YAML file

    :param str path_config:
    :param dict config:
    """
    with open(path_config, 'w') as fp:
        yaml.dump(config, fp, default_flow_style=False)
