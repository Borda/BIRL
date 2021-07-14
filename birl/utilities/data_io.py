"""
Useful function for managing Input/Output

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import os
import warnings
from functools import wraps

import cv2 as cv
import nibabel
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from PIL import Image
from skimage.color import gray2rgb, rgb2gray

#: landmarks coordinates, loading from CSV file
LANDMARK_COORDS = ['X', 'Y']
# PIL.Image.DecompressionBombError: could be decompression bomb DOS attack.
# SEE: https://gitlab.mister-muffin.de/josch/img2pdf/issues/42
Image.MAX_IMAGE_PIXELS = None


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
            logging.exception(
                'Something went wrong (probably parallel access), the status of "%s" is %s',
                path_folder,
                os.path.isdir(path_folder),
            )
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
    _, ext = os.path.splitext(path_file)
    if ext == '.csv':
        return load_landmarks_csv(path_file)
    if ext == '.pts':
        return load_landmarks_pts(path_file)
    logging.error('not supported landmarks file: %s', os.path.basename(path_file))


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
    points = [[float(n) for n in line.split()] for line in lines[2:] if line]
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
    path_file, _ = os.path.splitext(path_file)
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

    .. seealso:: https://simpleelastix.readthedocs.io/PointBasedRegistration.html

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
    if path_.startswith('~'):
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
    img_diag = np.sqrt(np.sum(np.array([height, width])**2))
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
        if image.ndim == 3 and image.shape[-1] < 3:
            image = image[:, :, 0]
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
def convert_image_to_nifti(path_image, path_out_dir=None):
    """ converting normal image to Nifty Image

    :param str path_image: input image
    :param str path_out_dir: path to output folder
    :return str: resulted image

    >>> path_img = os.path.join(update_path('data-images'), 'images',
    ...                         'artificial_moving-affine.jpg')
    >>> path_img2 = convert_image_to_nifti(path_img, '.')
    >>> path_img2  # doctest: +ELLIPSIS
    '...artificial_moving-affine.nii'
    >>> os.path.isfile(path_img2)
    True
    >>> path_img3 = convert_image_from_nifti(path_img2)
    >>> os.path.isfile(path_img3)
    True
    >>> list(map(os.remove, [path_img2, path_img3]))  # doctest: +ELLIPSIS
    [...]
    """
    path_image = update_path(path_image)
    path_img_out = _gene_out_path(path_image, '.nii', path_out_dir)
    logging.debug('Convert image to Nifti format "%s" ->  "%s"', path_image, path_img_out)

    # img = Image.open(path_file).convert('LA')
    img = load_image(path_image)
    nim = nibabel.Nifti1Pair(img, np.eye(4))
    del img
    nibabel.save(nim, path_img_out)

    return path_img_out


@io_image_decorate
def convert_image_to_nifti_gray(path_image, path_out_dir=None):
    """ converting normal image to Nifty Image

    :param str path_image: input image
    :param str path_out_dir: path to output folder
    :return str: resulted image

    >>> path_img = './sample-image.png'
    >>> save_image(path_img, np.zeros((100, 200, 3)))
    >>> path_img2 = convert_image_to_nifti_gray(path_img)
    >>> os.path.isfile(path_img2)
    True
    >>> path_img3 = convert_image_from_nifti(path_img2, '.')
    >>> os.path.isfile(path_img3)
    True
    >>> list(map(os.remove, [path_img, path_img2, path_img3]))  # doctest: +ELLIPSIS
    [...]
    """
    path_image = update_path(path_image)
    path_img_out = _gene_out_path(path_image, '.nii', path_out_dir)
    logging.debug('Convert image to Nifti format "%s" ->  "%s"', path_image, path_img_out)

    # img = Image.open(path_file).convert('LA')
    img = rgb2gray(load_image(path_image))
    nim = nibabel.Nifti1Pair(np.swapaxes(img, 1, 0), np.eye(4))
    del img
    nibabel.save(nim, path_img_out)

    return path_img_out


def _gene_out_path(path_file, file_ext, path_out_dir=None):
    """ generate new path with the same file name just changed extension (and folder)

    :param str path_file: path of source file (image)
    :param str file_ext: file extension of desired file
    :param str path_out_dir: destination folder, if NOne use the input dir
    :return str: desired file
    """
    if not path_out_dir:
        path_out_dir = os.path.dirname(path_file)
    img_name, _ = os.path.splitext(os.path.basename(path_file))
    path_out = os.path.join(path_out_dir, img_name + file_ext)
    return path_out


@io_image_decorate
def convert_image_from_nifti(path_image, path_out_dir=None):
    """ converting Nifti to standard image

    :param str path_image: path to input image
    :param str path_out_dir: destination directory, if Nne use the same folder
    :return str: path to new image
    """
    path_image = update_path(path_image)
    path_img_out = _gene_out_path(path_image, '.jpg', path_out_dir)
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

    .. seealso:: https://www.programcreek.com/python/example/96382/SimpleITK.WriteImage

    :param str path_image: path to the input image
    :param str path_out_dir: path to output directory, if None use the input dir
    :param str img_ext: image extension like PNG or JPEG
    :param float|None scaling: image down-scaling,
        resulting image will be larger by this factor
    :return str: path to exported image

    >>> path_img = os.path.join(update_path('data-images'), 'images',
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
    path_image = _gene_out_path(path_image, img_ext, path_out_dir)
    save_image(path_image, img)
    return path_image


@io_image_decorate
def convert_image_to_mhd(path_image, path_out_dir=None, to_gray=True, overwrite=True, scaling=None):
    """ converting standard image to MHD (Nifty format)

    .. seealso:: https://stackoverflow.com/questions/37290631

    :param str path_image: path to the input image
    :param str path_out_dir: path to output directory, if None use the input dir
    :param bool overwrite: allow overwrite existing image
    :param float|None scaling: image up-scaling
        resulting image will be smaller by this factor
    :return str: path to exported image

    >>> path_img = os.path.join(update_path('data-images'), 'images',
    ...                         'artificial_moving-affine.jpg')
    >>> convert_image_to_mhd(path_img, scaling=2)  # doctest: +ELLIPSIS
    '...artificial_moving-affine.mhd'
    """
    path_image = update_path(path_image)
    path_image_new = _gene_out_path(path_image, '.mhd', path_out_dir)
    # in case the image exists and you are not allowed to overwrite it
    if os.path.isfile(path_image_new) and not overwrite:
        logging.debug('skip converting since the image exists and no-overwrite: %s', path_image_new)
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


def load_config_args(path_config, comment='#'):
    """load config arguments from file with dropping comments

    :param str path_config: configuration file
    :param str comment: character defining comments
    :return str: concat arguments

    >>> p_conf = './sample-arg-config.txt'
    >>> with open(p_conf, 'w') as fp:
    ...     fp.writelines(os.linesep.join(['# comment', '', ' -a 1  ', ' --b c#d']))
    >>> load_config_args(p_conf)
    '-a 1 --b c'
    >>> os.remove(p_conf)
    """
    assert os.path.isfile(path_config), 'missing file: %s' % path_config
    lines = []
    with open(path_config, 'r') as fp:
        for ln in fp.readlines():
            # drop comments
            if comment in ln:
                ln = ln[:ln.index(comment)]
            # remove spaces from beinning and end
            ln = ln.strip()
            # skip empty lines
            if ln:
                lines.append(ln)
    config = ' '.join(lines)
    return config


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
