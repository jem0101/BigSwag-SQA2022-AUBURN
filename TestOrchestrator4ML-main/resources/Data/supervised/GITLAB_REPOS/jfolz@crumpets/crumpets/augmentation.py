from __future__ import print_function, division

import random
from math import ceil
import warnings

import numpy as np
import cv2
from simplejpeg import is_jpeg
from simplejpeg import decode_jpeg

from .rng import INTERP_LINEAR
from .rng import INTERP_AERA
from .rng import MAX_SUPERSAMPLING
from . import augmentation_cpu as cpuaugs


try:
    # noinspection PyUnresolvedReferences
    IMREAD_COLOR = cv2.IMREAD_COLOR
    # noinspection PyUnresolvedReferences
    IMREAD_GRAYSCALE = cv2.IMREAD_GRAYSCALE
except AttributeError:
    # noinspection PyUnresolvedReferences
    IMREAD_COLOR = cv2.CV_LOAD_IMAGE_COLOR
    # noinspection PyUnresolvedReferences
    IMREAD_GRAYSCALE = cv2.CV_LOAD_IMAGE_GRAYSCALE


def decode_opencv(data, color):
    a = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(a, IMREAD_COLOR if color else -1)
    if im is None:
        raise ValueError('OpenCV could not decode image')
    if color:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    return im


# don't use OpenCL
# prevents spawning GPU processes that hog memory
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass
# run single-threaded; only warpAffine benefits and scales poorly
cv2.setNumThreads(1)


def decode_image(data, color, min_height=0, min_width=0, min_factor=2):
    if data is None:
        return None
    try:
        # fast path for JPEG
        if is_jpeg(data):
            return decode_jpeg(data, 'rgb' if color else 'gray',
                               True, True, min_height, min_width, min_factor)
        # scrub-tier images
        else:
            return decode_opencv(data, color)
    except (ValueError, RuntimeError) as e:
        warnings.warn(RuntimeWarning('simplejpeg failed to decode image, '
                                     + 'falling back to OpenCV: '
                                     + repr(e)))
        return decode_opencv(data, color)


def calc_scale_ratio(source_size, target_size, scale, scale_mode):
    sh, sw = source_size
    sh /= 2
    sw /= 2
    th, tw = target_size
    th /= 2
    tw /= 2
    if scale_mode == 'longest':
        r = max(sh / th, sw / tw) / scale
    elif scale_mode == 'shortest':
        r = min(sh / th, sw / tw) / scale
    else:
        raise ValueError('unknown scale mode %r' % scale_mode)
    return r, sh, sw, th, tw


def make_transform(
        source_size,
        target_size,
        angle=0,
        scale=1,
        aspect=1,
        shift=None,
        hmirror=False,
        vmirror=False,
        shear=None,
        scale_mode='shortest',
        __identity__=np.eye(3)
):
    r, sh, sw, th, tw = calc_scale_ratio(source_size, target_size,
                                         scale, scale_mode)
    # first shift target to origin
    p = __identity__.copy()
    p[(0, 1), 2] = -tw, -th
    # shear
    if shear:
        q = __identity__.copy()
        q[(1, 0), (0, 1)] = shear
        p = np.dot(q, p)
    # resize and mirror
    q = __identity__.copy()
    q[0, 0] = r*aspect * (-1 if hmirror else 1)
    q[1, 1] = r/aspect * (-1 if vmirror else 1)
    p = np.dot(q, p)
    # then rotate
    if angle:
        q = __identity__.copy()
        q[:2] = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        p = np.dot(q, p)
    # finally shift to desired position in source image
    wdelta = max(sw - tw*r, tw*r - sw)
    hdelta = max(sh - th*r, th*r - sh)
    shift = shift or (0, 0)
    q = __identity__.copy()
    q[(0, 1), 2] = sw + wdelta * shift[1], sh + hdelta * shift[0]
    p = np.dot(q, p)
    return p[:2]


def rotate_and_resize(
        im,
        angle,
        target_size,
        scale,
        aspect,
        shift,
        method,
        background,
        hmirror,
        vmirror,
        shear=None,
        scale_mode='shortest',
        supersampling=0
):
    if background is None:
        background = 0
    if shear is None:
        shear = (0, 0)
    if supersampling == 0:
        r = calc_scale_ratio(im.shape[:2], target_size, scale, scale_mode)[0]
        # noinspection PyTypeChecker
        supersampling = max(1, min(MAX_SUPERSAMPLING, ceil(r)))
    if supersampling > 1:
        th, tw = target_size
        sampling_size = int(round(th * supersampling)), int(round(th * supersampling))
    else:
        sampling_size = target_size
    p = make_transform(
        im.shape[:2], sampling_size,
        angle, scale, aspect, shift, hmirror, vmirror,
        shear, scale_mode,
    )
    im = cv2.warpAffine(
        im, p, sampling_size[::-1],
        borderValue=background,
        flags=method + cv2.WARP_INVERSE_MAP,
    )
    if supersampling > 1:
        im = cv2.resize(im, target_size[::-1], interpolation=INTERP_AERA)
    # OpenCV swallows third dimension if single channel
    if len(im.shape) != 3:
        im = im[:, :, np.newaxis]
    return im


def randomize_image(
        im,
        size,
        background=None,
        color=True,
        angle=0,
        scale=1,
        shift=None,
        aspect=1,
        hmirror=False,
        vmirror=False,
        interp_method=INTERP_LINEAR,
        gamma_gray=None,
        gamma_color=None,
        contrast=None,
        noise=None,
        blur=None,
        shear=None,
        is_rgb=True,
        scale_mode='shortest',
        supersampling=0,
        gpu_augmentation=False,
        do_rotate_and_resize=True
):
    """
    Randomizes image according to given parameters.

    :param im:
        image to be transformed.
    :param size:
        target size of resulting image.
    :param background:
        background color that fills areas in the output
        where there is no pixel data;
        can be number or tuple with same number of elements as channel
    :param color:
        Boolean that flags if image is black-white or colored.
    :param angle:
        degrees of rotation
    :param scale:
        Scales the image with respect to its target size.
        `scale=1.0` scales the image to fit perfectly
        within the target size.
        Based on `scale_mode` either the shorter or longer
        edge is used as reference.
        `scale=2.0` doubles the length of the sides,
        `scale=0.5` halves it.
    :param shift:
        tuple of int (x,y) defining a shift of the picture, may create undefined space, if
        source image is moved out of target image, filled up with background color.
    :param aspect:
        float of aspect ratio change
    :param hmirror:
        boolean flag for horizontal mirror
    :param vmirror:
        boolean flag for vertical mirror
    :param interp_method:
        some interpolation method. At the moment one of:
        INTERP_NEAREST
        INTERP_LINEAR
        INTERP_CUBIC
        INTERP_LANCZOS4
        INTERP_AERA
    :param gamma_gray:
        float defining a black-white gamma
    :param gamma_color:
        tuple of floats defining a rgb gamma
    :param contrast:
        float between -1 and 1 defining a contrast change
    :param noise:
        float defining a noise strength
    :param blur:
        float defining a blur intensity, i.e. the standard deviation of a gaussian filter relative to image width
    :param shear:
        float defining shear intensity, i.e. the gradient of the horizontal edges.
        A shear of 0.0 therefore creates a rectangular image.
    :param is_rgb:
        boolean that flags if rgb color encoding is used
    :param scale_mode:
        Either `'shortest'` or `'longest'`.
        Scale the image using either shortest or longest edge
        as reference.
        `'shortest'` crops part of the image if the aspect ratio
        of image and target size do not match.
        `'longest'` ensures that the whole image can be
        fit into target size.
        A scale > 1.0 makes it bigger than target image, thus parts of it get cut out.
        A scale < 1.0 makes it smaller than target image, thus parts of the target image are undefined and
        filled up with background.
    :param supersampling:
        supersampling factor, 1 turns off supersampling, 2 means 4 samples per pixel,
        3 means 9 samples and so on;
        default of 0 means choose best based on true image size, output size and scale factor
    :param gpu_augmentation:
        boolean that flags if gpu augmentations is used elsewhere and thus disables cpu augmentations in
        this method for all augmentations where gpu versions are available.
    :param do_rotate_and_resize:
        boolean that flags if rotation and resize operations are used. Mostly used for test cases.
        Should usually not be changed.
    :return:
        randomized image
    """
    # decode image, resize sub
    if do_rotate_and_resize:
        im = rotate_and_resize(
            im, angle, size, scale, aspect, shift, interp_method,
            background, hmirror, vmirror, shear,
            scale_mode, supersampling,
        )
    if not gpu_augmentation:
        # randomize the order of operations
        order = []
        if gamma_gray or gamma_color or contrast:
            order.append(0)
        if noise:
            order.append(1)
        if blur is not None and blur > 0.1/448:
            order.append(2)
        random.shuffle(order)
        for op in order:
            if op == 0:
                im = cpuaugs.add_gamma(im, color, gamma_gray, gamma_color, contrast)
            if op == 1:
                if is_rgb:
                    im = cpuaugs.add_noise_rgb(im, noise)
                else:
                    im = cpuaugs.add_noise_other(im, noise)
            if op == 2:
                im = cpuaugs.add_blur(im, blur)
    return im
