from __future__ import print_function, division

from six.moves import zip
import numpy as np
import cv2


# don't use OpenCL
# prevents spawning GPU processes that hog memory
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass
# run single-threaded; only warpAffine benefits and scales poorly
cv2.setNumThreads(1)


# noinspection PyUnresolvedReferences
def _make_contrast_luts(n):
    pos = 6.279
    neg = 3.348e-03
    positive_lut = 1 / (1 + np.exp(np.linspace(pos, -pos, n)))
    positive_lut -= positive_lut.min()
    positive_lut /= positive_lut.max()
    negative_lut = 2 * np.arctanh(2 * np.linspace(neg, 1 - neg, n) - 1)
    negative_lut -= negative_lut.min()
    negative_lut /= negative_lut.max()
    return positive_lut, negative_lut


DTYPE_INFO = {
    np.uint8: (
        np.int16,
        -np.iinfo(np.uint8).max-1,
        np.iinfo(np.uint8).max,
    ),
    np.uint16: (
        np.int32,
        -np.iinfo(np.uint16).max-1,
        np.iinfo(np.uint16).max,
    ),
    np.int16: (
        np.int16,
        0,
        2**15-1,
    )
}


DTYPE_LUTS = {
    np.uint8: (
        np.linspace(0, 1, 2 ** 8),
        *_make_contrast_luts(2 ** 8),
    ),
    np.uint16: (
        np.linspace(0, 1, 2 ** 16),
        *_make_contrast_luts(2 ** 16),
    ),
    np.int16: (
        np.linspace(0, 1, 2 ** 15),
        *_make_contrast_luts(2 ** 15),
    )
}


def add_gamma(im, color, gamma_gray, gamma_color, contrast,
              _base_lut=None,
              _positive_contrast_lut=None,
              _negative_contrast_lut=None):
    """
    A Function that takes a numpy array that contains an Image and information about the desired gamma
    values and takes those gamma values to apply gamma correction to the images.

    :param im: the numpy array that contains the Image data
    :param color: flag that indicates if gamma_color should be used
    :param gamma_gray: gray parameter of the gamma correction
    :param gamma_color: color parameter of the gamma correction
    :param contrast: contrast parameter of the gamma correction
    :param base_lut: a lookup table that can be precomputed. Defaults to None. None indicates that the default lookup
    table should be used. The default lookup table is computed only once and then cached.
    :param _positive_contrast_lut: similar to base_lut, just for the positive part of the contrast
    :param _negative_contrast_lut: see positive... contrast is treated asymmetrically to give better results
    """
    base_lut, positive_contrast_lut, negative_contrast_lut = DTYPE_LUTS[im.dtype.type]
    _, _, maxv = DTYPE_INFO[im.dtype.type]
    _base_lut = _base_lut or base_lut
    _positive_contrast_lut = _positive_contrast_lut or positive_contrast_lut
    _negative_contrast_lut = _negative_contrast_lut or negative_contrast_lut

    # check inputs
    if not gamma_gray:
        gamma_gray = 1
    if not color or not gamma_color:
        gamma_color = (1,)
    if contrast is None:
        contrast = 0

    # make sure number of color gammas and channels match
    nchannels = im.shape[2]
    ngammas = len(gamma_color)
    if nchannels != ngammas:
        if ngammas == 1:
            ngammas *= nchannels
            gamma_color = gamma_color * nchannels
        elif nchannels == 1:
            im = im.repeat(ngammas, 2)
        else:
            raise ValueError(
                'cannot broadcast %d channels and %d gamma_color values'
                % (nchannels, ngammas)
            )

    # transform gamma
    gamma_color = [g*gamma_gray for g in gamma_color]
    if contrast >= 0:
        base_lut = (1-contrast)*_base_lut + contrast*_positive_contrast_lut
    else:
        base_lut = (1+contrast)*_base_lut - contrast*_negative_contrast_lut
    for i, gamma in enumerate(gamma_color):
        lut = (base_lut ** gamma * maxv).astype(im.dtype)
        im[:, :, i] = lut.take(im[:, :, i])
    if len(im.shape) != 3:
        im = im[:, :, np.newaxis]
    return im


def add_noise_rgb(im, strength):
    """
    A Function that takes a numpy array that contains an Image and information about the desired rgb noise
    and takes those values to add noise to the images. This function adds rgb noise, that mimics the noise of a
    camera sensor, what means that green has less noise.

    :param im: the numpy array that contains the Image data
    :param strength: strength of the noise

    """
    h, w, c = im.shape
    size = int(h//2), int(w//2)
    target_size = h, w
    if c == 1:
        sizes = target_size,
    else:
        sizes = size, target_size, size
    target_sizes = (target_size,) * c
    strengths = [strength*r for r in [1.0, 0.5, 1.0]]
    cs = range(c)
    noise_type, minv, maxv = DTYPE_INFO[im.dtype.type]
    for i, size, target_size, s in zip(cs, sizes, target_sizes, strengths):
        noisy = np.random.randint(
            int(minv*s), int(maxv*s), size, dtype=noise_type
        )
        if size != target_size:
            # cv2.resize does not support int32
            if noise_type is np.int32:
                noisy = noisy.astype(np.float32)
            noisy = cv2.resize(noisy, target_size[::-1],
                               interpolation=cv2.INTER_LINEAR)
        noisy += im[:, :, i]
        np.clip(noisy, 0, maxv, im[:, :, i])
    if len(im.shape) != 3:
        im = im[:, :, np.newaxis]
    return im


def add_noise_other(im, strength):
    """
    A Function that takes a numpy array that contains an Image and information about the desired noise
    and takes those values to add noise to the images.

    :param im: the numpy array that contains the Image data
    :param strength: strength of the noise

    """
    noise_type, minv, maxv = DTYPE_INFO[im.dtype.type]
    noisy = np.random.randint(
        int(minv*strength), int(maxv*strength), im.shape, dtype=noise_type
    )
    noisy += im
    np.clip(noisy, 0, maxv, im)
    return im


def add_blur(im, sigma):
    """
    A Function that takes a numpy array that contains an Image and information about the desired blur
    and blurs the image. It uses cv2 to blur the image, for more information about the sigma parameter have a look into
    the cv2 documentation. cv.GaussianBlur

    :param im: the numpy array that contains the Image data
    :param sigma: the sigma of the gaussian blur

    """
    # sigma is relative to image width
    sigma *= im.shape[1]
    im = cv2.GaussianBlur(im, (0, 0), sigma)
    if len(im.shape) != 3:
        im = im[:, :, np.newaxis]
    return im
