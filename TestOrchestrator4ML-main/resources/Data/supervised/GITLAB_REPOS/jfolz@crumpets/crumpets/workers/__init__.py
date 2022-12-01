from __future__ import print_function, division

import cv2

from ..broker import BufferWorker
from ..augmentation import decode_image
from ..presets import NO_AUGMENTATION
from ..augmentation import randomize_image


__all__ = [
    'ImageWorker',
    'ClassificationWorker',
    'FCNWorker'
]


def noop(im):
    return im


# noinspection PyUnresolvedReferences
def make_cvt(code):
    return lambda im: cv2.cvtColor(im, code)


# noinspection PyUnresolvedReferences
COLOR_CONVERSIONS = {
    None: noop,
    False: noop,
    '': noop,
    'rgb': noop,
    'RGB': noop,
    'hsv': make_cvt(cv2.COLOR_RGB2HSV_FULL),
    'HSV': make_cvt(cv2.COLOR_RGB2HSV_FULL),
    'hls': make_cvt(cv2.COLOR_RGB2HLS_FULL),
    'HLS': make_cvt(cv2.COLOR_RGB2HLS_FULL),
    'lab': make_cvt(cv2.COLOR_RGB2LAB),
    'LAB': make_cvt(cv2.COLOR_RGB2LAB),
    'ycrcb': make_cvt(cv2.COLOR_RGB2YCrCb),
    'YCrCb': make_cvt(cv2.COLOR_RGB2YCrCb),
    'YCRCB': make_cvt(cv2.COLOR_RGB2YCrCb),
    'gray': make_cvt(cv2.COLOR_RGB2GRAY),
    'GRAY': make_cvt(cv2.COLOR_RGB2GRAY),
}


def hwc2chw(im):
    return im.transpose((2, 0, 1))


def chw2hwc(im):
    return im.transpose((1, 2, 0))


def flat(array):
    return tuple(array.flatten().tolist())


class ImageWorker(BufferWorker):
    """
    Worker for processing images of any kind.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    :param gpu_augmentation:
        disables augmentations for which
        gpu versions are available (:class:`~crumpets.torch.randomizer`)
    """
    def __init__(self, image,
                 image_params=None,
                 image_rng=None,
                 **kwargs):
        BufferWorker.__init__(self, **kwargs)
        self.add_buffer('image', image)
        self.add_params('image', image_params, {})
        self.image_rng = image_rng or NO_AUGMENTATION

    def prepare_image(self, im, buffers, params, key):
        params = dict(params)
        params.update(self.params[key])
        cvt = COLOR_CONVERSIONS[params.pop('colorspace', None)]
        buffers[key][...] = hwc2chw(cvt(randomize_image(
            im, buffers[key].shape[1:],
            background=flat(self.fill_values[key]),
            **params
        )))
        return params

    def prepare(self, sample, batch, buffers):
        im = decode_image(sample['image'],
                          self.params['image'].get('color', True))
        params = self.image_rng(im, buffers['image'])
        params['gpu_augmentation'] = self.gpu_augmentation
        image_params = self.prepare_image(im, buffers, params, 'image')
        batch['augmentation'].append(image_params)
        return im, params


class ClassificationWorker(ImageWorker):
    """
    Worker for processing (Image, Label)-pairs for classification.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param label:
        tuple of label information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    """
    def __init__(self, image, label,
                 image_params=None,
                 image_rng=None,
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image_params,
                             image_rng,
                             **kwargs)
        self.add_buffer('label', label)

    def prepare(self, sample, batch, buffers):
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        buffers['label'][...] = sample['label']
        return im, params


class FCNWorker(ImageWorker):
    """
    Worker for fully convolutional networks (FCN).
    Produces `image`-`target_image`-pairs.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param target_image:
        tuple of target image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param target_image_params:
        dict of fixed target image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    """
    def __init__(self, image, target_image,
                 image_params=None, target_image_params=None,
                 image_rng=None,
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image_params,
                             image_rng,
                             **kwargs)
        self.add_buffer('target_image', target_image)
        self.add_params('target_image', target_image_params, {})

    def prepare(self, sample, batch, buffers):
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        self.prepare_image(im, buffers, params, 'target_image')
