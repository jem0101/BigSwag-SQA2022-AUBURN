from abc import abstractmethod
import random

import cv2

# noinspection PyUnresolvedReferences
INTERP_NEAREST = cv2.INTER_NEAREST
# noinspection PyUnresolvedReferences
INTERP_LINEAR = cv2.INTER_LINEAR
# noinspection PyUnresolvedReferences
INTERP_CUBIC = cv2.INTER_CUBIC
# noinspection PyUnresolvedReferences
INTERP_LANCZOS4 = cv2.INTER_LANCZOS4
# noinspection PyUnresolvedReferences
INTERP_AERA = cv2.INTER_AREA
INTERPOLATIONS = (
    INTERP_NEAREST,
    INTERP_LINEAR,
    # INTERP_CUBIC,
    # INTERP_LANCZOS4
)
MAX_SUPERSAMPLING = 3


def human2gamma(human):
    """
    Convert some human-understandable value
    to a value that is suitable for use in gamma correction.

    Values outside of `[-1,1]` are clipped.

    :param human:
        value to convert to gamma
    :return:
        value suitable for gamma correction
    """
    if human >= 0.999:
        return 0.00099999999999988987
    elif human <= -0.999:
        return 999.99999999999909
    gamma = human + 1
    if gamma > 1:
        gamma = 2 - gamma
    else:
        gamma = 1 / gamma
    return gamma


class RNG(object):
    """
    Abstract base class for augmentation random number generators (RNG).
    For further information about the semantics of individual parameters,
    have a look at :func:`~crumpets.augmentation.randomize_image`.
    """

    @abstractmethod
    def __call__(self, image, buffer):
        """
        Generate random augmentation values.

        :param image:
            Image that will be augmented.
        :param buffer:
            Target buffer for the augmented image.
        :return:
            Dict with randomly generated augmentation values.
            For instance:

            .. code-block:: python

                {
                  'angle': 10.7650,
                  'scale': 1.1139,
                  'aspect': 1.0063,
                  'shear': (0.0043, -0.0091),
                  'shift': (0.9163, 0.1045),
                  'gamma_gray': 1.2463,
                  'gamma_color': [0.9335, 1.05214, 0.9256],
                  'contrast': 0.0597,
                  'noise': 0.0140,
                  'blur': 0,
                  'interp_method': 1,
                  'hmirror': False,
                  'vmirror': False
                }
        """
        pass


class NoRNG(RNG):
    """
    Decidedly non-random RNG that simply returns an empty dict.
    Useful for validation runs.
    """

    def __call__(self, image, buffer):
        return {}


class MixtureRNG(RNG):
    """
    Old crumpets 2-style RNG that uses a mix of base probability,
    Gaussian and uniform distributions to generate parameters.
    See :func:`~crumpets.augmentation.randomize_image` for more details
    on allowed values.

    ..note:
        `*_range` parameters must be 2 numbers `(a,b)`.
        They define a uniform distribution `U[a,b]`

    ..note:
        `*_sigma` parameters define the standard deviation
        of a Gaussian distribution.

    :param prob:
        probability that pixel-based augmentations are applied;
        each augmentation rolls separately;
        does not apply to spatial transforms like scale, rotation, etc.
    :param scale_range:
        influences the `'scale'` value
    :param shift_range:
        influences the `'shift'` values
    :param noise_range:
        influences the `'noise'` value
    :param brightness_range:
        influences the `'gamma_gray'` value;
        brightness is converted to gamma by
        :func:`~crumpets.rng.convert_gamma`
    :param color_range:
        influences the `'gamma_color'` values;
        color values are converted to gamma by
        :func:`~crumpets.rng.convert_gamma`
    :param contrast_range:
        influences the `'contrast'` value
    :param blur_range:
        influences the `'blur'` value;
        effective sigma for Gaussian blur is `blur*image_width`
    :param rotation_sigma:
        influences the `'angle'` value
    :param aspect_sigma:
        influences the `'aspect'` value
    :param interpolations:
        list of possible interpolation methods to use;
        influences the `'interp_method'` value
    :param hmirror:
        probability that horizontal mirror is applied;
        influences the `'hmirror'` value
    :param vmirror:
        probability that vertical mirror is applied;
        influences the `'vmirror'` value
    :param shear_range:
        influences the `'shear'` values
    """
    def __init__(
            self,
            prob=1,
            scale_range=None,
            shift_range=None,
            noise_range=None,
            brightness_range=None,
            color_range=None,
            contrast_range=None,
            blur_range=None,
            rotation_sigma=0,
            aspect_sigma=0,
            interpolations=None,
            hmirror=0,
            vmirror=0,
            shear_range=None,
    ):
        self.prob = prob
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.noise_range = noise_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        self.contrast_range = contrast_range
        self.blur_range = blur_range
        self.rotation_sigma = rotation_sigma
        self.aspect_sigma = aspect_sigma
        self.interpolations = interpolations
        self.hmirror = hmirror
        self.vmirror = vmirror
        self.shear_range = shear_range

    def __call__(
            self,
            image,
            buffer,
    ):
        prob = self.prob
        if not prob:
            return {}
        kwargs = dict(
            angle=0,
            scale=1,
            aspect=1,
            shear=None,
            shift=None,
            gamma_gray=None,
            gamma_color=None,
            contrast=0,
            noise=0,
            blur=0,
            interp_method=0,
            hmirror=random.random() < self.hmirror,
            vmirror=random.random() < self.vmirror,
        )
        if self.rotation_sigma:
            kwargs['angle'] = random.gauss(0, self.rotation_sigma)
        if self.scale_range is not None:
            kwargs['scale'] = random.uniform(*self.scale_range)
        if self.shear_range is not None:
            kwargs['shear'] = tuple(random.uniform(*self.shear_range) for _ in range(2))
        if self.shift_range is not None:
            kwargs['shift'] = tuple(random.uniform(*self.shift_range) for _ in range(2))
        if self.aspect_sigma:
            kwargs['aspect'] = random.gauss(1, self.aspect_sigma)
        if self.brightness_range is not None and random.random() < prob:
            kwargs['gamma_gray'] = human2gamma(random.uniform(*self.brightness_range))
        if self.color_range is not None and random.random() < prob:
            kwargs['gamma_color'] = [human2gamma(random.uniform(*self.color_range))
                                     for _ in range(3)]
        if self.contrast_range is not None and random.random() < prob:
            kwargs['contrast'] = random.uniform(*self.contrast_range)
        if self.noise_range is not None and random.random() < prob:
            kwargs['noise'] = random.uniform(*self.noise_range)
        if self.blur_range is not None and random.random() < prob:
            kwargs['blur'] = random.uniform(*self.blur_range)
        if self.interpolations:
            kwargs['interp_method'] = random.choice(self.interpolations)

        return kwargs
