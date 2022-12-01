from .rng import INTERP_LINEAR
from .rng import NoRNG
from .rng import MixtureRNG


IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
SCALE = 256 / 224
AUGMENTATION_TRAIN = MixtureRNG(
    prob=0.5,
    scale_range=(0.5 * SCALE, 2 * SCALE),
    shift_range=(-1, 1),
    # noise_range=(0.03, 0.1),
    noise_range=None,
    brightness_range=(-0.5, 0.5),
    color_range=(-0.5, 0.5),
    contrast_range=(-1, 1),
    # blur_range=(0.01, 0.75 / 224),
    blur_range=None,
    rotation_sigma=15,
    aspect_sigma=0.1,
    interpolations=(INTERP_LINEAR,),
    hmirror=0.5,
    vmirror=0,
    shear_range=(-0.1, 0.1),
)
AUGMENTATION_ALL = MixtureRNG(
    prob=1,
    scale_range=(0.5 * SCALE, 2 * SCALE),
    shift_range=(-1, 1),
    noise_range=(0.03, 0.1),
    brightness_range=(-0.5, 0.5),
    color_range=(-0.5, 0.5),
    contrast_range=(-1, 1),
    blur_range=(0.01, 0.75 / 224),
    rotation_sigma=15,
    aspect_sigma=0.1,
    interpolations=(INTERP_LINEAR,),
    hmirror=0.5,
    vmirror=0,
    shear_range=(-0.1, 0.1),
)
INPUT_PARAMS_TRAIN = dict()
AUGMENTATION_VAL = NoRNG()
INPUT_PARAMS_VAL = dict(
    scale=SCALE,
    interp_method=INTERP_LINEAR,
)
NO_AUGMENTATION = NoRNG()
