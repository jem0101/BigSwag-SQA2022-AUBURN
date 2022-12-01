from __future__ import print_function, division

from .. import ImageWorker
from .. import FCNWorker
from ...augmentation import decode_image
from ...rng import INTERP_NEAREST


class SegmentationWorker(FCNWorker):
    """
    Worker for image segmentation tasks.
    `target_image_params` defaults nearest neighbor interpolation,
    no supersampling, and to disable all pixel-based
    augmentations like brightness and color.
    """
    def __init__(self, image, target_image,
                 image_params=None, target_image_params=None,
                 image_rng=None,
                 **kwargs):
        if target_image_params is None:
            target_image_params = dict(image_params)
            target_image_params.update(
                color=False,
                background=None,
                interp_method=INTERP_NEAREST,
                gamma_gray=None,
                gamma_color=None,
                contrast=None,
                noise=None,
                blur=None,
                shear=None,
                supersampling=1,
            )
        FCNWorker.__init__(
            self, image, target_image,
            image_params, target_image_params,
            image_rng, **kwargs
        )

    def prepare(self, sample, batch, buffers):
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        target_im = decode_image(sample['target_image'], False)
        self.prepare_image(target_im, buffers, params, 'target_image')
        return im, params, target_im
