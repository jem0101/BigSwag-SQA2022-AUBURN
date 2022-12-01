from copy import copy
import random
import warnings

import torch.nn as nn

from . import augmentation_cuda as cudaaugs


class Randomizer(nn.Module):
    """
    Given a network (or in general, some pytorch module), it is wrapped around the nets forward pass.
    If the randomizer's forward function is invoked, it first randomizes the image in the sample dictionary.
    That means it basically works like :func:`~crumpets.augmentation.randomize_image`,
    which is usually applied to the image in one of the workers.
    The major difference here is that all augmentations are gpu powered, and thus faster.
    Also not all augmentation operations are supported. The randomizer does not rotate or resize.
    The values used for augmenting are picked out of the dictionary.
    Therefore the sample dictionary must contain these. Usually crumpets worker take care of that.

    :param net: some network the randomizer shall be wrapped around
    """

    def __init__(self, net=None):
        super(Randomizer, self).__init__()
        self.net = net
        self.use_cuda = False

    def forward(self, sample, *args, **kwargs):
        """
        Applies different randomizing augmentations to input images and then
        forwards result through net, if given.

        :param sample:
            dictonary with
            {"image": Tensor of shape n,c,h,w,
             "augmentation": list of augmentation parameters per image in batch}
        :return:
            modified dictionary with randomized image and network modified entries
        """
        if self.net is not None:  # asssumes ParallelApply is used
            if 'gpu_augmentation' in sample['augmentation'][0] \
                    and not sample['augmentation'][0]['gpu_augmentation']:
                warnings.warn('gpu_augmentation for randomization is not activated, '
                              'but Randomizer Module is used! '
                              'Directly forwarding to net now.')
                return self.net.forward(sample) if self.net is not None else sample
        if 'gpu_augmentation' in sample['augmentation'][0] \
                and not sample['augmentation'][0]['gpu_augmentation']:
            warnings.warn('gpu_augmentation for randomization is not activated, '
                          'but Randomizer Module is used! '
                          'Directly forwarding to net now.')
            return self.net.forward(sample)

        im = sample['image']

        if len(im.shape) != 4:
            raise AttributeError(
                'image shape length {} != 4, but 4 is required'.format(im.shape)
            )

        # randomize the order of operations
        order = list(range(3))
        random.shuffle(order)

        for op in order:
            if op == 0:
                im = cudaaugs.add_gamma(im, sample['augmentation'])

            if op == 1:
                if im.shape[1] > 1:
                    im = cudaaugs.add_noise_rgb(im, sample['augmentation'])
                else:
                    im = cudaaugs.add_noise_other(im, sample['augmentation'])
            if op == 2:
                im = cudaaugs.add_blur(im, sample['augmentation'])
        result = copy(sample)
        result['image'] = im

        # forward through net
        return self.net.forward(result) if self.net is not None else result

    def cuda(self, device_id=None):
        super(Randomizer, self).cuda(device_id)
        self.use_cuda = True
        return self

    def cpu(self):
        super(Randomizer, self).cpu()
        self.use_cuda = False
        return self
