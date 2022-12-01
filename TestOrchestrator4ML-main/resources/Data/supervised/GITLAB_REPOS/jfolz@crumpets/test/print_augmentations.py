"""
Given an example image, this program creates and stores augmented versions for
for it, for each available augmentation. It can either use the default
ranges from which random intensity values are chosen or accept custom ones.
"""
from __future__ import print_function, division

import io
import os
import os.path as pt
import sys

import cv2
import msgpack
import msgpack_numpy
import numpy as np
from itertools import cycle

from crumpets.presets import NO_AUGMENTATION
from crumpets.torch.dataloader import TorchTurboDataLoader as TurboDataLoader
from crumpets.workers import ImageWorker

ROOT = pt.dirname(__file__)
sys.path.insert(0, pt.join(ROOT, '..'))


def _show(im, title='Random', wait=3000):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, im)
    cv2.waitKey(wait)


def main(source, cuda=True, target='result', show=3000, save=True):
    tests = [
        ('no_augmentation', NO_AUGMENTATION),
        ('scale_in', (lambda x: x.update(dict(prob=1.0, scale_range=(1.4, 1.4))) or x)(dict(NO_AUGMENTATION))),
        ('scale_out', (lambda x: x.update(dict(prob=1.0, scale_range=(0.6, 0.6))) or x)(dict(NO_AUGMENTATION))),
        ('shift_up', (lambda x: x.update(dict(prob=1.0, shift_range=(1, 1))) or x)(dict(NO_AUGMENTATION))),
        ('shift_down', (lambda x: x.update(dict(prob=1.0, shift_range=(-1, -1))) or x)(dict(NO_AUGMENTATION))),
        ('noise', (lambda x: x.update(dict(prob=1.0, noise_range=(0.2, 0.2))) or x)(dict(NO_AUGMENTATION))),
        ('brightness', (lambda x: x.update(dict(prob=1.0, brightness_range=(0.45, 0.45))) or x)(dict(NO_AUGMENTATION))),
        ('color', (lambda x: x.update(dict(prob=1.0, color_range=(-0.3, 0.3))) or x)(dict(NO_AUGMENTATION))),
        ('contrast', (lambda x: x.update(dict(prob=1.0, contrast_range=(0.35, 0.35))) or x)(dict(NO_AUGMENTATION))),
        ('blur', (lambda x: x.update(dict(prob=1.0, blur_range=(1.50/448, 1.50/448))) or x)(dict(NO_AUGMENTATION))),
        ('rotation', (lambda x: x.update(dict(prob=1.0, rotation_sigma=24)) or x)(dict(NO_AUGMENTATION))),
        ('aspect', (lambda x: x.update(dict(prob=1.0, aspect_sigma=0.3)) or x)(dict(NO_AUGMENTATION))),
        ('hmirror', (lambda x: x.update(dict(prob=1.0, hmirror=1)) or x)(dict(NO_AUGMENTATION))),
        ('vmirror', (lambda x: x.update(dict(prob=1.0, vmirror=1)) or x)(dict(NO_AUGMENTATION))),
        ('shear', (lambda x: x.update(dict(prob=1.0,  shear_range=(0.06, 0.06))) or x)(dict(NO_AUGMENTATION))),
        ('default_augmentation', {}),
    ]

    with io.FileIO(source) as f:
        img = {'image': f.read()}
    img = [msgpack.packb(img, use_bin_type=True, default=msgpack_numpy.encode)]
    cycler = cycle(img)

    for name, randomizer_params in tests:
        print(name)

        worker = ImageWorker(
            ((3, 256, 256), np.uint8, (128, 128, 128)),
            dict(scale_mode='longest'),
            randomizer_params,
            gpu_augmentation=cuda,
        )
        loader = TurboDataLoader(
            cycler, 1, worker, 1, length=1, use_cuda=cuda, gpu_augmentation=cuda
        )

        with loader:
            for iteration, mini_batch in loader:
                sample = next(mini_batch)[0][0] if cuda else next(mini_batch)
                image = sample['image']
                result = image.squeeze().transpose(0, 2).transpose(0, 1).cpu().numpy()
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                if show:
                    _show(result, wait=show)
                if save:
                    cv2.imwrite(
                        pt.join(target, "{}.jpg").format(name),
                        result,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--target', '-t',
        help='directory where resulting augmented images are stored',
        default='result'
    )
    parser.add_argument(
        '--source', '-s',
        help='path to example image, defaults to example.jpg in doc/sphinx/source',
        default=pt.join(ROOT, '..', 'doc', 'sphinx', 'source', 'example.jpg')
    )
    parser.add_argument(
        '--cpu',
        dest='cuda',
        action='store_false',
    )
    parser.add_argument(
        '--show',
        type=int,
        default=0,
        help='milliseconds the images are shown beside being stored'
    )
    parser.add_argument(
        '--no-save',
        dest='save',
        action='store_false'
    )

    parser.set_defaults(
        cuda=True, save=True
    )
    args, unknown = parser.parse_known_args()

    if not pt.exists(args.target):
        os.makedirs(args.target)
    main(**vars(args))
