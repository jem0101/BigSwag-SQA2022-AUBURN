from __future__ import print_function, division

import random
import itertools as it
from math import ceil

import cv2
import numpy as np
from numpy.random import multivariate_normal

from ...augmentation import make_transform
from ...augmentation import calc_scale_ratio
from .. import ImageWorker


def check_range(points, h, w):
    # remove NaNs
    points = points[np.logical_not(np.isnan(points).any(axis=1))]
    # x/y >= 0
    # noinspection PyUnresolvedReferences
    points = points[(points >= 0).all(1)]
    # x < w, y < h
    ind = np.logical_and(points[:, 0] < w, points[:, 1] < h)
    return points[ind]


def interpolate_points(points, h, w):
    points = check_range(points, h, w)
    lower = np.floor(points).astype(np.int32)
    upper = np.ceil(points).astype(np.int32)
    x = points[:, 0]
    y = points[:, 1]
    xl = lower[:, 0]
    yl = lower[:, 1]
    xu = upper[:, 0]
    yu = upper[:, 1]
    np.clip(xl, 0, w-1, xl)
    np.clip(yl, 0, h-1, yl)
    np.clip(xu, 0, w-1, xu)
    np.clip(yu, 0, h-1, yu)
    xlyl = (1 - (x - xl)) * (1 - (y - yl))
    xlyu = (1 - (x - xl)) * (1 - (yu - y))
    xuyl = (1 - (xu - x)) * (1 - (y - yl))
    xuyu = (1 - (xu - x)) * (1 - (yu - y))
    s = xlyl + xlyu + xuyl + xuyu + 1e-9
    return \
        it.chain(yl, yu, yl, yu), \
        it.chain(xl, xl, xu, xu), \
        it.chain(xlyl / s, xlyu / s, xuyl / s, xuyu / s)


def discretize_points(points, h, w):
    return check_range(points.round(), h, w).astype(np.int32)


def show(name, mat):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 600)
    cv2.imshow(name, mat)
    cv2.waitKey()


class SaliencyWorker(ImageWorker):
    """
    Worker that outputs images and saliency maps created from raw
    gaze locations.
    Expects the following keys present in each sample:

    {"image": encoded image data
     "experiments": [experiment, ...]}

    Each experiment is first checked for fixations points under key
    "fixations". Falls back to key "locations" of raw gaze data
    if no fixations are found.

    The following parameters can be configured:

    - image_params: see ImageWorker
    - target_image_params:
        - "sample_ratio" (default: 1):
            float in [0, 1]; percentage of experiments
            sampled from the list of all experiments
        - "jitter" (default: 0):
            add noise to the individual gaze locations;
            sigma of a Gaussian distribution,
            scaled by the size of the target_images:
            noise ~ N(jitter * target_image_size)
        - "interpolate" (default: False):
            use linear interpolation to map gaze locations
            to the target_image
        - "blur" (default: 0):
            apply Gaussian blur with sigma blur * target_image_size
            to target_image
        - "maxnorm" (default: False):
            apply maximum norm to target_image
    """
    def __init__(self, image, target_image,
                 image_params=None,
                 target_image_params=None,
                 image_rng=None,
                 **kwargs):
        ImageWorker.__init__(self, image, image_params, image_rng, **kwargs)
        self.add_buffer('target_image', target_image)
        self.add_params('target_image', target_image_params)

    def prepare(self, sample, batch, buffers):
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        ih, iw = im.shape[:2]

        map_params = dict(params)
        map_params.update(self.params['image'])
        map_params.update(self.params['target_image'])
        target = buffers['target_image']
        r = calc_scale_ratio(
            (ih, iw), target.shape, map_params.get('scale', 1),
            map_params.get('scale_mode', 'shortest')
        )[0]
        mh, mw = [int(s*ceil(r)) for s in target.shape]
        map_buffer = np.zeros((mh, mw), np.float32)
        # create experiment arrays, using only 1st and 2nd column
        experiments = [
            check_range(np.float32(
                experiment.get('fixations', experiment['locations'])
            ), ih, iw)  # fixations
            for experiment in sample['experiments']
        ]
        # remove (now) empty experiments
        experiments = [
            experiment for experiment in experiments if experiment.size
        ]
        # sample a subset of experiments
        sample_ratio = map_params.get('sample_ratio', 1.0)
        n = int(ceil(len(experiments)*sample_ratio))
        experiments = random.sample(experiments, n)
        # create combined points array
        points = np.concatenate([
            experiment for experiment in experiments if experiment.size
        ])
        if not points.size:
            raise ValueError('no gaze points')

        # transform points to map_buffer coordinates
        m = np.eye(3, dtype=np.float32)
        m[:2] = make_transform(
            (ih, iw), (mh, mw),
            map_params.get('angle', 0),
            map_params.get('scale', 1),
            map_params.get('aspect', 1),
            map_params.get('shift', None),
            map_params.get('hmirror', False),
            map_params.get('vmirror', False),
            map_params.get('shear', None),
            map_params.get('scale_mode', 'shortest')
        )
        points = points.reshape((1, -1, 2))
        points = cv2.perspectiveTransform(points, m).reshape((-1, 2))
        # add random noise to points
        jitter = map_params.get('jitter', 0)
        if jitter:
            # TODO fix quadratic sigma scaling
            mean = np.array([0, 0])
            cov = np.array([[jitter * mw, 0], [0, jitter * mw]])
            points += multivariate_normal(mean, cov, len(points))
            points = check_range(points, mh, mw)
            if not points.size:
                raise ValueError('no gaze points')

        interpolate = map_params.get('interpolate', False)
        if interpolate:
            y, x, v = interpolate_points(points, mh, mw)
            map_buffer[y, x] = v
        else:
            points = discretize_points(points, mh, mw)
            map_buffer[points[:, 1], points[:, 0]] = 1
        cv2.resize(map_buffer, target.shape, target, 0, 0,
                   interpolation=cv2.INTER_AREA)
        blur = map_params.get('blur', 0)
        if blur:
            target[...] = cv2.GaussianBlur(target, (0, 0), blur*target.shape[1])
        if map_params.get('maxnorm'):
            target /= target.max() + 1e-9