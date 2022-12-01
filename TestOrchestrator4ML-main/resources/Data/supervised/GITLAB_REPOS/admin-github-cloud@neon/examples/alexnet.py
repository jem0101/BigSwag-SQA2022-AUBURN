#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Alexnet - An implementation of a deep convolutional neural network for
classification of images from the ImageNet 2012 competition based on
Krizhevsky, Sutskever and Hinton, 2012.

Reference:

    ImageNet Classification with Deep Convolutional Neural Networks `[Krizhevsky2015]`_
..  _[Krizhevsky2015]: http://papers.nips.cc/paper/\
4824-imagenet-classification-with-deep-convolutional-neural-networks

Usage:

    Before training, prepare ImageNet macrobatches as described at
    http://neon.nervanasys.com/docs/latest/datasets.html#imagenet

    python examples/alexnet.py --data_dir </path/to/ImageNet/macrobatches> --epochs 90

"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
img_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', scale_range=(256, 256), **img_set_options)
test = ImageLoader(set_name='validation', scale_range=(256, 256), do_transforms=False,
                   **img_set_options)

layers = [Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0),
               activation=Rectlin(), padding=3, strides=4),
          Pooling(3, strides=2),
          Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1),
               activation=Rectlin(), padding=2),
          Pooling(3, strides=2),
          Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0),
               activation=Rectlin(), padding=1),
          Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
               activation=Rectlin(), padding=1),
          Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
               activation=Rectlin(), padding=1),
          Pooling(3, strides=2),
          Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
          Dropout(keep=0.5),
          Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
          Dropout(keep=0.5),
          Affine(nout=1000, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())]
model = Model(layers=layers)

# drop weights LR by 1/250**(1/3) at epochs (23, 45, 66), drop bias LR by 1/10 at epoch 45
weight_sched = Schedule([22, 44, 65], 0.15874)
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched,
                                  stochastic_round=args.rounding)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1),
                                     stochastic_round=args.rounding)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=test, metric=valmetric, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
