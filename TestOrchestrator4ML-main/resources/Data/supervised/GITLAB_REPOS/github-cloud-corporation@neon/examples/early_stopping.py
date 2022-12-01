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
Example that trains an MLP using early stopping.
Training will stop when the stopping condition is satisfied
or when num_epochs has been reached, whichever is first.

Usage:

    python examples/early_stopping.py

"""

import os
from neon.data import ArrayIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)
train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
valid_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# initialize model
layers = []
layers.append(Affine(nout=100, init=init_norm, batch_norm=True, activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True)))
cost = GeneralizedCost(costfunc=CrossEntropyBinary())
mlp = Model(layers=layers)

# define stopping function
# it takes as input a tuple (State,val[t])
# which describes the cumulative validation state (generated by this function)
# and the validation error at time t
# and returns as output a tuple (State', Bool),
# which represents the new state and whether to stop


# Stop if validation error ever increases from epoch to epoch
def stop_func(s, v):
    if s is None:
        return (v, False)

    return (min(v, s), v > s)

# fit and validate
optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1

callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)
callbacks.add_early_stop_callback(stop_func)
callbacks.add_save_best_state_callback(os.path.join(args.data_dir, "early_stop-best_state.pkl"))
mlp.fit(train_set,
        optimizer=optimizer,
        num_epochs=args.epochs,
        cost=cost,
        callbacks=callbacks)
