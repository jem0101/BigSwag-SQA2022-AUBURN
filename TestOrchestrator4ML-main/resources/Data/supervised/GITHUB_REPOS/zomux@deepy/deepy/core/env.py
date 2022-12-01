#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import theano.tensor as T
import logging as loggers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams as SharedRandomStreams

logging = loggers.getLogger(__name__)

class GlobalEnvironment(object):

    DEFAULT_SEED = 3
    FLOATX = "float32"
    EPSILON = T.constant(1.0e-8, dtype=FLOATX)

    def __init__(self, seed=DEFAULT_SEED):
        """
        Initialize seed and global random variables.
        """
        if seed != self.DEFAULT_SEED:
            self._seed = seed
        elif 'DEEPY_SEED' in os.environ:
            self._seed = int(os.environ['DEEPY_SEED'])
        else:
            self._seed = self.DEFAULT_SEED
        if self._seed != self.DEFAULT_SEED:
            logging.info("set global random seed to %d" % self._seed)

        self._numpy_rand = np.random.RandomState(seed=self._seed)
        self._theano_rand = RandomStreams(seed=self._seed)
        self._shared_rand = SharedRandomStreams(seed=self._seed)
        self._default_initializer = None

    @property
    def numpy_rand(self):
        return self._numpy_rand

    @property
    def theano_rand(self):
        return self._theano_rand

    @property
    def shared_rand(self):
        return self._shared_rand

    @property
    def default_initializer(self):
        from deepy.utils.initializers import UniformInitializer
        if not self._default_initializer:
            self._default_initializer = UniformInitializer()
        return self._default_initializer

    def set_default_initializer(self, initializer):
        self._default_initializer = initializer


if "env" not in globals():
    env = GlobalEnvironment()
    FLOATX = env.FLOATX
    EPSILON = env.EPSILON