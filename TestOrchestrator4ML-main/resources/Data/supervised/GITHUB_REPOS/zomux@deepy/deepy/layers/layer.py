#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging as loggers

import numpy as np
import theano

from deepy.utils import UniformInitializer
from deepy.core.env import env
from deepy.core.tensor_conversion import neural_computation_prefer_tensor, convert_to_theano_var

logging = loggers.getLogger("deepy")

class NeuralLayer(object):

    def __init__(self, name=None):
        """
        Create a neural layer.
        """
        self.name = name if name else self.__class__.__name__
        self.input_dim = 0
        self.input_dims = [0]
        self.output_dim = 0
        self.output_dims= [0]

        self._linked_block = None

        self.initialized = False
        self.updates = []
        self.training_updates = []
        self.free_parameters = []
        self.parameters = []
        self.training_monitors = []
        self.testing_monitors = []
        self._registered_monitors = set()
        self._registered_updates = set()
        self._registered_training_updates = set()
        self.external_inputs = []
        self.external_targets = []
        self.parameter_count = 0
        self.epoch_callbacks = []
        self.training_callbacks = []
        self.testing_callbacks = []

    def init(self, input_dim=0, input_dims=None, no_prepare=False):
        """
        Initialize the layer.
        :param no_prepare: avoid calling preparation function
        """
        if self.initialized:
            return
        # configure input dimensions
        if input_dims:
            self.input_dims = input_dims
            self.input_dim = input_dims[0]
        else:
            self.input_dim = input_dim
            self.input_dims = [input_dims]
        # set default output dimension
        if self.output_dim == 0:
            self.output_dim = self.input_dim
        self.initialized = True
        # call prepare
        if not no_prepare:
            self.prepare()
        return self

    def compute(self, *inputs, **kwargs):
        """
        Compute based on NeuralVariable.
        :type inputs:  list of NeuralVariable
        :return: NeuralVariable
        """
        from deepy.core.neural_var import NeuralVariable
        from deepy.core.graph import graph
        if type(inputs[0]) != NeuralVariable:
            raise SystemError("The input of `compute` must be NeuralVar")

        dims = [t.dim() for t in inputs]
        if len(inputs) == 1:
            self.init(input_dim=dims[0])
        else:
            self.init(input_dims=dims)
        # Check block
        if self.parameters and not self._linked_block:
            self.belongs_to(graph.default_block())
        # convert kwargs
        train_kwargs, _, _ = convert_to_theano_var(kwargs)

        output = self.compute_tensor(*[t.tensor for t in inputs], **train_kwargs)

        if type(output) != list and type(output) != tuple:
            return NeuralVariable(output, dim=self.output_dim)
        else:
            return [NeuralVariable(*item) for item in zip(output, self.output_dims)]

    def prepare(self):
        """
        Prepare function will be called after connected.
        """

    @neural_computation_prefer_tensor
    def compute_tensor(self, *args, **kwargs):
        """
        Compute with tensors in Theano.
        """
        raise NotImplementedError("output function of '%s' is not implemented" % self.name)

    def belongs_to(self, block):
        """
        Let the given block or network manage the parameters of this layer.
        :param block: Block or NeuralNetwork
        :return: NeuralLayer
        """
        if self._linked_block:
            raise SystemError("The layer {} has already blonged to {}".format(self.name, self._linked_block.name))
        self._linked_block = block
        block.register_layer(self)
        return self

    def register(self, *layers):
        """
        Register inner layers.
        """
        self.register_inner_layers(*layers)

    def register_inner_layers(self, *layers):
        for layer in layers:
            self.register_parameters(*layer.parameters)
            self.register_updates(*layer.updates)
            self.register_training_updates(*layer.training_updates)
            self.training_monitors.extend(layer.training_monitors)
            self.testing_monitors.extend(layer.testing_monitors)

    def register_parameters(self, *parameters):
        """
        Register parameters.
        """
        for param in parameters:
            self.parameter_count += np.prod(param.get_value().shape)
        self.parameters.extend(parameters)

    def register_free_parameters(self, *free_parameters):
        """
        Register free parameters, which means their value will not be learned by trainer.
        """
        return self.free_parameters.extend(free_parameters)

    def register_updates(self, *updates):
        """
        Register updates that will be executed in each iteration.
        """
        for key, node in updates:
            if key not in self._registered_updates:
                self.updates.append((key, node))
                self._registered_updates.add(key)

    def register_training_updates(self, *updates):
        """
        Register updates that will only be executed in training phase.
        """
        for key, node in updates:
            if key not in self._registered_training_updates:
                self.training_updates.append((key, node))
                self._registered_training_updates.add(key)

    def register_monitors(self, *monitors):
        """
        Register monitors they should be tuple of name and Theano variable.
        """
        for key, node in monitors:
            if key not in self._registered_monitors:
                node *= 1.0 # Avoid CudaNdarray
                self.training_monitors.append((key, node))
                self.testing_monitors.append((key, node))
                self._registered_monitors.add(key)

    def register_external_inputs(self, *variables):
        """
        Register external input variables.
        """
        self.external_inputs.extend(variables)

    def register_external_targets(self, *variables):
        """
        Register extenal target variables.
        """
        self.external_targets.extend(variables)

    def register_training_callbacks(self, *callbacks):
        """
        Register callback for each iteration in the training.
        """
        self.training_callbacks.extend(callbacks)

    def register_testing_callbacks(self, *callbacks):
        """
        Register callback for each iteration in the testing.
        """
        self.testing_callbacks.extend(callbacks)

    def register_epoch_callbacks(self, *callbacks):
        """
        Register callback which will be called after epoch finished.
        """
        self.epoch_callbacks.extend(callbacks)

    def create_weight(self, input_n=1, output_n=1, label="W", initializer=None, shape=None):
        if not shape:
            shape = (input_n, output_n)

        if not initializer:
            initializer = env.default_initializer

        weight = theano.shared(initializer.sample(shape).astype(env.FLOATX), name='{}_{}'.format(self.name, label))

        logging.info('create param %s %s for %s', label, str(shape), self.name)
        return weight

    def create_bias(self, output_n=1, label="B", value=0., shape=None):
        if not shape:
            shape = (output_n, )
        bs =  np.ones(shape)
        bs *= value
        bias = theano.shared(bs.astype(env.FLOATX), name='{}_{}'.format(self.name, label))
        logging.info('create param %s %s for %s', label, str(shape), self.name)
        return bias

    def create_scalar(self, name="S", value=0, dtype=env.FLOATX):
        bs = np.array(0, dtype=dtype)
        bs += value
        v = theano.shared(bs, name='{}_{}'.format(self.name, name))

        logging.info('create scalar %s', name)
        return v

    def create_vector(self, n, name="V", dtype=env.FLOATX):
        bs =  np.zeros(n, dtype=dtype)
        v = theano.shared(bs, name='{}_{}'.format(self.name, name))

        logging.info('create vector %s: %d', name, n)
        return v

    def create_matrix(self, m, n, name="M"):

        matrix = theano.shared(np.zeros((m, n)).astype(env.FLOATX), name="{}_{}".format(self.name, name))

        logging.info('create matrix %s: %d x %d', name, m, n)
        return matrix

    def activation(self, name):
        from deepy.tensor.activations import get_activation
        return get_activation(name)

    def callback_forward_propagation(self):
        pass

    def set_name(self, name):
        """
        Set the name of this layer.
        This will be the key of saved parameters.
        """
        self.name = name