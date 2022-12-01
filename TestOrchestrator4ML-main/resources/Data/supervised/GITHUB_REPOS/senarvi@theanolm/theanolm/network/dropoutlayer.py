#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the dropout layer.
"""

import logging

import numpy
import theano
import theano.tensor as tensor

from theanolm.network.basiclayer import BasicLayer

class DropoutLayer(BasicLayer):
    """Dropout Layer

    A dropout layer is not a regular layer in the sense that it doesn't contain
    any neurons. It simply randomly sets some activations to zero at train time
    to prevent overfitting.

    N. Srivastava et al. (2014)
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    Journal of Machine Learning Research, 15, 1929-1958
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Validates the parameters of this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        if 'dropout_rate' in layer_options:
            self._dropout_rate = float(layer_options['dropout_rate'])
            # Don't allow dropout rate too close to 1.0 to prevent division by
            # zero.
            self._dropout_rate = min(0.9999, self._dropout_rate)
        else:
            self._dropout_rate = 0.5
        logging.debug("  dropout_rate=%f", self._dropout_rate)

        # Make sure the user hasn't tried to change the number of connections.
        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.output_size
        if input_size != output_size:
            raise ValueError("Dropout layer size has to match the previous "
                             "layer.")

        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets ``self.output`` to a symbolic matrix that describes the output of
        this layer. During training sets randomly some of the outputs to zero.
        """

        float_type = numpy.dtype(theano.config.floatX).type

        layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
        # Pass rate is the probability of not dropping a unit.
        pass_rate = 1.0 - self._dropout_rate
        pass_rate = float_type(pass_rate)
        sample = self._network.random.uniform(size=layer_input.shape)
        mask = tensor.cast(sample < pass_rate, theano.config.floatX)
        # Multiply the output by the inverse of the pass rate before dropping
        # units to compensate the scaling effect.
        scale_correction = 1.0 / pass_rate
        scale_correction = float_type(scale_correction)
        self.output = tensor.switch(self._network.is_training,
                                    layer_input * scale_correction * mask,
                                    layer_input)
