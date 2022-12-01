#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements the softmax output layer.
"""

import numpy
import theano
from theano import tensor

from theanolm.network.samplingoutputlayer import SamplingOutputLayer

class SoftmaxLayer(SamplingOutputLayer):
    """Softmax Output Layer

    The output layer is a simple softmax layer that outputs the word
    probabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(*args, **kwargs)

        # Create the parameters. Weight matrix and bias for each input.
        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.output_size
        self._init_weight('input/W', (input_size, output_size), scale=0.01)
        if self._network.class_prior_probs is None:
            self._init_bias('input/b', output_size)
        else:
            initial_bias = numpy.log(self._network.class_prior_probs + 1e-10)
            self._init_bias('input/b', output_size, initial_bias)

        vocabulary = self._network.vocabulary
        self._unk_id = vocabulary.word_to_id['<unk>']
        self._unk_class_id = vocabulary.word_id_to_class_id[self._unk_id]

        self.output_probs = None
        self.target_probs = None
        self.unnormalized_logprobs = None
        self._layer_input = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        word projection. When generating text, there's just one sequence and one
        time step in the input.

        Sets ``self.output_probs`` to a symbolic matrix that specifies output
        probabilities for all classes, ``self.target_probs`` to one that
        specifies output probabilities for the target classes, and
        ``self.unnormalized_logprobs``, ``self.sample``,
        ``self.sample_logprobs``, ``self.seqshared_sample``,
        ``self.seqshared_sample_logprobs``, ``self.shared_sample``, and
        ``self.shared_sample_logprobs`` to symbolic matrices that can be used to
        obtain the data required to compute sampling output costs.
        """

        layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
        preact = self._tensor_preact(layer_input, 'input')

        # Combine the first two dimensions so that softmax is taken
        # independently for each location, over the output classes. If <unk> is
        # excluded, set those activations to -inf before normalization.
        # NOTE: We're checking the class ID so this will fail if there are other
        # words in the same class with <unk> word!
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        preact = preact.reshape([num_time_steps * num_sequences,
                                 self.output_size])
        if self._network.exclude_unk:
            float_type = numpy.dtype(theano.config.floatX).type
            log_zero = float_type('-inf')
            preact = tensor.set_subtensor(preact[:, self._unk_class_id],
                                          log_zero)
        output_probs = tensor.nnet.softmax(preact)

        # This variable contains probabilities for the whole vocabulary.
        self.output_probs = output_probs.reshape([num_time_steps,
                                                  num_sequences,
                                                  self.output_size])

        # The following variables can only be used when
        # self._network.target_class_ids is given to the function.
        element_ids = tensor.arange(num_time_steps * num_sequences)
        target_class_ids = self._network.target_class_ids.flatten()
        target_probs = output_probs[(element_ids, target_class_ids)]
        self.target_probs = target_probs.reshape([num_time_steps,
                                                  num_sequences])

        # Define layer input and unnormalized logprobs for computing
        # sampling-based costs. Only possible if the parameters are not split to
        # multiple devices.
        self._layer_input = layer_input
        if not len(self._devices) > 1:
            self.unnormalized_logprobs = \
                self._get_unnormalized_logprobs()
