#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the TextSampler class.
"""

import numpy
import theano
from theanolm.network import RecurrentState

class TextSampler(object):
    """Neural network language model sampler

    A Theano function that generates text using a neural network language
    model.
    """

    def __init__(self, network):
        """Creates a Theano function that samples the next word of a set of word
        sequences.

        Creates the function self.step_function that takes as input a set of
        word sequences and the current recurrent states. It uses the previous
        states and word IDs to compute the output distributions. It samples from
        the output distributions and returns the sampled word IDs along with the
        output states of this time step.

        :type network: Network
        :param network: the neural network object
        """

        self._network = network
        self._vocabulary = network.vocabulary
        self._random = network.random

        inputs = [network.input_word_ids, network.input_class_ids]
        inputs.extend(network.recurrent_state_input)

        # multinomial() is only implemented for dimension <= 2, but the matrix
        # contains only one time step anyway.
        output_probs = network.output_probs()[0]
        # Theano doesn't set a test value to some of it's internal variables.
        # Disable the warning since there's nothing we can do.
        compute_test_value = theano.config.compute_test_value
        theano.config.compute_test_value = 'off'
        class_ids = self._random.multinomial(pvals=output_probs).argmax(1)
        class_ids.tag.test_value = numpy.zeros(4, dtype='int64')
        theano.config.compute_test_value = compute_test_value
        class_ids = class_ids.reshape([1, class_ids.shape[0]])
        outputs = [class_ids]
        outputs.extend(network.recurrent_state_output)

        # Ignore unused input, because is_training is only used by dropout
        # layer.
        self.step_function = theano.function(
            inputs,
            outputs,
            givens=[(network.is_training, numpy.int8(0))],
            name='step_sampler',
            on_unused_input='ignore')

    def generate(self, length, num_sequences=1, seed_sequence=''):
        """Generates a text sequence.

        Calls self.step_function() repeatedly, reading the word output and
        the state output of the hidden layer and passing the hidden layer state
        output to the next time step.

        :type length: int
        :param length: number of words (tokens) in each sequence

        :type num_sequences: int
        :param num_sequences: number of sequences to generate in parallel

        :rtype: list of list of strs
        :returns: list of word sequences
        """
        seed_tokens = seed_sequence.strip().split()
        sos_id = self._vocabulary.word_to_id['<s>']
        sos_class_id = self._vocabulary.word_id_to_class_id[sos_id]

        input_word_ids = sos_id * \
                         numpy.ones(shape=(1, num_sequences)).astype('int64')
        input_class_ids = sos_class_id * \
                          numpy.ones(shape=(1, num_sequences)).astype('int64')
        result = sos_id * \
                 numpy.ones(shape=(len(seed_tokens)+length,
                   num_sequences)).astype('int64')
        state = RecurrentState(self._network.recurrent_state_size,
                               num_sequences)
       
        #First, possibly compute forward passes with the seed sequence
        for time_step, token in enumerate(seed_tokens, start=1):
            step_result = self.step_function(input_word_ids,
                                             input_class_ids,
                                             *state.get())
            token_id = self._vocabulary.word_to_id[token]
            token_class_id = \
                self._vocabulary.word_id_to_class_id[token_id]
            input_word_ids = token_id * \
                             numpy.ones(shape=(1,      
                               num_sequences)).astype('int64')
            input_class_ids = token_class_id * \
                              numpy.ones(shape=(1,
                                num_sequences)).astype('int64')
            step_word_ids = input_word_ids
            result[time_step] = step_word_ids
            state.set(step_result[1:])
        
        #Then sample:
        for time_step in range(len(seed_tokens) + 1, length+len(seed_tokens)):
            # the input is the output from the previous step.
            step_result = self.step_function(input_word_ids,
                                             input_class_ids,
                                             *state.get())
            class_ids = step_result[0]
            # The class IDs from the single time step.
            step_class_ids = class_ids[0]
            step_word_ids = numpy.array(
                self._vocabulary.class_ids_to_word_ids(step_class_ids))
            result[time_step] = step_word_ids
            input_word_ids = step_word_ids[numpy.newaxis]
            input_class_ids = class_ids
            state.set(step_result[1:])

        return self._vocabulary.id_to_word[result.transpose()].tolist()
