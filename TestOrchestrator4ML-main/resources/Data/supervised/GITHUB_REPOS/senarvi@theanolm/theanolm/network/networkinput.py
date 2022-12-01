#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the NetworkInput class.
"""

from theanolm.network.basiclayer import BasicLayer

class NetworkInput(BasicLayer):
    """Neural Network Input Element

    A dummy layer that provides the input for the first layer.
    """

    def __init__(self, input_options, network):
        """Creates a neural network input with a given vocabulary size, which
        specifies the input size of the first layer.

        :type input_options: dict
        :param input_options: dictionary of input options

        :type network: Network
        :param network: the network object which uses this input
        """

        self.input_type = input_options['type']
        if self.input_type == 'word':
            output_size = network.vocabulary.num_shortlist_words()
        elif self.input_type == 'class':
            output_size = network.vocabulary.num_classes()
        else:
            raise ValueError(
                "Invalid network input type: {}".format(self.input_type))
        input_options['size'] = output_size
        input_options['input_layers'] = []
        input_options['devices'] = []

        self.output = None

        super().__init__(input_options, network)

    def create_structure(self):
        """Creates the symbolic matrix that describes the network input.

        The tensor variable will be set to a matrix of word IDs, with
        [ number of time steps * number of sequences ] elements. When generating
        text, the matrix will contain only one element.
        """

        if self.input_type == 'word':
            self.output = self._network.input_word_ids
        elif self.input_type == 'class':
            self.output = self._network.input_class_ids
        else:
            assert False
