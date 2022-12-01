#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

from deepy.layers import NeuralLayer


class WordEmbedding(NeuralLayer):
    """
    Word embedding layer.
    The word embeddings are randomly initialized, and are learned over the time.
    """
    def __init__(self, size, vocab_size, zero_index=None, mask=None, load_values=None, init=None):
        from deepy.core.neural_var import NeuralVariable
        super(WordEmbedding, self).__init__("word_embed")
        self.size = size
        self.vocab_size = vocab_size
        self.output_dim = size
        self.zero_index = zero_index
        self._mask = mask.tensor if type(mask) == NeuralVariable else mask
        self._init = init
        self._load_values = load_values
        self.init(1)

    def prepare(self):
        if self._load_values is not None:
            self.embed_matrix = theano.shared(self._load_values, name="embeddings")
        else:
            self.embed_matrix = self.create_weight(self.vocab_size, self.size, "embeddings", initializer=self._init)
        self.register_parameters(self.embed_matrix)

    def compute_tensor(self, x, mask=None):
        mask = mask if mask else self._mask
        if self.zero_index is not None:
            mask = T.neq(x, self.zero_index)
            # To avoid negative index
            x = T.cast(x * mask, "int32")

        if x.ndim == 1:
            ret_tensor = self.embed_matrix[x]
        else:
            ret_tensor = self.embed_matrix[x.flatten()].reshape(list(x.shape) + [self.size])

        if mask:
            if x.ndim == 2:
                ret_tensor *= mask[:, :, None]
            elif x.ndim == 1:
                ret_tensor *= mask[:, None]
        return ret_tensor
