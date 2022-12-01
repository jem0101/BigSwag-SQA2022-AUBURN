#!/usr/bin/env python
# -*- coding: utf-8 -*-


from network import NeuralNetwork
from deepy.core import env
import deepy.tensor as DT
import theano.tensor as T

class NeuralClassifier(NeuralNetwork):
    """
    A class of defining stacked neural network for classifiers.
    """

    def __init__(self, input_dim, input_tensor=None):
        super(NeuralClassifier, self).__init__(input_dim, input_tensor=input_tensor)

    def setup_variables(self):
        super(NeuralClassifier, self).setup_variables()

        self.k = T.ivector('k')
        self.target_variables.append(self.k)

    def _cost_func(self, y):
        y = T.clip(y, env.EPSILON, 1.0 - env.EPSILON)
        return DT.costs.cross_entropy(y, self.k)

    def _error_func(self, y):
        return 100 * T.mean(T.neq(T.argmax(y, axis=1), self.k))

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)

    def prepare_training(self):
        self.training_monitors.append(("err", self._error_func(self.output)))
        self.testing_monitors.append(("err", self._error_func(self.test_output)))
        super(NeuralClassifier, self).prepare_training()

    def predict(self, x):
        return self.compute(x).argmax(axis=1)

class MultiTargetNeuralClassifier(NeuralClassifier):
    """
    Classifier for multiple targets.
    """

    def __init__(self, class_num, input_dim=None):
        super(MultiTargetNeuralClassifier, self).__init__(input_dim)
        self.class_num = class_num

    def setup_vars(self):
        super(NeuralClassifier, self).setup_variables()

        self.k = T.imatrix('k')
        self.target_variables.append(self.k)

    def _cost_func(self, y):
        entropy_sum = T.constant(0, dtype=env.FLOATX)
        for i in range(self.class_num):
            entropy_sum += T.sum(T.nnet.categorical_crossentropy(self._output[:, i, :], self._output[:,i]))
        return entropy_sum / (self.k.shape[0] * self.k.shape[1])

    def _error_func(self, y):
        return 100 * T.mean(T.neq(T.argmax(self._output, axis=2), self.k))


