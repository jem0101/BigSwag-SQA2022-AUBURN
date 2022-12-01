#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor as T
import deepy.tensor as DT
from . import NeuralNetwork
from deepy.utils import dim_to_var


class NeuralRegressor(NeuralNetwork):
    """
    A class of defining stacked neural network regressors.
    """
    def __init__(self, input_dim, target_tensor=2, clip_value=None, input_tensor=None):
        self.target_tensor = dim_to_var(target_tensor, "k") if type(target_tensor) == int else target_tensor
        self.clip_value = clip_value
        super(NeuralRegressor, self).__init__(input_dim, input_tensor=input_tensor)

    def setup_variables(self):
        super(NeuralRegressor, self).setup_variables()
        self.k = self.target_tensor
        self.target_variables.append(self.k)

    def _cost_func(self, y):
        if self.clip_value:
            y = T.clip(y, -self.clip_value, self.clip_value)
        return DT.costs.least_squares(y, self.k)

    @property
    def cost(self):
        return self._cost_func(self.output)

    @property
    def test_cost(self):
        return self._cost_func(self.test_output)