#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the RMSProp optimizer with Nesterov momentum.
"""

import numpy
import theano.tensor as tensor

from theanolm.backend import Parameters
from theanolm.training.basicoptimizer import BasicOptimizer

class RMSPropNesterovOptimizer(BasicOptimizer):
    """RMSProp Variation of Nesterov Momentum Optimization Method

    At the time of writing, RMSProp is an unpublished method. Usually people
    cite slide 29 of Lecture 6 of Geoff Hinton's Coursera class:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    The idea is simply to maintain a running average of the squared gradient for
    each parameter, and divide the gradient by the root of the mean squared
    gradient (RMS). This makes RMSProp take steps near 1 whenever the gradient
    is of constant magnitude, and larger steps whenever the local scale of the
    gradient starts to increase.

    RMSProp has been implemented over many optimization methods. This
    implementation is based on the Nesterov Momentum method. We use an
    alternative formulation that requires the gradient to be computed only at
    the current parameter values, described here:
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
    except that we divide the gradient by the RMS gradient:

    rmsprop_{t-1} = -lr * gradient(params_{t-1}) / rms_gradient(params_{t-1})
    v_{t} = mu * v_{t-1} + rmsprop_{t-1}
    params_{t} = params_{t-1} + mu * v_{t} + rmsprop_{t-1}
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an RMSProp momentum optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self._params = Parameters()
        for path, param in network.get_variables().items():
            # Initialize mean squared gradient to ones, otherwise the first
            # update will be divided by close to zero.
            self._params.add(path + '_mean_sqr_gradient',
                             numpy.ones_like(param.get_value()))
            self._params.add(path + '_velocity',
                             numpy.zeros_like(param.get_value()))

        # geometric rate for averaging gradients
        if 'gradient_decay_rate' not in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma = optimization_options['gradient_decay_rate']

        # momentum
        if 'momentum' not in optimization_options:
            raise ValueError("Momentum is not given in optimization options.")
        self._momentum = optimization_options['momentum']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_param_updates(self, alpha):
        """Returns Theano expressions for updating the model parameters and any
        additional parameters required by the optimizer.

        :type alpha: Variable
        :param alpha: a scale to be applied to the model parameter updates

        :rtype: iterable over pairs (shared variable, new expression)
        :returns: expressions how to update the optimizer parameters
        """

        result = []
        deltas = dict()
        for path, gradient in zip(self.network.get_variables(),
                                  self._gradients):
            ms_gradient_old = self._params[path + '_mean_sqr_gradient']
            ms_gradient = \
                self._gamma * ms_gradient_old + \
                (1.0 - self._gamma) * tensor.sqr(gradient)
            result.append((ms_gradient_old, ms_gradient))

            rms_gradient = tensor.sqrt(ms_gradient + self._epsilon)
            deltas[path] = -gradient / rms_gradient
        self._normalize(deltas)

        result = []
        for path, param_old in self.network.get_variables().items():
            delta = deltas[path]
            velocity_old = self._params[path + '_velocity']
            velocity = self._momentum * velocity_old + alpha * delta
            param = param_old + self._momentum * velocity + alpha * delta
            result.append((velocity_old, velocity))
            result.append((param_old, param))
        return result
