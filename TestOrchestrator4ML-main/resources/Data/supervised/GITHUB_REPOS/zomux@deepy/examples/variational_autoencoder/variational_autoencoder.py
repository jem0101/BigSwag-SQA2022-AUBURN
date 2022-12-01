#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
from deepy import NeuralLayer, AutoEncoder, Dense
from deepy import GaussianInitializer, global_theano_rand


class ReparameterizationLayer(NeuralLayer):
    """
    Reparameterization layer in a Variational encoder.
    Only binary output cost function is supported now.
    The prior value is recorded after the computation graph created.
    """

    def __init__(self, size, sample=True):
        """
        :param size: the size of latent variable
        :param sample: whether to get a clean latent variable
        """
        super(ReparameterizationLayer, self).__init__("VariationalEncoder")
        self.size = size
        self.output_dim = size
        self.sample = sample
        self._prior = None

    def prepare(self):
        self._mu_encoder = Dense(self.size, 'linear', init=GaussianInitializer(), random_bias=True).init(
            self.input_dim)
        self._log_sigma_encoder = Dense(self.size, 'linear', init=GaussianInitializer(), random_bias=True).init(
            self.input_dim)
        self.register_inner_layers(self._mu_encoder, self._log_sigma_encoder)

    def compute_tensor(self, x):
        # Compute p(z|x)
        mu = self._mu_encoder.compute_tensor(x)
        log_sigma = 0.5 * self._log_sigma_encoder.compute_tensor(x)
        self._prior = 0.5* T.sum(1 + 2*log_sigma - mu**2 - T.exp(2*log_sigma))
        # Reparameterization
        eps = global_theano_rand.normal((x.shape[0], self.size))

        if not self.sample:
            z = mu
        else:
            z = mu + T.exp(log_sigma) * eps
        return z

    def prior(self):
        """
        Get the prior value.
        """
        return self._prior


class VariationalAutoEncoder(AutoEncoder):
    """
    Variational Auto Encoder.
    Only binary output cost function is supported now.
    """

    def __init__(self, input_dim, rep_dim, input_tensor=None, sample=True):
        """
        """
        super(VariationalAutoEncoder, self).__init__(input_dim, rep_dim)
        self.sample = sample
        self._setup_monitors = True


    def stack_reparameterization_layer(self, layer_size):
        """
        Perform reparameterization trick for latent variables.
        :param layer_size: the size of latent variable
        """
        self.rep_layer = ReparameterizationLayer(layer_size, sample=self.sample)
        self.stack_encoders(self.rep_layer)

    def _cost_func(self, y):
        logpxz  = - T.nnet.binary_crossentropy(y, self.input_variables[0]).sum()
        logp = logpxz + self.rep_layer.prior()
        # the lower bound is the mean value of logp
        cost = - logp
        if self._setup_monitors:
            self._setup_monitors = False
            self.training_monitors.append(("lower_bound", logp / y.shape[0]))
            self.testing_monitors.append(("lower_bound", logp / y.shape[0]))
        return cost
