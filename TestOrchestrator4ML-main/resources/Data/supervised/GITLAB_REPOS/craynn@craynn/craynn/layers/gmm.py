import numpy as np
import tensorflow as tf

from ..parameters import zeros_init
from ..utils import normalize_axis

from .meta import Layer, model_from
from .meta import get_output_shape

__all__ = [
  'GaussianMixtureLayer',
  'gaussian_mixture',
]

class GaussianMixtureLayer(Layer):
  def __init__(
    self, incoming, num_components,
    mean=zeros_init(), log_sigma=zeros_init(), logit_priors=zeros_init(),
    name=None, axis=-1
  ):
    input_shape = get_output_shape(incoming)
    self.axis = normalize_axis(len(input_shape), axis)

    self.mean = mean(
        shape=tuple(input_shape[a] for a in self.axis) + (num_components, ),
        name='mean',
        biases=True, trainable=True
      )

    self.log_sigma = log_sigma(
      shape=(num_components,),
      name='sigma',
      scales=True, trainable=True
    )

    self.logit_priors = logit_priors(
      shape=(num_components, ),
      name='logit_priors',
      trainable=True
    )

    self.mean_broadcast = tuple(
      None if a in self.axis else slice(None, None, None)
      for a in self.axis
    ) + (slice(None, None, None), )

    self.input_broadcast = (slice(None, None, None), ) * len(input_shape) + (None, )

    self.sigma_broadcast = (None, ) * (len(input_shape) - len(self.axis)) + (slice(None, None, None), )

    self.log_2_pi = tf.constant(np.log(2 * np.pi), dtype=tf.float32)

    super(GaussianMixtureLayer, self).__init__(
      incoming,
      name=name,
      parameters=(self.mean, self.log_sigma, self.logit_priors)
    )

  def get_output_for(self, mean, log_sigma, logit_priors, X):
    dim = len(self.axis)
    sigma = tf.exp(log_sigma)
    log_priors = tf.nn.log_softmax(logit_priors)

    diff = 0.5 * tf.reduce_sum(
      (X[self.input_broadcast] - mean[self.mean_broadcast]) ** 2,
      axis=self.axis
    ) / (sigma[self.sigma_broadcast] ** 2)

    log_probs = -dim / 2 * self.log_2_pi - dim * log_sigma - diff

    return tf.reduce_logsumexp(log_priors + log_probs, axis=-1)

  def get_output_shape_for(self, mean_shape, log_sigma_shape, input_shape):
    return tuple(
      d
      for i, d in enumerate(input_shape)
      if i not in self.axis
    )

gaussian_mixture = model_from(GaussianMixtureLayer)()