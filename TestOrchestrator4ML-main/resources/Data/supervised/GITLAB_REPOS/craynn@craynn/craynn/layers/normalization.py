### inspired by
### https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/normalization.py#L124-L326

import tensorflow as tf

from ..nonlinearities import linear
from ..parameters import zeros_init, ones_init

from .meta import Layer, get_output_shape, model_from

__all__ = [
  'BatchNormLayer', 'batch_norm',
  'LayerNormLayer', 'layer_norm',
]

class BatchNormLayer(Layer):
  def __init__(self, incoming, gamma=ones_init(), beta=zeros_init(),
               axes=(0, ), activation=linear(), epsilon=1.0e-4, name=None):
    self.epsilon = tf.constant(epsilon, dtype='float32')

    incoming_shape = get_output_shape(incoming)
    self.axes = list(range(len(incoming_shape) - 1)) if axes == 'auto' else axes

    self.parameter_shape = tuple(
      d
      for i, d in enumerate(incoming_shape)
      if i not in self.axes
    )

    self.broadcast = tuple(
      (None if i in axes else slice(None, None, None))
      for i, _ in enumerate(incoming_shape)
    )

    parameters = list()

    if gamma is not None:
      self.gamma = gamma(
        self.parameter_shape,
        weights=True, normalization=True,
        normalization_scales=True, trainable=True
      )
      parameters.append(self.gamma)
    else:
      self.gamma = None

    if beta is not None:
      self.beta = beta(
        self.parameter_shape,
        biases=True, normalization=True,
        normalization_biases=True, trainable=True
      )
      parameters.append(self.beta)
    else:
      self.beta = None

    self.activation = activation

    super(BatchNormLayer, self).__init__(
      incoming,
      name=name,
      parameters=tuple(parameters)
    )

  def _split_arguments(self, *args):
    if self.gamma is not None:
      gamma = args[0]
    else:
      gamma = None

    if self.beta is not None:
      beta = args[1] if self.gamma is not None else args[0]
    else:
      beta = None

    return gamma, beta, args[-1]

  def get_output_for(self, *inputs):
    gamma, beta, input = self._split_arguments(*inputs)

    input_mean = tf.reduce_mean(input, axis=self.axes)
    input_var = tf.reduce_mean((input - input_mean[self.broadcast]) ** 2, axis=self.axes)
    input_inverse_std = 1 / tf.sqrt(input_var + self.epsilon)

    y = (input - input_mean[self.broadcast]) * input_inverse_std[self.broadcast]

    scaled = y if gamma is None else y * gamma[self.broadcast]
    biased = scaled if beta is None else scaled + beta[self.broadcast]

    return self.activation(biased)

  def get_output_shape_for(self, *input_shapes):
    _, _, input_shape = self._split_arguments(*input_shapes)
    return input_shape

batch_norm = model_from(BatchNormLayer)()


class LayerNormLayer(Layer):
  def __init__(self, incoming, axes='auto', epsilon=1.0e-4, name=None):
    self.epsilon = tf.constant(epsilon, dtype='float32')

    incoming_shape = get_output_shape(incoming)
    self.axes = list(range(len(incoming_shape) - 1)) if axes == 'auto' else axes

    self.broadcast = tuple(
      (tf.newaxis if i in self.axes else slice(None, None, None))
      for i, _ in enumerate(incoming_shape)
    )

    super(LayerNormLayer, self).__init__(incoming, name=name)

  def get_output_for(self, input):
    input_mean = tf.reduce_mean(input, axis=self.axes)
    input_var = tf.reduce_mean((input - input_mean[self.broadcast]) ** 2, axis=self.axes)
    input_inverse_std = 1 / tf.sqrt(input_var + self.epsilon)

    return (input - input_mean[self.broadcast]) * input_inverse_std[self.broadcast]

  def get_output_shape_for(self, input_shape):
    return input_shape

layer_norm = model_from(LayerNormLayer)()