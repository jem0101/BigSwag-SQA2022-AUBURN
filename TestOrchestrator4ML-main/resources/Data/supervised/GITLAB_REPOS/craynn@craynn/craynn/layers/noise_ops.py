import tensorflow as tf

from craygraph import derive
from .meta import Layer, model_from, get_output_shape

__all__ = [
  'NoiseLayer', 'noise',
  'GaussianNoiseLayer', 'gaussian_noise',

  'GaussianRandomVariableLayer', 'gaussian_rv',
  'DropoutLayer', 'dropout'
]

class NoiseLayer(Layer):
  def __init__(self, incoming, noise_op, scale=1.0e-3, name=None):
    self.noise_op = noise_op
    self.scale = tf.constant(scale, dtype=tf.float32)

    super(NoiseLayer, self).__init__(incoming, name=name)

  def get_output_for(self, input, deterministic=False):
    if deterministic:
      return input
    else:
      return input + self.noise_op(shape=tf.shape(input), dtype=input.dtype) * self.scale

  def get_output_shape_for(self, input_shape):
    return input_shape


noise = model_from(NoiseLayer)()

GaussianNoiseLayer = derive('GaussianNoiseLayer').based_on(NoiseLayer).with_fixed(noise_op=tf.random.normal)
gaussian_noise = model_from(GaussianNoiseLayer)()

class DropoutLayer(Layer):
  def __init__(self, incoming, p=0.2, name=None):
    self.p = p

    super(DropoutLayer, self).__init__(incoming, name=name)

  def get_output_for(self, input, deterministic=False):
    if deterministic:
      return input
    else:
      return tf.nn.dropout(input, rate=self.p, noise_shape=None, name=self.name())

  def get_output_shape_for(self, input_shape):
    return input_shape

dropout = model_from(DropoutLayer)()


class GaussianRandomVariableLayer(Layer):
  def __init__(self, *incoming, name=None):
    assert len(incoming) == 2

    super(GaussianRandomVariableLayer, self).__init__(*incoming, name=name)

  def get_output_for(self, mean, std, deterministic=False):
    if deterministic:
      return mean
    else:
      u = tf.random.normal(shape=tf.shape(mean), dtype=mean.dtype)
      return u * std + mean

  def get_output_shape_for(self, *input_shapes):
    return input_shapes[0]

gaussian_rv = model_from(GaussianRandomVariableLayer)()

