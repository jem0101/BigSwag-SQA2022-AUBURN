import tensorflow as tf

from .defaults import default_weight_init
from .meta import Parameter, parameter_model

__all__ = [
  'NormalizedParameter', 'normalized_parameter'
]

class NormalizedParameter(Parameter):
  def __init__(self, shape, w=default_weight_init, name=None, **properties):
    self.w = w(shape, **properties, name=name)
    self.broadcast = (slice(None, None, None), ) + tuple(
      None for _ in range(len(shape) - 1)
    )

    super(NormalizedParameter, self).__init__(
      self.w, shape=shape, name=name,
      **properties,
    )

  def get_output_for(self, w):
    norm = tf.sqrt(
      tf.reduce_sum(
        w ** 2,
        axis=range(1, len(self.shape()))
      )
    )
    return w / norm[self.broadcast]

  def get_output_shape_for(self, w_shape):
    return w_shape

normalized_parameter = parameter_model(NormalizedParameter)()