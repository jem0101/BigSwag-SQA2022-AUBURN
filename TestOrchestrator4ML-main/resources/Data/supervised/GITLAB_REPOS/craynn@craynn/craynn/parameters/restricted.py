import tensorflow as tf

from .common import zeros_init
from .glorot import glorot_normal_init
from .meta import Parameter, parameter_model

__all__ = [
  'SoftmaxParameter', 'softmax_parameter',
  'SoftplusParameter', 'softplus_parameter',
]

class SoftmaxParameter(Parameter):
  def __init__(self, shape, scale=1, w=zeros_init(), name=None, **properties):
    self.scale = None if scale == 1 else tf.constant(scale, dtype=tf.float32)

    self.w = w(shape=shape, **properties, name=(name + '_w') if name is not None else None)

    super(SoftmaxParameter, self).__init__(
      self.w, shape=shape, name=name,
      **properties
    )

  def get_output_for(self, w):
    restricted = tf.nn.softmax(w)
    if self.scale is not None:
      return self.scale * restricted
    else:
      return restricted

  def get_output_shape_for(self, w_shape):
    return w_shape

softmax_parameter = parameter_model(SoftmaxParameter)()


class SoftplusParameter(Parameter):
  def __init__(self, shape, w=glorot_normal_init(gain=0.5), name=None, **properties):
    self.w = w(shape=shape, **properties, name=(name + '_w') if name is not None else None)

    super(SoftplusParameter, self).__init__(
      self.w, shape=shape, name=name,
      **properties
    )

  def get_output_for(self, w):
    return tf.nn.softplus(w)

  def get_output_shape_for(self, w_shape):
    return w_shape

softplus_parameter = parameter_model(SoftplusParameter)()