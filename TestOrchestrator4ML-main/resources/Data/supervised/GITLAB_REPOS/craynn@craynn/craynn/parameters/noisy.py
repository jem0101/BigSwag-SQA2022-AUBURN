import tensorflow as tf

from .meta import Parameter, parameter_model
from .defaults import default_weight_init

__all__ = [
  'NoisyParameter', 'noisy_parameter'
]

class NoisyParameter(Parameter):
  def __init__(self, shape, eps=1e-3, w=default_weight_init, name=None, **properties):
    self.eps = tf.constant(eps, dtype=tf.float32, shape=())

    self.w = w(
      shape=shape, **properties,
      name=(name + '_w') if name is not None else None
    )

    super(NoisyParameter, self).__init__(
      self.w, shape=shape, name=name,
      **properties
    )

  def get_output_for(self, w, deterministic=False):
    if deterministic:
      return w
    else:
      return w + tf.random.normal(shape=tf.shape(w), stddev=self.eps, dtype=w.dtype)

  def get_output_shape_for(self, w_shape):
    return w_shape

noisy_parameter = parameter_model(NoisyParameter)()
