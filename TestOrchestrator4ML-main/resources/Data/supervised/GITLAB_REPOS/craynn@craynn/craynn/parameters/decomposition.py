import tensorflow as tf

from .defaults import default_weight_init
from .meta import Parameter, parameter_model

__all__ = [
  'DecompositionParameter',
  'decomposition'
]

class DecompositionParameter(Parameter):
  def __init__(self, shape, n, w1=default_weight_init, w2=default_weight_init, name=None, **properties):
    shape1 = (shape[0], n) + shape[2:]
    shape2 = (n, shape[1]) + shape[2:]

    dependencies_properties = properties.copy()
    dependencies_properties.pop('composite', None)

    self.w1 = w1(
      shape=shape1,
      **dependencies_properties,
      name=(name + '_w1') if name is not None else None
    )

    self.w2 = w2(
      shape=shape2,
      **dependencies_properties,
      name=(name + '_w2') if name is not None else None
    )

    super(DecompositionParameter, self).__init__(
      self.w1, self.w2,
      shape=shape, name=name,
      **properties
    )

  def get_output_for(self, w1, w2):
    return tf.tensordot(w1, w2, axes=[(1,), (0,)])

  def get_output_shape_for(self, w1_shape, w2_shape):
    assert w1_shape[1] == w2_shape[0], 'parameters shapes are not compatible'
    assert w1_shape[2:] == w2_shape[2:], 'parameters shapes are not compatible'
    return (w1_shape[0], w2_shape[1]) + w1_shape[2:]

decomposition = parameter_model(DecompositionParameter, composite=True)()