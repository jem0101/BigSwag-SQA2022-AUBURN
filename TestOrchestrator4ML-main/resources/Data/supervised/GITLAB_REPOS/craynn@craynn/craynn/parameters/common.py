import tensorflow as tf

from .meta import *

__all__ = [
  'ZerosInit', 'zeros_init',
  'OnesInit', 'ones_init',
  'ConstInit', 'const_init',
  'NormalInit', 'normal_init',
  'UniformInit', 'uniform_init',
]

ZerosInit = as_free_parameter(tf.zeros)
zeros_init = parameter_model(ZerosInit)()

OnesInit = as_free_parameter(tf.ones)
ones_init = parameter_model(OnesInit)()


def _const(shape, value, dtype=tf.float32, name=None):
  if hasattr(value, 'shape') and len(value.shape) > 0:
    assert value.shape == shape, \
      'If `value` is a non-scalar array then `shape` (%s) must be equal to `value.shape` (%s)' % (shape, value.shape)

    return tf.constant(value, dtype=dtype, name=name)
  else:
    return tf.constant(
      tf.fill(dims=shape, value=tf.constant(value, dtype=dtype), name=name),
      dtype=dtype, name=name
    )

ConstInit = as_free_parameter(_const)
const_init = parameter_model(ConstInit)()

NormalInit = as_free_parameter(tf.random.normal)
normal_init = parameter_model(NormalInit)()

UniformInit = as_free_parameter(tf.random.uniform)
uniform_init = parameter_model(UniformInit)()