import numpy as np
import tensorflow as tf

__all__ = [
  'normalize_variable_spec',
  'make_vars'
]

def _norm_spec(spec):
  if isinstance(spec, tuple):
    if len(spec) == 2 and (isinstance(spec[1], str) or isinstance(spec[1], tf.dtypes.DType)):
      return spec
    else:
      if not all([ isinstance(s, int) for s in spec ]):
        raise ValueError(
          'variable spec should be either an instance of tf.Variable, '
          'a tuple of integers (dtype defaults to float32), or shape-dtype pair, not %s' % (spec, )
        )

      return (spec, tf.float32)
  elif isinstance(spec, tf.Variable):
    return (
      tuple(spec.shape.as_list()),
      spec.dtype
    )

def normalize_variable_spec(var_spec):
  return [
    _norm_spec(spec) for spec in var_spec
  ]

def make_vars(var_spec, initial_value=None, trainable=False):
  var_spec = normalize_variable_spec(var_spec)

  if initial_value is None:
    initial_value_f = lambda shape, dtype: tf.zeros(shape=shape, dtype=dtype)
  elif not callable(initial_value):
    initial_value_f = lambda shape, dtype: tf.ones(shape=shape, dtype=dtype) * initial_value
  else:
    initial_value_f = initial_value

  return [
    tf.Variable(initial_value=initial_value_f(shape, dtype=dtype), dtype=dtype, trainable=trainable)
    for (shape, dtype) in var_spec
  ]