import tensorflow as tf
from .meta import as_free_parameter, parameter_model

__all__ = [
  'glorot_scaling',
  'GlorotNormalInit', 'glorot_normal_init',
  'GlorotUniformInit', 'glorot_uniform_init',
  'GlorotNormalDoubleInit', 'glorot_normal_double_init',
]

def glorot_scaling(shape, gain=1.0):
  in_units, out_units = shape[-2:]
  receptive_field_area = tf.cast(tf.reduce_prod(shape[:-2]), tf.float32)

  return gain * tf.sqrt(2.0 / (in_units + out_units) / receptive_field_area)


def _glorot_normal(shape, gain=1.0, dtype=tf.float32, name=None):
  if len(shape) < 2:
    return tf.random.normal(shape=shape, mean=0.0, stddev=gain, dtype=dtype, name=name)
  else:
    scale = glorot_scaling(shape, gain)
    return tf.random.normal(shape=shape, mean=0.0, stddev=scale, dtype=dtype, name=name)

def _double_glorot_normal(shape, target_shape, gain=1.0, dtype=tf.float32, name=None):
  gain = glorot_scaling(target_shape, gain=gain)
  return _glorot_normal(shape, gain=gain, dtype=dtype, name=name)


GlorotNormalInit = as_free_parameter(_glorot_normal)
glorot_normal_init = parameter_model(GlorotNormalInit)()

GlorotNormalDoubleInit = as_free_parameter(_double_glorot_normal)
glorot_normal_double_init = parameter_model(GlorotNormalDoubleInit)()


def _glorot_uniform(shape, gain=1.0, dtype=tf.float32, name=None):
  if len(shape) < 2:
    return tf.random.normal(shape=shape, mean=0.0, stddev=gain, dtype=dtype, name=name)
  else:
    scale = tf.constant(
      glorot_scaling(shape, gain) * tf.sqrt(3.0),
      dtype=dtype
    )

    return tf.random.uniform(shape=shape, minval=-scale, maxval=scale, dtype=dtype, name=name)

GlorotUniformInit = as_free_parameter(_glorot_uniform)
glorot_uniform_init = parameter_model(GlorotUniformInit)()