import numpy as np
import tensorflow as tf

from .meta import as_free_parameter, parameter_model

__all__ = [
  'BinaryInit', 'binary_init'
]

def binary(shape, dtype='float32', name=None):
  if isinstance(dtype, tf.dtypes.DType):
    _dtype = dtype.as_numpy_dtype
  else:
    _dtype = dtype

  n, m = np.prod(shape[:-1]), shape[-1]
  result = np.ndarray(shape=(n, m), dtype=_dtype)

  k = int(np.ceil(np.log2(n)))

  seq = np.arange(n)
  step = 1
  for i in range(k):
    result[:, i] = seq % step
    step *= 2

  return tf.constant(result, dtype=dtype, name=name)

BinaryInit = as_free_parameter(binary)
binary_init = parameter_model(BinaryInit)()

