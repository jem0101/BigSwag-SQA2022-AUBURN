### inspired by https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py#L327-L367
### Copyright (c) 2014-2015 Lasagne contributors, (c) 2020 Maxim Borisyak

import numpy as np
import tensorflow as tf

from .meta import as_free_parameter, parameter_model

__all__ = [
  'orthogonal_init'
]

def _orthogonal(shape, gain=1.0):
  matrix_shape = (shape[0], np.prod(shape[1:]))

  R = tf.random.normal(shape=matrix_shape, dtype=tf.float32)
  _, u, v = tf.linalg.svd(R, full_matrices=True)
  W = (u if matrix_shape[0] > matrix_shape[1] else v) * gain

  return tf.reshape(W, shape)

OrthogonalInit = as_free_parameter(_orthogonal)
orthogonal_init = parameter_model(OrthogonalInit)()

