import tensorflow as tf
from .meta import Objective

__all__ = [
  'l2_reg',
  'l1_reg',

  'output_gaussian_normalization'
]

class l2_reg(Objective):
  def __init__(self, variables, c, mean=False):
    """
    :param variables: list of variables to apply l2 regularization to;
    :param c: regularization coefficient;
    :param mean: if True, average penalty within each variable.
    """
    self.variables = variables
    self.c = tf.constant(c, dtype=tf.float32)

    self.reduce = tf.reduce_mean if mean else tf.reduce_sum

  def __call__(self):
    return self.c * sum(
      self.reduce(W ** 2)
      for W in self.variables
    )

class l1_reg(Objective):
  def __init__(self, variables, c, mean=False):
    """
    :param variables: list of variables to apply l2 regularization to;
    :param c: regularization coefficient;
    :param mean: if True, average penalty within each variable.
    """
    self.variables = variables
    self.c = tf.constant(c, dtype=tf.float32)

    self.reduce = tf.reduce_mean if mean else tf.reduce_sum

  def __call__(self):
    return self.c * sum(
      self.reduce(tf.abs(W))
      for W in self.variables
    )

class output_gaussian_normalization(Objective):
  def __call__(self, *outputs):
    means = [
      tf.reduce_mean(out)
      for out in outputs
    ]

    stds = [
      tf.sqrt(tf.reduce_mean((out - mean) ** 2))
      for out, mean in zip(outputs, means)
    ]

    kl = sum(
      mean ** 2 + std ** 2 - 2 * tf.math.log(std)
      for mean, std in zip(means, stds)
    )

    return kl